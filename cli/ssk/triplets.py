"""Generate training triplets for student distillation.

Strategy:

1. **Seed queries** — loaded from a ``queries.json`` file (benchmark test cases).
   Each seed is optionally expanded with deterministic paraphrase templates.
2. **Teacher scoring** — a large teacher model scores every query against all
   corpus passages to produce a soft supervision signal.
3. **Hard-negative mining** — for each query:
   - top-1 positive (highest teacher score that is actually relevant)
   - top-N hard negatives (high teacher score but NOT relevant)
4. **Output** — three JSON files:
   - ``triplets_train.json`` / ``triplets_val.json`` — for MNRL loss
   - ``soft_scores.json`` — for MarginMSE loss (teacher soft scores)

Typical usage::

    from ssk.triplets import generate_triplets
    from ssk.schema import CorpusFile, QueriesFile

    corpus = CorpusFile.load("corpus.json")
    queries = QueriesFile.load("queries.json")
    generate_triplets(corpus, queries, output_dir="./output")
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console
from rich.progress import track

from ssk.passage import make_passages
from ssk.schema import CorpusFile, QueriesFile

console = Console()

# ---------------------------------------------------------------------------
# Query prefix helpers (E5 family)
# ---------------------------------------------------------------------------

_E5_QUERY_PREFIX = "query: "
_E5_INSTRUCT_PREFIX = (
    "Instruct: Given a search query, retrieve the most relevant document\n"
    "Query: "
)


def get_query_prefix(model_name: str) -> str:
    """Return the appropriate query prefix for a given model."""
    if "instruct" in model_name.lower():
        return _E5_INSTRUCT_PREFIX
    if "e5" in model_name.lower():
        return _E5_QUERY_PREFIX
    return ""


# ---------------------------------------------------------------------------
# Paraphrase expansion
# ---------------------------------------------------------------------------

_EN_TEMPLATES = [
    "{q}",
    "find {q}",
    "search for {q}",
    "looking for {q}",
    "show me {q}",
]

_DE_TEMPLATES = [
    "{q}",
    "suche {q}",
    "finde {q}",
    "zeig mir {q}",
    "ich suche {q}",
]

_DE_CHARS = set("äöüÄÖÜß")


def expand_query(query: str) -> list[str]:
    """Return up to 5 variants of a query (including the original)."""
    is_de = any(c in _DE_CHARS for c in query)
    templates = _DE_TEMPLATES if is_de else _EN_TEMPLATES
    variants: list[str] = []
    for t in templates:
        v = t.format(q=query).strip()
        if v not in variants:
            variants.append(v)
    return variants


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

def _is_relevant(title: str, expected: list[str]) -> bool:
    """True if *title* contains any expected substring (case-insensitive)."""
    lower = title.lower()
    return any(e.lower() in lower for e in expected)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_triplets(
    corpus: CorpusFile,
    queries: QueriesFile,
    output_dir: str | Path,
    *,
    teacher_model: str = "intfloat/multilingual-e5-large-instruct",
    n_hard_neg: int = 3,
    expand: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
    batch_size: int = 16,
) -> Path:
    """Generate training triplets and soft scores.

    Args:
        corpus: Validated corpus file.
        queries: Validated queries file with ground-truth expectations.
        output_dir: Directory to write output JSON files.
        teacher_model: HuggingFace model name for the teacher encoder.
        n_hard_neg: Number of hard negatives per query.
        expand: Whether to expand queries with paraphrase templates.
        val_split: Fraction of triplets reserved for validation.
        seed: Random seed for shuffling and splitting.
        batch_size: Batch size for encoding passages.

    Returns:
        Path to the output directory.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = list(corpus)
    titles = [item.title for item in items]
    passages = make_passages(items)

    console.print(f"[bold]Corpus:[/bold] {len(items)} items")

    # -- Load teacher --
    console.print(f"[bold]Loading teacher:[/bold] {teacher_model}")
    query_prefix = get_query_prefix(teacher_model)
    teacher = SentenceTransformer(teacher_model)

    # -- Encode all passages --
    console.print("Encoding passages with teacher...")
    doc_embs = teacher.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    console.print(f"  shape: {doc_embs.shape}")

    # -- Build query set --
    seeds: list[tuple[str, list[str]]] = [
        (qc.query, qc.expected) for qc in queries
    ]

    if expand:
        all_queries: list[tuple[str, list[str]]] = []
        for q, exp in seeds:
            for variant in expand_query(q):
                all_queries.append((variant, exp))
        # Deduplicate
        seen: set[str] = set()
        deduped: list[tuple[str, list[str]]] = []
        for q, exp in all_queries:
            if q not in seen:
                seen.add(q)
                deduped.append((q, exp))
        all_queries = deduped
    else:
        all_queries = seeds

    console.print(
        f"[bold]Query variants:[/bold] {len(all_queries)} "
        f"(seeds={len(seeds)}, expand={'on' if expand else 'off'})"
    )

    # -- Mine triplets --
    triplets: list[dict[str, str]] = []
    soft_rows: list[dict] = []
    skipped = 0

    for query, expected in track(all_queries, description="Mining triplets..."):
        q_text = query_prefix + query
        q_emb = teacher.encode([q_text], normalize_embeddings=True)[0]
        scores: np.ndarray = doc_embs @ q_emb

        ranked_idx = np.argsort(scores)[::-1]

        # Find first relevant positive in top-20
        positive_pass: str | None = None
        for i in ranked_idx[:20]:
            if _is_relevant(titles[i], expected):
                positive_pass = passages[i]
                break

        if positive_pass is None:
            skipped += 1
            continue

        # Hard negatives: high-scoring but NOT relevant
        hard_negs: list[str] = []
        for i in ranked_idx:
            if len(hard_negs) >= n_hard_neg:
                break
            if not _is_relevant(titles[i], expected):
                hard_negs.append(passages[i])

        if not hard_negs:
            skipped += 1
            continue

        # Triplets (one per hard negative)
        for neg in hard_negs:
            triplets.append({
                "query": query,
                "positive": positive_pass,
                "negative": neg,
            })

        # Soft scores row (top-20 passages + teacher scores)
        top20_idx = ranked_idx[:20].tolist()
        soft_rows.append({
            "query": query,
            "passages": [passages[i] for i in top20_idx],
            "scores": [float(scores[i]) for i in top20_idx],
        })

    console.print(
        f"\n[bold green]Mined {len(triplets)} triplets[/bold green] "
        f"from {len(all_queries)} queries ({skipped} skipped)"
    )

    # -- Save --
    rng = random.Random(seed)
    rng.shuffle(triplets)

    split_idx = int(len(triplets) * (1.0 - val_split))
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]

    train_path = out / "triplets_train.json"
    val_path = out / "triplets_val.json"
    soft_path = out / "soft_scores.json"

    for path, data in [
        (train_path, train_triplets),
        (val_path, val_triplets),
        (soft_path, soft_rows),
    ]:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"  Saved {path.name} ({len(data)} entries)")

    console.print(
        f"\n[bold]Split:[/bold] {len(train_triplets)} train / "
        f"{len(val_triplets)} val"
    )

    return out
