"""Generate pre-computed document embeddings for the iOS/macOS app bundle.

Encodes every corpus item into a fixed-dimension normalised embedding vector
and writes ``embeddings_index.json`` — the compact bundle file that
:class:`~SemanticSearchKit.EmbeddingStorage` loads at runtime.

Both a compact bundle file (id + embedding) and an optional debug file
(id + passage text + embedding) are produced.

Typical usage::

    from ssk.embeddings import generate_embeddings
    from ssk.schema import CorpusFile

    corpus = CorpusFile.load("corpus.json")
    generate_embeddings(corpus, output_dir="./output", model="./output/student_model_best")
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from sentence_transformers import SentenceTransformer

from ssk.passage import make_passages
from ssk.schema import CorpusFile

console = Console()


def generate_embeddings(
    corpus: CorpusFile,
    output_dir: str | Path,
    *,
    model: str = "intfloat/multilingual-e5-small",
    batch_size: int = 16,
    debug: bool = False,
) -> Path:
    """Generate document embeddings for the app bundle.

    Args:
        corpus: Validated corpus file.
        output_dir: Directory to write output files.
        model: SentenceTransformer model name or local path.
        batch_size: Encoding batch size.
        debug: If True, also write a debug file with passage text.

    Returns:
        Path to the ``embeddings_index.json`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = list(corpus)
    ids = [item.id for item in items]
    passages = make_passages(items)

    console.print(f"[bold]Corpus:[/bold] {len(items)} items")
    console.print(f"[bold]Model:[/bold]  {model}")

    if passages:
        console.print(f"\n[dim]Sample passage:[/dim]\n  {passages[0][:200]}...\n")

    # -- Load model + encode --
    console.print(f"Loading {model}...")
    st_model = SentenceTransformer(model)

    dim = st_model.get_sentence_embedding_dimension()
    console.print(f"  embedding dim: {dim}")

    t0 = time.time()
    console.print(f"Encoding {len(passages)} passages (batch={batch_size})...")
    embeddings = st_model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    console.print(f"Done in {elapsed:.1f}s  —  shape: {embeddings.shape}")

    # -- Build output --
    bundle = [
        {"id": rid, "embedding": emb.tolist()}
        for rid, emb in zip(ids, embeddings)
    ]

    # -- Write bundle --
    bundle_path = out / "embeddings_index.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    bundle_kb = bundle_path.stat().st_size / 1024
    console.print(
        f"\n[bold green]Bundle:[/bold green] {bundle_path.name}  "
        f"({bundle_kb:.0f} KB, dim={dim})"
    )

    # -- Optional debug file --
    if debug:
        debug_data = [
            {"id": rid, "text": txt, "embedding": emb.tolist()}
            for rid, txt, emb in zip(ids, passages, embeddings)
        ]
        debug_path = out / "embeddings_debug.json"
        debug_path.write_text(
            json.dumps(debug_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"[bold]Debug:[/bold]  {debug_path.name}")

    # -- Also generate query embeddings if a queries file exists nearby --
    # (This is handled by the pipeline orchestrator, not here.)

    console.print(f"\n[bold]Items:[/bold]  {len(bundle)}")
    console.print(f"[bold]Dim:[/bold]    {dim}")

    console.print(f"""
[bold]Next steps:[/bold]
  1. Add {bundle_path.name} to your Xcode project bundle resources
  2. Call SemanticSearchKit.configure(modelName: "YourModel", bundle: .main)
""")

    return bundle_path


def generate_query_embeddings(
    queries_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = "intfloat/multilingual-e5-small",
    query_prefix: str = "query: ",
    batch_size: int = 16,
) -> Path:
    """Generate pre-computed query embeddings for benchmark tests.

    Reads a ``queries.json`` file and encodes each query string into an
    embedding vector. The output is a JSON object mapping query strings
    to embedding vectors.

    Args:
        queries_path: Path to the queries JSON file.
        output_dir: Directory to write the output file.
        model: SentenceTransformer model name or local path.
        query_prefix: Prefix prepended to each query before encoding.
        batch_size: Encoding batch size.

    Returns:
        Path to the ``query_embeddings.json`` file.
    """
    from ssk.schema import QueriesFile

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    qfile = QueriesFile.load(queries_path)
    queries = [qc.query for qc in qfile]

    console.print(f"[bold]Queries:[/bold] {len(queries)}")
    console.print(f"[bold]Model:[/bold]   {model}")

    st_model = SentenceTransformer(model)
    dim = st_model.get_sentence_embedding_dimension()

    prefixed = [query_prefix + q for q in queries]
    embeddings = st_model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    result = {q: emb.tolist() for q, emb in zip(queries, embeddings)}

    out_path = out / "query_embeddings.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console.print(
        f"\n[bold green]Saved:[/bold green] {out_path.name}  "
        f"({len(result)} queries, dim={dim})"
    )

    return out_path
