"""Distill a student embedding model from a teacher using dual losses.

Two losses are combined:

1. **MultipleNegativesRankingLoss (MNRL)** — in-batch negatives drive the
   student to rank positives higher than random distractors.
2. **MarginMSELoss** — teacher soft scores teach the student the *margin*
   between a positive and each hard negative, preserving teacher geometry.

Architecture:
    - Teacher: large instruct model (frozen, used only for triplet scoring)
    - Student: small E5 model (fine-tuned)

Typical usage::

    from ssk.train import train_student

    train_student(
        data_dir="./output",
        output_dir="./output",
        student_model="intfloat/multilingual-e5-small",
    )
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from rich.console import Console

from ssk.metrics import aggregate

console = Console()

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_STUDENT = "intfloat/multilingual-e5-small"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_triplets(path: Path) -> list:
    from sentence_transformers import InputExample

    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        InputExample(texts=[r["query"], r["positive"], r["negative"]])
        for r in rows
    ]


def _load_mnrl_examples(triplets: list) -> list:
    """Convert triplets to deduplicated (query, positive) pairs for MNRL."""
    from sentence_transformers import InputExample

    seen: set[tuple[str, str]] = set()
    out: list = []
    for ex in triplets:
        key = (ex.texts[0], ex.texts[1])
        if key not in seen:
            seen.add(key)
            out.append(InputExample(texts=[ex.texts[0], ex.texts[1]]))
    return out


def _load_margin_examples(soft_path: Path) -> list:
    """Build MarginMSE examples from teacher soft scores.

    Format: ``InputExample(texts=[query, pos, neg], label=score_pos - score_neg)``
    """
    from sentence_transformers import InputExample

    rows = json.loads(soft_path.read_text(encoding="utf-8"))
    examples: list = []
    for row in rows:
        query = row["query"]
        passages = row["passages"]
        scores = row["scores"]
        if len(passages) < 2:
            continue
        pos_pass = passages[0]
        pos_score = scores[0]
        for neg_pass, neg_score in zip(passages[1:4], scores[1:4]):
            margin = float(pos_score - neg_score)
            examples.append(
                InputExample(texts=[query, pos_pass, neg_pass], label=margin)
            )
    return examples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_student(
    model: object,
    passages: list[str],
    titles: list[str],
    test_cases: list[tuple[str, list[str]]],
    query_prefix: str = "query: ",
    k: int = 10,
) -> dict[str, float]:
    """Compute retrieval metrics on a benchmark test set."""
    import numpy as np

    doc_embs = model.encode(  # type: ignore[union-attr]
        passages,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    batch_results: list[tuple[list[str], list[str]]] = []
    for query, expected in test_cases:
        q_vec = model.encode(  # type: ignore[union-attr]
            [query_prefix + query], normalize_embeddings=True
        )[0]
        scores = doc_embs @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]
        top_titles = [titles[i] for i in top_idx]
        batch_results.append((top_titles, expected))

    agg = aggregate(batch_results)
    return {
        "hit": agg.hit,
        "recall": agg.recall,
        "mrr": agg.mrr,
        "ndcg": agg.ndcg,
        "gate": agg.gate,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_student(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    student_model: str = DEFAULT_STUDENT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    resume_from: str | Path | None = None,
    passages: list[str] | None = None,
    titles: list[str] | None = None,
    test_cases: list[tuple[str, list[str]]] | None = None,
    smoke_test: bool = False,
) -> Path:
    """Train a student model via knowledge distillation.

    Args:
        data_dir: Directory containing triplets and soft scores from
            :func:`ssk.triplets.generate_triplets`.
        output_dir: Directory for saving the trained model and metrics.
        student_model: HuggingFace model name for the student.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        warmup_ratio: Fraction of total steps for warmup.
        weight_decay: AdamW weight decay.
        resume_from: Path to a checkpoint to resume from.
        passages: Pre-built passage strings for evaluation (optional).
        titles: Corpus titles for evaluation (optional).
        test_cases: ``(query, expected)`` pairs for evaluation (optional).
        smoke_test: If True, use tiny data subset for pipeline verification.

    Returns:
        Path to the best model directory.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    student_out = out / "student_model"
    best_out = out / "student_model_best"

    # -- Load data --
    train_path = data / "triplets_train.json"
    val_path = data / "triplets_val.json"
    soft_path = data / "soft_scores.json"

    for p in [train_path, val_path, soft_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `ssk triplets` first to generate training data."
            )

    train_triplets = _load_triplets(train_path)

    if smoke_test:
        train_triplets = train_triplets[:64]
        console.print("[yellow]SMOKE TEST MODE — using 64 train examples[/yellow]")

    mnrl_examples = _load_mnrl_examples(train_triplets)
    margin_examples = _load_margin_examples(soft_path)

    if smoke_test:
        margin_examples = margin_examples[:64]

    console.print(f"[bold]MNRL pairs:[/bold]    {len(mnrl_examples)}")
    console.print(f"[bold]MarginMSE ex:[/bold]  {len(margin_examples)}")

    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    # -- Load student --
    if resume_from and Path(resume_from).exists():
        console.print(f"Resuming from {resume_from}")
        student = SentenceTransformer(str(resume_from))
    else:
        console.print(f"[bold]Loading student:[/bold] {student_model}")
        student = SentenceTransformer(student_model)

    # -- Losses --
    mnrl_loss = losses.MultipleNegativesRankingLoss(student)
    margin_loss = losses.MarginMSELoss(student)

    mnrl_loader = DataLoader(
        mnrl_examples, shuffle=True, batch_size=batch_size, drop_last=True
    )
    margin_loader = DataLoader(
        margin_examples, shuffle=True, batch_size=batch_size, drop_last=False
    )

    # -- Baseline benchmark (optional) --
    baseline: dict[str, float] | None = None
    if passages and titles and test_cases:
        console.print("\n[bold]Baseline student metrics:[/bold]")
        baseline = _evaluate_student(student, passages, titles, test_cases)
        _print_metrics(baseline)

    # -- Training --
    warmup_steps = math.ceil(len(mnrl_loader) * epochs * warmup_ratio)
    console.print(
        f"\n[bold]Training:[/bold] epochs={epochs}  batch={batch_size}  "
        f"lr={lr}  warmup={warmup_steps}"
    )

    student.fit(
        train_objectives=[
            (mnrl_loader, mnrl_loss),
            (margin_loader, margin_loss),
        ],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr, "weight_decay": weight_decay},
        show_progress_bar=True,
        output_path=str(student_out),
        save_best_model=False,
        checkpoint_path=None,
        use_amp=torch.cuda.is_available(),
    )

    # -- Post-training benchmark (optional) --
    final: dict[str, float] | None = None
    gate = 0.0
    if passages and titles and test_cases:
        console.print("\n[bold]Final student metrics:[/bold]")
        final = _evaluate_student(student, passages, titles, test_cases)
        _print_metrics(final)
        gate = final["gate"]
        console.print(f"[bold]Gate score:[/bold] {gate:.4f}")

    # Always save best model
    student.save(str(best_out))
    console.print(f"[bold green]Best model saved:[/bold green] {best_out}")

    # -- Save metrics --
    result = {
        "student": student_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "baseline": baseline,
        "final": final,
        "gate": gate,
    }
    metrics_path = out / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"Training metrics saved: {metrics_path}")

    return best_out


def _print_metrics(m: dict[str, float]) -> None:
    """Print retrieval metrics in a compact single-line format."""
    console.print(
        f"  hit={m['hit']:.3f}  recall={m['recall']:.3f}  "
        f"mrr={m['mrr']:.3f}  ndcg={m['ndcg']:.3f}  gate={m['gate']:.3f}"
    )
