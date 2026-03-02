"""Full pipeline orchestrator with Rich TUI progress.

Chains all five pipeline stages end-to-end:

1. **Baseline benchmark** — evaluate the teacher model (skipped if no queries)
2. **Triplet mining**     — generate hard-negative training data
3. **Student training**   — dual-loss knowledge distillation (MNRL + MarginMSE)
4. **CoreML export**      — trace → convert → fp16 quantise
5. **Embedding generation** — pre-compute bundle embeddings

Usage::

    from ssk.pipeline import run_pipeline

    run_pipeline(
        corpus_path="corpus.json",
        output_dir="./output",
        name="MySearch",
        queries_path="queries.json",  # optional
    )
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Stage result tracking
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Timing and status for a single pipeline stage."""

    name: str
    elapsed: float = 0.0
    status: str = "pending"  # pending | running | done | skipped | failed
    detail: str = ""


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _print_summary(stages: list[StageResult], total: float) -> None:
    """Print a final summary table of all stages."""
    table = Table(title="Pipeline Summary", show_lines=False, pad_edge=False)
    table.add_column("Stage", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Detail")

    status_style = {
        "done": "[green]done[/green]",
        "skipped": "[dim]skipped[/dim]",
        "failed": "[red]FAILED[/red]",
    }

    for s in stages:
        table.add_row(
            s.name,
            status_style.get(s.status, s.status),
            _format_elapsed(s.elapsed) if s.status == "done" else "—",
            s.detail,
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold]Total time:[/bold] {_format_elapsed(total)}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    corpus_path: str | Path,
    output_dir: str | Path,
    name: str,
    *,
    queries_path: str | Path | None = None,
    teacher_model: str = "intfloat/multilingual-e5-large-instruct",
    student_model: str = "intfloat/multilingual-e5-small",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_seq_len: int = 64,
    n_hard_neg: int = 3,
    quantize: bool = True,
    verify_export: bool = False,
    smoke_test: bool = False,
) -> Path:
    """Run the complete distillation pipeline.

    Args:
        corpus_path: Path to the corpus JSON file.
        output_dir: Base output directory for all artefacts.
        name: Model name for the exported CoreML package.
        queries_path: Optional path to benchmark queries for evaluation.
        teacher_model: HuggingFace model name for the teacher encoder.
        student_model: HuggingFace model name for the student base.
        epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        max_seq_len: Max token sequence length for CoreML export.
        n_hard_neg: Hard negatives per query during triplet mining.
        quantize: Quantise CoreML model to fp16.
        verify_export: Run PyTorch vs CoreML parity check after export.
        smoke_test: Use tiny data subsets for pipeline verification.

    Returns:
        Path to the output directory.

    Raises:
        FileNotFoundError: If corpus file does not exist.
        ValueError: If corpus or queries file is malformed.
    """
    from ssk.schema import CorpusFile, QueriesFile

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    has_queries = queries_path is not None and Path(queries_path).exists()

    # -- Banner --
    console.print(Panel.fit(
        f"[bold]Semantic Search Kit[/bold]  —  {name}\n"
        f"corpus: {corpus_path}\n"
        f"queries: {queries_path or '(none)'}\n"
        f"output: {out}",
        border_style="blue",
    ))

    # -- Load and validate inputs --
    console.print("\n[bold]Loading corpus...[/bold]")
    corpus = CorpusFile.load(corpus_path)
    console.print(f"  {len(corpus)} items loaded")

    queries: QueriesFile | None = None
    if has_queries:
        console.print("[bold]Loading queries...[/bold]")
        queries = QueriesFile.load(queries_path)  # type: ignore[arg-type]
        console.print(f"  {len(queries)} test cases loaded")

    # -- Prepare data for evaluation callbacks --
    from ssk.passage import make_passages

    items = list(corpus)
    passages = make_passages(items)
    titles = [item.title for item in items]
    test_cases: list[tuple[str, list[str]]] | None = None
    if queries:
        test_cases = [(qc.query, qc.expected) for qc in queries]

    # -- Define stages --
    stages = [
        StageResult(name="1. Baseline benchmark"),
        StageResult(name="2. Triplet mining"),
        StageResult(name="3. Student training"),
        StageResult(name="4. CoreML export"),
        StageResult(name="5. Embedding generation"),
    ]

    pipeline_t0 = time.time()

    # -----------------------------------------------------------------------
    # Stage 1: Baseline benchmark (teacher)
    # -----------------------------------------------------------------------
    stage = stages[0]
    if not has_queries or queries is None:
        stage.status = "skipped"
        stage.detail = "no queries file"
        console.print("\n[dim]Stage 1: Baseline benchmark — skipped (no queries)[/dim]")
    else:
        console.print("\n[bold blue]Stage 1: Baseline benchmark (teacher)[/bold blue]")
        stage.status = "running"
        t0 = time.time()
        try:
            from ssk.metrics import aggregate
            from ssk.triplets import get_query_prefix
            from sentence_transformers import SentenceTransformer

            teacher = SentenceTransformer(teacher_model)

            # Use the teacher's appropriate query prefix
            query_prefix = get_query_prefix(teacher_model)

            import numpy as np

            doc_embs = teacher.encode(
                passages,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

            batch_results: list[tuple[list[str], list[str]]] = []
            for query_text, expected in test_cases:  # type: ignore[union-attr]
                q_vec = teacher.encode(
                    [query_prefix + query_text],
                    normalize_embeddings=True,
                )[0]
                scores = doc_embs @ q_vec
                top_idx = np.argsort(scores)[::-1][:10]
                top_titles = [titles[i] for i in top_idx]
                batch_results.append((top_titles, expected))

            agg = aggregate(batch_results)
            stage.detail = (
                f"hit={agg.hit:.3f}  MRR={agg.mrr:.3f}  "
                f"gate={agg.gate:.3f}"
            )
            console.print(f"  {stage.detail}")

            # Free teacher memory
            del teacher, doc_embs
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
            stage.status = "failed"
            stage.detail = str(exc)
            console.print(f"[red]  FAILED: {exc}[/red]")
            console.print(traceback.format_exc(), style="dim")
        else:
            stage.elapsed = time.time() - t0
            stage.status = "done"

    # -----------------------------------------------------------------------
    # Stage 2: Triplet mining
    # -----------------------------------------------------------------------
    stage = stages[1]
    if not has_queries or queries is None:
        stage.status = "skipped"
        stage.detail = "no queries file"
        console.print("\n[dim]Stage 2: Triplet mining — skipped (no queries)[/dim]")
    else:
        console.print("\n[bold blue]Stage 2: Triplet mining[/bold blue]")
        stage.status = "running"
        t0 = time.time()
        try:
            from ssk.triplets import generate_triplets

            generate_triplets(
                corpus,
                queries,
                output_dir=out,
                teacher_model=teacher_model,
                n_hard_neg=n_hard_neg,
            )
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
            stage.status = "failed"
            stage.detail = str(exc)
            console.print(f"[red]  FAILED: {exc}[/red]")
            console.print(traceback.format_exc(), style="dim")
        else:
            stage.elapsed = time.time() - t0
            stage.status = "done"
            stage.detail = f"triplets saved to {out}"

    # -----------------------------------------------------------------------
    # Stage 3: Student training
    # -----------------------------------------------------------------------
    stage = stages[2]
    if stages[1].status not in ("done", "skipped"):
        stage.status = "skipped"
        stage.detail = "triplet mining failed"
        console.print(
            "\n[dim]Stage 3: Student training — skipped (no training data)[/dim]"
        )
    elif stages[1].status == "skipped":
        stage.status = "skipped"
        stage.detail = "no queries file"
        console.print(
            "\n[dim]Stage 3: Student training — skipped (no queries)[/dim]"
        )
    else:
        console.print("\n[bold blue]Stage 3: Student training[/bold blue]")
        stage.status = "running"
        t0 = time.time()
        try:
            from ssk.train import train_student

            train_student(
                data_dir=out,
                output_dir=out,
                student_model=student_model,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                passages=passages,
                titles=titles,
                test_cases=test_cases,
                smoke_test=smoke_test,
            )
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
            stage.status = "failed"
            stage.detail = str(exc)
            console.print(f"[red]  FAILED: {exc}[/red]")
            console.print(traceback.format_exc(), style="dim")
        else:
            stage.elapsed = time.time() - t0
            stage.status = "done"
            stage.detail = f"model saved to {out / 'student_model_best'}"

    # -----------------------------------------------------------------------
    # Stage 4: CoreML export
    # -----------------------------------------------------------------------
    stage = stages[3]

    # Determine which model to export
    best_model = out / "student_model_best"
    if not best_model.exists():
        # No trained model — export the base student directly
        export_source = student_model
    else:
        export_source = str(best_model)

    coreml_path = out / f"{name}.mlpackage"

    console.print("\n[bold blue]Stage 4: CoreML export[/bold blue]")
    stage.status = "running"
    t0 = time.time()
    try:
        from ssk.export import export_coreml

        export_coreml(
            model_path=export_source,
            output_path=coreml_path,
            name=name,
            max_seq_len=max_seq_len,
            quantize=quantize,
            verify=verify_export,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        stage.status = "failed"
        stage.detail = str(exc)
        console.print(f"[red]  FAILED: {exc}[/red]")
        console.print(traceback.format_exc(), style="dim")
    else:
        stage.elapsed = time.time() - t0
        stage.status = "done"
        stage.detail = f"{coreml_path.name}"

    # -----------------------------------------------------------------------
    # Stage 5: Embedding generation
    # -----------------------------------------------------------------------
    stage = stages[4]

    embed_model = str(best_model) if best_model.exists() else student_model

    console.print("\n[bold blue]Stage 5: Embedding generation[/bold blue]")
    stage.status = "running"
    t0 = time.time()
    try:
        from ssk.embeddings import generate_embeddings, generate_query_embeddings

        generate_embeddings(
            corpus,
            output_dir=out,
            model=embed_model,
            batch_size=16,
        )

        # Also generate query embeddings if queries are available
        if has_queries and queries_path:
            generate_query_embeddings(
                queries_path=queries_path,
                output_dir=out,
                model=embed_model,
            )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        stage.status = "failed"
        stage.detail = str(exc)
        console.print(f"[red]  FAILED: {exc}[/red]")
        console.print(traceback.format_exc(), style="dim")
    else:
        stage.elapsed = time.time() - t0
        stage.status = "done"
        stage.detail = "embeddings_index.json"

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = time.time() - pipeline_t0
    _print_summary(stages, total)

    # Final status
    failed = [s for s in stages if s.status == "failed"]
    if failed:
        console.print(
            f"\n[bold red]{len(failed)} stage(s) failed.[/bold red]  "
            f"Check output above for details."
        )
        sys.exit(1)
    else:
        console.print(
            f"\n[bold green]Pipeline complete![/bold green]  "
            f"Output: {out}"
        )

    return out
