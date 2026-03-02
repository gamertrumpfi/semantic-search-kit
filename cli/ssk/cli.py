"""CLI entry point for Semantic Search Kit.

Provides the ``ssk`` command with subcommands for each pipeline stage:

- ``ssk run``       — full end-to-end pipeline
- ``ssk benchmark`` — evaluate a model on the test queries
- ``ssk triplets``  — generate training triplets from teacher
- ``ssk train``     — distil student from triplet data
- ``ssk export``    — convert trained model to CoreML
- ``ssk embed``     — generate bundle embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

console = Console()

app = typer.Typer(
    name="ssk",
    help="Semantic Search Kit — train and export on-device CoreML search models.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Common option aliases
# ---------------------------------------------------------------------------

CorpusOpt = Annotated[
    Path,
    typer.Option(
        "--corpus", "-c",
        help="Path to the corpus JSON file.",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
]

QueriesOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--queries", "-q",
        help="Path to the queries JSON file (optional).",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
]

OutputOpt = Annotated[
    Path,
    typer.Option(
        "--output", "-o",
        help="Output directory for generated artefacts.",
        resolve_path=True,
    ),
]

NameOpt = Annotated[
    str,
    typer.Option(
        "--name", "-n",
        help="Model name for the exported CoreML package.",
    ),
]


# ---------------------------------------------------------------------------
# ssk run — full pipeline
# ---------------------------------------------------------------------------

@app.command()
def run(
    corpus: CorpusOpt,
    name: NameOpt,
    output: OutputOpt = Path("./output"),
    queries: QueriesOpt = None,
    teacher: Annotated[
        str,
        typer.Option(help="HuggingFace teacher model name."),
    ] = "intfloat/multilingual-e5-large-instruct",
    student: Annotated[
        str,
        typer.Option(help="HuggingFace student model name."),
    ] = "intfloat/multilingual-e5-small",
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 10,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Training batch size."),
    ] = 32,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 2e-5,
    max_seq_len: Annotated[
        int,
        typer.Option("--max-seq-len", help="Max token sequence length for CoreML."),
    ] = 64,
    n_hard_neg: Annotated[
        int,
        typer.Option("--n-hard-neg", help="Hard negatives per query."),
    ] = 3,
    no_quantize: Annotated[
        bool,
        typer.Option("--no-quantize", help="Skip fp16 quantisation."),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Run PyTorch vs CoreML parity check."),
    ] = False,
    smoke_test: Annotated[
        bool,
        typer.Option("--smoke-test", help="Use tiny data subset for quick verification."),
    ] = False,
) -> None:
    """Run the full distillation pipeline end-to-end.

    Chains: baseline benchmark → triplet mining → student training →
    CoreML export → embedding generation.
    """
    from ssk.pipeline import run_pipeline

    run_pipeline(
        corpus_path=corpus,
        output_dir=output,
        name=name,
        queries_path=queries,
        teacher_model=teacher,
        student_model=student,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_seq_len=max_seq_len,
        n_hard_neg=n_hard_neg,
        quantize=not no_quantize,
        verify_export=verify,
        smoke_test=smoke_test,
    )


# ---------------------------------------------------------------------------
# ssk benchmark — evaluate a model
# ---------------------------------------------------------------------------

@app.command()
def benchmark(
    corpus: CorpusOpt,
    queries: Annotated[
        Path,
        typer.Option(
            "--queries", "-q",
            help="Path to the queries JSON file.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(help="SentenceTransformer model name or local path."),
    ] = "intfloat/multilingual-e5-small",
    k: Annotated[int, typer.Option(help="Rank cutoff for metrics.")] = 10,
    query_prefix: Annotated[
        str,
        typer.Option(
            "--query-prefix",
            help="Prefix prepended to queries. Auto-detected from model name if not set.",
        ),
    ] = "",
) -> None:
    """Evaluate a model on the benchmark queries.

    Computes hit@k, recall@k, MRR@k, nDCG@k, and gate score.

    The query prefix is auto-detected from the model name (e.g. "query: " for
    E5, instruct prefix for E5-instruct). Override with --query-prefix.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from ssk.metrics import aggregate
    from ssk.passage import make_passages
    from ssk.schema import CorpusFile, QueriesFile
    from ssk.triplets import get_query_prefix

    corpus_data = CorpusFile.load(corpus)
    queries_data = QueriesFile.load(queries)

    # Auto-detect query prefix from model name if not explicitly set
    effective_prefix = query_prefix if query_prefix else get_query_prefix(model)

    items = list(corpus_data)
    passages = make_passages(items)
    titles = [item.title for item in items]

    console.print(f"[bold]Model:[/bold]  {model}")
    console.print(f"[bold]Prefix:[/bold] {effective_prefix!r}")
    console.print(f"[bold]Corpus:[/bold] {len(items)} items")
    console.print(f"[bold]Queries:[/bold] {len(queries_data)} test cases")
    console.print(f"[bold]k:[/bold]       {k}\n")

    st_model = SentenceTransformer(model)
    doc_embs = st_model.encode(
        passages,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    batch_results: list[tuple[list[str], list[str]]] = []
    for qc in queries_data:
        q_vec = st_model.encode(
            [effective_prefix + qc.query], normalize_embeddings=True
        )[0]
        scores: np.ndarray = doc_embs @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]
        top_titles = [titles[i] for i in top_idx]
        batch_results.append((top_titles, qc.expected))

    agg = aggregate(batch_results)
    console.print(f"\n{agg}")


# ---------------------------------------------------------------------------
# ssk triplets — generate training data
# ---------------------------------------------------------------------------

@app.command()
def triplets(
    corpus: CorpusOpt,
    queries: Annotated[
        Path,
        typer.Option(
            "--queries", "-q",
            help="Path to the queries JSON file.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output: OutputOpt = Path("./output"),
    teacher: Annotated[
        str,
        typer.Option(help="HuggingFace teacher model name."),
    ] = "intfloat/multilingual-e5-large-instruct",
    n_hard_neg: Annotated[
        int,
        typer.Option("--n-hard-neg", help="Hard negatives per query."),
    ] = 3,
    no_expand: Annotated[
        bool,
        typer.Option("--no-expand", help="Disable paraphrase expansion."),
    ] = False,
) -> None:
    """Generate training triplets via teacher-scored hard-negative mining.

    Outputs: triplets_train.json, triplets_val.json, soft_scores.json.
    """
    from ssk.schema import CorpusFile, QueriesFile
    from ssk.triplets import generate_triplets

    corpus_data = CorpusFile.load(corpus)
    queries_data = QueriesFile.load(queries)

    generate_triplets(
        corpus_data,
        queries_data,
        output_dir=output,
        teacher_model=teacher,
        n_hard_neg=n_hard_neg,
        expand=not no_expand,
    )


# ---------------------------------------------------------------------------
# ssk train — student distillation
# ---------------------------------------------------------------------------

@app.command()
def train(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", "-d",
            help="Directory containing triplets and soft scores.",
            exists=True,
            resolve_path=True,
        ),
    ],
    output: OutputOpt = Path("./output"),
    corpus: Annotated[
        Optional[Path],
        typer.Option(
            "--corpus", "-c",
            help="Corpus file for evaluation during training.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    queries: QueriesOpt = None,
    student: Annotated[
        str,
        typer.Option(help="HuggingFace student model name."),
    ] = "intfloat/multilingual-e5-small",
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 10,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Training batch size."),
    ] = 32,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 2e-5,
    resume_from: Annotated[
        Optional[Path],
        typer.Option("--resume-from", help="Checkpoint to resume from."),
    ] = None,
    smoke_test: Annotated[
        bool,
        typer.Option("--smoke-test", help="Use tiny data subset."),
    ] = False,
) -> None:
    """Train a student model via MNRL + MarginMSE knowledge distillation.

    Requires triplet data from ``ssk triplets``.
    """
    from ssk.train import train_student

    # Optionally prepare evaluation data
    passages: list[str] | None = None
    titles: list[str] | None = None
    test_cases: list[tuple[str, list[str]]] | None = None

    if corpus and queries:
        from ssk.passage import make_passages
        from ssk.schema import CorpusFile, QueriesFile

        corpus_data = CorpusFile.load(corpus)
        queries_data = QueriesFile.load(queries)
        items = list(corpus_data)
        passages = make_passages(items)
        titles = [item.title for item in items]
        test_cases = [(qc.query, qc.expected) for qc in queries_data]

    train_student(
        data_dir=data_dir,
        output_dir=output,
        student_model=student,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resume_from=resume_from,
        passages=passages,
        titles=titles,
        test_cases=test_cases,
        smoke_test=smoke_test,
    )


# ---------------------------------------------------------------------------
# ssk export — CoreML conversion
# ---------------------------------------------------------------------------

@app.command()
def export(
    model_path: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="Trained model path or HuggingFace name.",
        ),
    ] = "intfloat/multilingual-e5-small",
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output .mlpackage path.",
            resolve_path=True,
        ),
    ] = Path("./output/SearchModel.mlpackage"),
    name: NameOpt = "SearchModel",
    max_seq_len: Annotated[
        int,
        typer.Option("--max-seq-len", help="Max token sequence length."),
    ] = 64,
    no_quantize: Annotated[
        bool,
        typer.Option("--no-quantize", help="Skip fp16 quantisation."),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Run PyTorch vs CoreML parity check."),
    ] = False,
) -> None:
    """Export a trained model to CoreML (.mlpackage).

    Produces an iOS 16+ / macOS 13+ compatible model with optional fp16 quantisation.
    """
    from ssk.export import export_coreml

    export_coreml(
        model_path=model_path,
        output_path=output,
        name=name,
        max_seq_len=max_seq_len,
        quantize=not no_quantize,
        verify=verify,
    )


# ---------------------------------------------------------------------------
# ssk embed — generate bundle embeddings
# ---------------------------------------------------------------------------

@app.command()
def embed(
    corpus: CorpusOpt,
    output: OutputOpt = Path("./output"),
    model: Annotated[
        str,
        typer.Option(help="SentenceTransformer model name or local path."),
    ] = "intfloat/multilingual-e5-small",
    queries: QueriesOpt = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Encoding batch size."),
    ] = 16,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Also write debug file with passage text."),
    ] = False,
) -> None:
    """Generate pre-computed embeddings for the app bundle.

    Encodes every corpus item and writes embeddings_index.json.
    Optionally also generates query_embeddings.json if queries are provided.
    """
    from ssk.embeddings import generate_embeddings, generate_query_embeddings
    from ssk.schema import CorpusFile

    corpus_data = CorpusFile.load(corpus)

    generate_embeddings(
        corpus_data,
        output_dir=output,
        model=model,
        batch_size=batch_size,
        debug=debug,
    )

    if queries:
        generate_query_embeddings(
            queries_path=queries,
            output_dir=output,
            model=model,
            batch_size=batch_size,
        )
