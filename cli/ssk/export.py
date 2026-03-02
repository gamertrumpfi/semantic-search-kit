"""Export a trained student model to CoreML format.

The exported ``.mlpackage``:

- Accepts tokenised input (``input_ids`` + ``attention_mask``)
- Returns a single L2-normalised embedding vector
- Targets iOS 16+ / macOS 13+ Neural Engine via ``ComputeUnit.ALL``
- Optionally quantised to fp16 to halve on-device memory

Typical usage::

    from ssk.export import export_coreml

    export_coreml(
        model_path="./output/student_model_best",
        output_path="./output/MySearch.mlpackage",
        name="MySearch",
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sentence_transformers import SentenceTransformer

console = Console()

DEFAULT_MAX_SEQ_LEN = 64


# ---------------------------------------------------------------------------
# Wrapper: HF BERT -> mean-pooled + L2-normalised embedding
# ---------------------------------------------------------------------------

class EmbeddingWrapper(nn.Module):
    """Wraps a HuggingFace BERT-family model with mean pooling + L2 normalisation.

    This matches sentence-transformers' ``model.encode()`` behaviour, making
    the CoreML model a drop-in replacement.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return nn.functional.normalize(pooled, p=2, dim=1)


# ---------------------------------------------------------------------------
# Parity verification
# ---------------------------------------------------------------------------

def _verify(
    coreml_model: object,
    st_model: SentenceTransformer,
    tokenizer: object,
    queries: list[str],
    max_seq_len: int,
    query_prefix: str,
) -> None:
    """Check that PyTorch and CoreML outputs match within tolerance."""
    console.print("\n[bold]Parity check (PyTorch vs CoreML):[/bold]")
    max_err = 0.0
    for q in queries:
        enc = tokenizer(
            query_prefix + q,
            return_tensors="pt",
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
        )
        pt_emb = st_model.encode(
            [query_prefix + q], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        preds = coreml_model.predict({  # type: ignore[union-attr]
            "input_ids": enc["input_ids"].numpy().astype(np.int32),
            "attention_mask": enc["attention_mask"].numpy().astype(np.int32),
        })
        ml_emb = list(preds.values())[0][0]
        err = float(np.abs(pt_emb - ml_emb).max())
        max_err = max(max_err, err)
        console.print(f"  query: '{q[:50]}'  max_abs_err={err:.6f}")

    status = "[green]PASS[/green]" if max_err < 1e-3 else "[yellow]WARN[/yellow]"
    console.print(f"  max error: {max_err:.6f}  [{status}]")


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_coreml(
    model_path: str | Path,
    output_path: str | Path,
    name: str,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    quantize: bool = True,
    verify: bool = False,
    query_prefix: str = "query: ",
) -> Path:
    """Export a SentenceTransformer model to CoreML.

    Args:
        model_path: Path to the trained model (or HuggingFace name).
        output_path: Output ``.mlpackage`` path.
        name: Human-readable model name for metadata.
        max_seq_len: Maximum token sequence length.
        quantize: Whether to quantise weights to fp16.
        verify: Run parity check between PyTorch and CoreML.
        query_prefix: Query prefix for parity check queries.

    Returns:
        Path to the saved ``.mlpackage``.
    """
    try:
        import coremltools as ct
    except ImportError as e:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install it with: pip install coremltools"
        ) from e

    out = Path(output_path)

    console.print(f"[bold]Loading model:[/bold] {model_path}")
    st_model = SentenceTransformer(str(model_path), device="cpu")
    transformer = st_model[0].auto_model
    tokenizer = st_model[0].tokenizer
    transformer.eval().to("cpu")

    embed_dim = st_model.get_sentence_embedding_dimension()
    console.print(f"  embedding dim: {embed_dim}")

    # -- Trace --
    console.print(f"Tracing (seq_len={max_seq_len})...")
    dummy = tokenizer(
        "query: sample search query",
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
    )
    dummy = {k: v.to("cpu") for k, v in dummy.items()}

    wrapper = EmbeddingWrapper(transformer)
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            strict=False,
        )

    # -- Convert --
    console.print("Converting to CoreML...")
    precision = ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids", shape=(1, max_seq_len), dtype=np.int32
            ),
            ct.TensorType(
                name="attention_mask", shape=(1, max_seq_len), dtype=np.int32
            ),
        ],
        outputs=[
            ct.TensorType(name="embedding", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
    )

    # -- Metadata --
    prec_str = "fp16" if quantize else "fp32"
    mlmodel.short_description = (
        f"{name} query encoder — {embed_dim}-dim, {prec_str}"
    )
    mlmodel.input_description["input_ids"] = f"Token IDs (1 x {max_seq_len})"
    mlmodel.input_description["attention_mask"] = (
        f"Attention mask (1 x {max_seq_len})"
    )
    mlmodel.output_description["embedding"] = (
        f"L2-normalised query embedding ({embed_dim}-dim)"
    )

    # -- Save --
    mlmodel.save(str(out))
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    console.print(
        f"\n[bold green]Saved:[/bold green] {out}  ({size_mb:.0f} MB, {prec_str})"
    )

    # -- Parity check --
    if verify:
        _verify(mlmodel, st_model, tokenizer, [
            "wireless headphones",
            "science fiction novel",
            "hiking trail map",
        ], max_seq_len, query_prefix)

    console.print(f"""
[bold]CoreML Export Complete[/bold]

  File : {out}
  Dim  : {embed_dim}
  Size : {size_mb:.0f} MB

  Xcode integration:
    1. Drag {out.name} into your Xcode project
    2. Call SemanticSearchKit.configure(modelName: "{out.stem}", bundle: .main)
    3. Run `ssk embed` to generate embeddings_index.json
""")

    return out
