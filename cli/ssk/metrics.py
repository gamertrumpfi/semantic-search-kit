"""Information retrieval metrics for evaluating search quality.

All metrics operate on a ranked list of result titles compared against a set
of expected title substrings.  A result is "relevant" if its title contains
any of the expected substrings (case-insensitive).

Supported metrics (all computed at rank cutoff *k*):

- **hit@k** — binary per query: 1 if *any* expected result appears in top-k.
- **recall@k** — fraction of expected items found in top-k.
- **MRR@k** — mean reciprocal rank of the first relevant hit.
- **nDCG@k** — normalised discounted cumulative gain (binary relevance).
- **gate score** — composite: ``0.5 * MRR@k + 0.5 * nDCG@k``.

Typical usage::

    from ssk.metrics import hit_at_k, aggregate

    hit = hit_at_k(["Result A", "Result B"], ["Result A"])
    agg = aggregate(batch_results, k=10)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

def relevant_mask(top_titles: list[str], expected: list[str]) -> list[bool]:
    """Boolean mask: ``True`` at position *i* if ``top_titles[i]`` is relevant.

    A title is relevant if it contains any expected substring (case-insensitive).
    """
    return [
        any(e.lower() in t.lower() for e in expected)
        for t in top_titles
    ]


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def hit_at_k(top_titles: list[str], expected: list[str]) -> int:
    """1 if any relevant result in the ranked list, else 0."""
    return int(any(relevant_mask(top_titles, expected)))


def recall_at_k(top_titles: list[str], expected: list[str]) -> float:
    """Fraction of expected items found in the ranked list.

    Each expected substring is counted as "found" if at least one title in
    the list contains it.
    """
    if not expected:
        return 0.0
    mask = relevant_mask(top_titles, expected)
    relevant_titles = [t.lower() for t, rel in zip(top_titles, mask) if rel]
    found = sum(
        1 for e in expected
        if any(e.lower() in rt for rt in relevant_titles)
    )
    return found / len(expected)


def mrr_at_k(top_titles: list[str], expected: list[str]) -> float:
    """Reciprocal rank of the first relevant result, or 0."""
    for i, rel in enumerate(relevant_mask(top_titles, expected)):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(top_titles: list[str], expected: list[str]) -> float:
    """Normalised DCG with binary relevance.

    ``DCG  = sum(1 / log2(i + 2))`` for each relevant position *i*.
    ``IDCG = sum(1 / log2(i + 2))`` for ``i`` in ``0 .. n_relevant - 1``.
    """
    mask = relevant_mask(top_titles, expected)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(mask) if r)
    n_rel = sum(mask)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg


def gate_score(mrr: float, ndcg: float) -> float:
    """Composite metric for model selection: ``0.5 * MRR + 0.5 * nDCG``."""
    return 0.5 * mrr + 0.5 * ndcg


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregateResult:
    """Aggregated metrics across a batch of queries."""

    hit: float
    recall: float
    mrr: float
    ndcg: float
    gate: float
    query_count: int

    def __str__(self) -> str:
        return (
            f"Aggregate ({self.query_count} queries):\n"
            f"  hit    = {self.hit:.3f}\n"
            f"  recall = {self.recall:.3f}\n"
            f"  MRR    = {self.mrr:.3f}\n"
            f"  nDCG   = {self.ndcg:.3f}\n"
            f"  gate   = {self.gate:.3f}"
        )


def aggregate(
    results: list[tuple[list[str], list[str]]],
) -> AggregateResult:
    """Compute aggregate metrics over a batch of ``(top_titles, expected)`` pairs."""
    n = len(results)
    if n == 0:
        return AggregateResult(hit=0, recall=0, mrr=0, ndcg=0, gate=0, query_count=0)

    sum_hit = sum_recall = sum_mrr = sum_ndcg = 0.0
    for top_titles, expected in results:
        sum_hit += hit_at_k(top_titles, expected)
        sum_recall += recall_at_k(top_titles, expected)
        sum_mrr += mrr_at_k(top_titles, expected)
        sum_ndcg += ndcg_at_k(top_titles, expected)

    avg_hit = sum_hit / n
    avg_recall = sum_recall / n
    avg_mrr = sum_mrr / n
    avg_ndcg = sum_ndcg / n

    return AggregateResult(
        hit=avg_hit,
        recall=avg_recall,
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        gate=gate_score(avg_mrr, avg_ndcg),
        query_count=n,
    )
