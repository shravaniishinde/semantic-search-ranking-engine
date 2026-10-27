"""Standard IR metrics over a ranked list of document ids and graded relevance judgments.

A document counts as relevant for binary metrics when its grade is > 0.
"""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if relevance.get(d, 0) > 0) / k


def recall_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = sum(1 for g in relevance.values() if g > 0)
    if relevant == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if relevance.get(d, 0) > 0) / relevant


def reciprocal_rank(ranked: list[str], relevance: dict[str, int]) -> float:
    for position, doc_id in enumerate(ranked, start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / position
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    ideal = dcg_at_k(sorted(relevance.values(), reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k([relevance.get(d, 0) for d in ranked], k) / ideal


def evaluate_run(runs: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """Mean metrics over queries. ``runs`` maps query -> ranked doc ids."""
    totals: dict[str, float] = {}
    queries = [q for q in qrels if q in runs]
    for query in queries:
        ranked, rel = runs[query], qrels[query]
        scores = {"MRR": reciprocal_rank(ranked, rel)}
        for k in ks:
            scores[f"P@{k}"] = precision_at_k(ranked, rel, k)
            scores[f"R@{k}"] = recall_at_k(ranked, rel, k)
            scores[f"NDCG@{k}"] = ndcg_at_k(ranked, rel, k)
        for name, value in scores.items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: round(value / len(queries), 4) for name, value in totals.items()} if queries else {}
