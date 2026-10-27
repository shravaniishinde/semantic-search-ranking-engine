"""Compare BM25, semantic, hybrid and hybrid+rerank on the held-out evaluation qrels."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.metrics import evaluate_run
from app.services.search_service import SearchEngine

SYSTEMS = ["bm25", "semantic", "hybrid", "hybrid+rerank"]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["query"]: item["relevance"] for item in data["queries"]}


def run_systems(engine: SearchEngine, queries: list[str], depth: int = 10) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[str]]] = {name: {} for name in SYSTEMS}
    for query in queries:
        response = engine.search(query, top_k=depth, include_retrieval=False, record_stats=False)
        bm25, semantic, hybrid = engine.hybrid.retrieve_all(query, depth)
        runs["bm25"][query] = [r.document_id for r in bm25[:depth]]
        runs["semantic"][query] = [r.document_id for r in semantic[:depth]]
        runs["hybrid"][query] = [c.document_id for c in hybrid[:depth]]
        runs["hybrid+rerank"][query] = [r.document_id for r in response.results]
    return runs


def evaluate(engine: SearchEngine, qrels: dict[str, dict[str, int]], ks: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
    runs = run_systems(engine, list(qrels), depth=max(ks))
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ks": list(ks),
        "queries": len(qrels),
        "ranker": engine.reranker.name,
        "bm25_backend": engine.bm25_backend,
        "metrics": {name: evaluate_run(run, qrels, ks) for name, run in runs.items()},
        "per_query_first_relevant_rank": {
            name: {q: next((i + 1 for i, d in enumerate(r) if qrels[q].get(d, 0) > 0), None) for q, r in run.items()}
            for name, run in runs.items()
        },
    }


def format_table(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    columns = list(next(iter(metrics.values())).keys())
    lines = [f"{'system':<15}" + "".join(f"{c:>9}" for c in columns)]
    lines.append("-" * len(lines[0]))
    for name, values in metrics.items():
        lines.append(f"{name:<15}" + "".join(f"{values[c]:>9.3f}" for c in columns))
    return "\n".join(lines)
