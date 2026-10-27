"""Evaluate BM25 vs semantic vs hybrid vs hybrid+rerank on the held-out qrels.

    python scripts/evaluate.py
Writes data/processed/eval_report.json.
"""

import json

import _bootstrap  # noqa: F401

from app.config.settings import get_settings
from app.evaluation.evaluator import evaluate, format_table, load_qrels
from app.logging_config import configure_logging
from app.services.search_service import SearchEngine


def main() -> None:
    settings = get_settings()
    configure_logging("WARNING")
    engine = SearchEngine(settings)
    engine.bootstrap()
    qrels = load_qrels(settings.evaluation_qrels_path)
    report = evaluate(engine, qrels)
    print(
        f"Evaluation on {report['queries']} held-out queries "
        f"(bm25 backend: {report['bm25_backend']}, ranker: {report['ranker']})\n"
    )
    print(format_table(report))
    print("\nNote: tiny hand-labelled dataset; differences of a few points are not statistically meaningful.")
    out = settings.processed_dir / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
