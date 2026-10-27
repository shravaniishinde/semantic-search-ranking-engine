"""Backend validation demo: runs representative queries through the full pipeline.

python scripts/demo_search.py [--query "your query"] [--top-k 5]
"""

import argparse

import _bootstrap  # noqa: F401

from app.config.settings import get_settings
from app.logging_config import configure_logging
from app.services.search_service import SearchEngine

DEMO_QUERIES = [
    "security architecture for AWS",  # entity + topical
    "stop attackers from guessing user passwords",  # paraphrase: semantic should beat BM25
    "CVE-2021-44228",  # exact identifier: BM25 territory
    "why is my kubernetes pod restarting repeatedly",  # troubleshooting
    "compare TCP and UDP",  # comparison
]


def _title(engine: SearchEngine, doc_id: str, width: int = 52) -> str:
    title = engine.documents[doc_id].title
    return title if len(title) <= width else title[: width - 3] + "..."


def show(engine: SearchEngine, query: str, top_k: int) -> None:
    r = engine.search(query, top_k=top_k)
    print("=" * 100)
    print(f"Query            : {r.query}")
    print(f"Predicted intent : {r.intent.label} (confidence {r.intent.confidence:.2f}, {r.intent.source})")
    print(f"Entities         : {', '.join(f'{e.text} [{e.label}]' for e in r.entities) or '(none)'}")
    print(f"Latency          : {r.latency_ms:.1f} ms  | candidates: {r.candidate_count} | ranker: {r.ranker}")

    print("\n  Top BM25 results" + f"  [{r.retrieval.bm25_backend}]")
    for h in r.retrieval.bm25[:3]:
        print(f"    {h.rank}. {_title(engine, h.document_id):<55} bm25={h.score:.3f}")
    if not r.retrieval.bm25:
        print("    (no lexical matches)")
    print("  Top semantic results")
    for h in r.retrieval.semantic[:3]:
        print(f"    {h.rank}. {_title(engine, h.document_id):<55} cos={h.score:.3f}")
    print("  Top hybrid (RRF) results")
    for i, h in enumerate(r.retrieval.hybrid[:3], start=1):
        print(f"    {i}. {_title(engine, h.document_id):<55} rrf={h.hybrid_score:.4f}")

    print("\n  Final reranked results")
    print(f"    {'Rank':<5}{'Title':<55}{'Final':>9}{'BM25':>9}{'Semantic':>10}{'Hybrid':>9}")
    for item in r.results:
        bm25 = f"{item.bm25_score:.3f}" if item.bm25_score is not None else "-"
        sem = f"{item.semantic_score:.3f}" if item.semantic_score is not None else "-"
        print(
            f"    {item.rank:<5}{_title(engine, item.document_id):<55}{item.final_score:>9.3f}{bm25:>9}{sem:>10}"
            f"{item.hybrid_score:>9.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="query to run (repeatable); defaults to demo queries")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    configure_logging("WARNING")
    engine = SearchEngine(get_settings())
    engine.bootstrap()
    a = engine.analytics()
    print(
        f"Loaded {a['document_count']} documents ({a['indexed_count']} indexed, {a['excluded_count']} excluded) | "
        f"BM25 backend: {a['bm25_backend']} | clusters: {a['cluster_count']} | outliers: {a['outlier_count']} | "
        f"confirmed duplicates: {a['duplicate_count']}"
    )
    models = {k: v for k, v in a["models"].items() if k != "reranker_feature_importance"}
    print(f"Models: {models}\n")
    for query in args.query or DEMO_QUERIES:
        show(engine, query, args.top_k)


if __name__ == "__main__":
    main()
