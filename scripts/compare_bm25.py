"""Compare the in-memory BM25 fallback with real Elasticsearch BM25 on the same documents.

    docker compose up -d elasticsearch
    python scripts/compare_bm25.py [--url http://localhost:9200]

Indexes the current indexable documents into a throwaway ES index, runs the same queries through
both backends (with and without a category filter) and prints top results, overlap and scores.
Absolute scores are not expected to match (different analyzers); ranking behaviour should be
similar. Exits with status 2 if Elasticsearch is unreachable.
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from app.config.settings import get_settings
from app.ingestion.loader import load_documents
from app.logging_config import configure_logging
from app.models.schemas import SearchFilters
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from app.services.search_service import SearchEngine

QUERIES = [
    ("my kubernetes pod keeps restarting", None),
    ("cloud security architecture", None),
    ("database query performance", None),
    ("CVE-2021-44228", None),
    ("security", "networking"),
]
COMPARE_INDEX = "bm25-compare"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="Elasticsearch URL (default: ELASTICSEARCH_URL)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")
    es = ElasticsearchRetriever.from_url(args.url or settings.elasticsearch_url, COMPARE_INDEX, timeout=5.0)
    if not es.is_available():
        print(f"Elasticsearch is not reachable at {args.url or settings.elasticsearch_url}.")
        sys.exit(2)

    engine = SearchEngine(settings, lexical=InMemoryBM25Retriever())
    if not engine.load_state():
        docs, rejected = load_documents(settings.sample_documents_path)
        engine.ingest(docs, pre_rejected=rejected)
    indexable = [d for d in engine.documents.values() if d.indexed]
    memory = engine.lexical

    es.create_index(recreate=True)
    indexed = es.index_documents(indexable)
    print(f"Indexed {indexed} documents into '{COMPARE_INDEX}' (in-memory BM25 holds {memory.count()}).\n")

    total_overlap = 0
    for query, category in QUERIES:
        filters = SearchFilters(category=category) if category else None
        mem = memory.retrieve(query, args.top_k, filters)
        real = es.retrieve(query, args.top_k, filters)
        mem_ids, es_ids = [r.document_id for r in mem], [r.document_id for r in real]
        overlap = len(set(mem_ids) & set(es_ids))
        total_overlap += overlap
        label = f"{query!r}" + (f" [category={category}]" if category else "")
        print(
            f"{label}\n  top-1 agree: {bool(mem_ids and es_ids and mem_ids[0] == es_ids[0])}   "
            f"overlap@{args.top_k}: {overlap}/{max(len(mem_ids), len(es_ids), 1)}   "
            f"hits: memory={len(mem_ids)} es={len(es_ids)}"
        )
        for i in range(max(len(mem), len(real))):
            m = f"{mem[i].document_id:<12} {mem[i].score:7.3f}" if i < len(mem) else " " * 20
            e = f"{real[i].document_id:<12} {real[i].score:7.3f}" if i < len(real) else ""
            print(f"    {i + 1}. memory {m}   |   es {e}")
        if category:
            bad = [d for d in es_ids if engine.documents[d].category != category]
            print(f"  ES filter respected: {not bad}")
        print()
    es.client.indices.delete(index=COMPARE_INDEX, ignore_unavailable=True)
    print(f"Mean overlap@{args.top_k}: {total_overlap / len(QUERIES):.2f}")


if __name__ == "__main__":
    main()
