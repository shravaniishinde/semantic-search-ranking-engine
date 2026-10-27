"""Ingest documents (default: the bundled sample corpus) and build FAISS + BM25 indexes.

python scripts/build_index.py [--input path.jsonl] [--reset]
"""

import argparse
import shutil

import _bootstrap  # noqa: F401

from app.config.settings import get_settings
from app.ingestion.loader import load_documents
from app.logging_config import configure_logging
from app.services.search_service import SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON or JSONL file of documents")
    parser.add_argument("--reset", action="store_true", help="delete data/processed before ingesting")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")
    if args.reset and settings.processed_dir.exists():
        # Remove generated artifacts but keep the tracked .gitkeep placeholder.
        for child in settings.processed_dir.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    engine = SearchEngine(settings)
    engine.load_state()
    docs, rejected = load_documents(args.input or settings.sample_documents_path)
    report = engine.ingest(docs, pre_rejected=rejected)
    print(report.model_dump_json(indent=2))
    a = engine.analysis
    print(
        f"\nclusters={a.cluster_count} outliers={a.outlier_count} "
        f"duplicate_candidates={a.duplicate_candidate_count} confirmed_duplicates={a.confirmed_duplicate_count}"
    )
    for pair in a.duplicate_pairs:
        print(
            f"  {pair.document_id} ~ {pair.other_document_id} cos={pair.cosine_similarity} "
            f"jaccard={pair.jaccard_similarity} confirmed={pair.confirmed}"
        )
    print(f"outliers: {a.outliers}")


if __name__ == "__main__":
    main()
