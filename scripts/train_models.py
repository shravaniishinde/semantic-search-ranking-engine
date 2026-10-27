"""Train and persist the XGBoost intent classifier and the XGBoost reranker.

python scripts/train_models.py
"""

import json

import _bootstrap  # noqa: F401

from app.classification.intent import IntentClassifier
from app.config.settings import get_settings
from app.ingestion.loader import load_documents
from app.logging_config import configure_logging
from app.services.search_service import SearchEngine


def main() -> None:
    settings = get_settings()
    configure_logging("WARNING")
    engine = SearchEngine(settings)
    if not engine.load_state():
        docs, rejected = load_documents(settings.sample_documents_path)
        engine.ingest(docs, pre_rejected=rejected)

    queries, labels = IntentClassifier.load_training_data(settings.intent_training_path)
    cv = IntentClassifier(engine.ner).cross_validate(queries, labels)
    print(
        f"Intent classifier 5-fold CV accuracy on {len(queries)} synthetic queries: "
        f"{cv['mean_accuracy']:.3f} +/- {cv['std_accuracy']:.3f}  (small demo dataset - not a real benchmark)"
    )
    print(json.dumps(engine.train_intent_classifier(), indent=2))

    summary = engine.train_reranker()
    print(f"\nReranker trained on {summary['queries']} queries / {summary['rows']} candidate rows")
    print("Feature importance (gain share):")
    for name, value in summary["feature_importance"].items():
        if value > 0:
            print(f"  {name:<24}{value:.4f}")
    print(f"\nSaved: {settings.intent_model_path}\n       {settings.reranker_model_path}")


if __name__ == "__main__":
    main()
