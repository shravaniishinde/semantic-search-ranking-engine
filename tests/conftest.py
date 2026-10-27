from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import PROJECT_ROOT, Settings
from app.embeddings.encoder import HashingEmbedder
from app.ingestion.loader import load_documents
from app.nlp.ner import EntityExtractor
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.services.search_service import SearchEngine

SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample" / "documents.jsonl"


def make_settings(tmp: Path, **overrides) -> Settings:
    processed = tmp / "processed"
    values = dict(
        processed_dir=processed,
        faiss_index_path=processed / "faiss.index",
        embedding_cache_path=processed / "embedding_cache.npz",
        document_store_path=processed / "documents.json",
        intent_model_path=processed / "models" / "intent.joblib",
        reranker_model_path=processed / "models" / "reranker.joblib",
        bm25_backend="memory",
        log_level="WARNING",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="session")
def ner() -> EntityExtractor:
    return EntityExtractor("en_core_web_sm")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def sample_documents():
    docs, rejected = load_documents(SAMPLE_DOCS)
    assert not rejected
    return docs


@pytest.fixture
def fast_engine(settings: Settings, ner: EntityExtractor, sample_documents) -> SearchEngine:
    """Full engine with a hashing embedder (no transformer download) and in-memory BM25."""
    engine = SearchEngine(
        settings, embedder=HashingEmbedder(), entity_extractor=ner, lexical=InMemoryBM25Retriever(), persist=False
    )
    engine.ingest(sample_documents)
    engine.ensure_models(train_if_missing=True)
    return engine


@pytest.fixture(scope="session")
def real_engine(tmp_path_factory, ner: EntityExtractor) -> SearchEngine:
    """Engine with the real sentence-transformer model; built once per test session."""
    tmp = tmp_path_factory.mktemp("real_engine")
    engine = SearchEngine(make_settings(tmp), entity_extractor=ner, lexical=InMemoryBM25Retriever(), persist=True)
    docs, rejected = load_documents(SAMPLE_DOCS)
    engine.ingest(docs, pre_rejected=rejected)
    engine.ensure_models(train_if_missing=True)
    return engine
