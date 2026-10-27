"""Central configuration, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_queries: bool = True  # set false to keep raw query text out of logs
    # Browser origins allowed to call the API (the Vite dev server and the dockerised dashboard).
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    # --- Data paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    sample_documents_path: Path = PROJECT_ROOT / "data" / "sample" / "documents.jsonl"
    intent_training_path: Path = PROJECT_ROOT / "data" / "sample" / "intent_training.jsonl"
    ranking_training_path: Path = PROJECT_ROOT / "data" / "sample" / "ranking_training_qrels.json"
    evaluation_qrels_path: Path = PROJECT_ROOT / "data" / "sample" / "evaluation_qrels.json"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    faiss_index_path: Path = PROJECT_ROOT / "data" / "processed" / "faiss.index"
    embedding_cache_path: Path = PROJECT_ROOT / "data" / "processed" / "embedding_cache.npz"
    document_store_path: Path = PROJECT_ROOT / "data" / "processed" / "documents.json"
    intent_model_path: Path = PROJECT_ROOT / "data" / "processed" / "models" / "intent_classifier.joblib"
    reranker_model_path: Path = PROJECT_ROOT / "data" / "processed" / "models" / "reranker.joblib"

    # --- Startup behaviour ---
    load_sample_on_startup: bool = True  # ingest the bundled sample corpus when the store is empty
    auto_train_models: bool = True  # train intent/reranker models at startup if artifacts are missing

    # --- NLP / embeddings ---
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # --- Lexical retrieval ---
    bm25_backend: Literal["auto", "elasticsearch", "memory"] = "auto"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "documents"
    elasticsearch_timeout: float = 5.0

    # --- Hybrid retrieval ---
    rrf_k: int = Field(default=60, ge=1)
    candidate_k: int = Field(default=30, ge=1)  # depth fetched from each retriever before fusion

    # --- DBSCAN analysis ---
    dbscan_eps: float = 0.75  # cosine distance on mean-centred embeddings
    dbscan_min_samples: int = 2
    duplicate_candidate_threshold: float = 0.90  # cosine similarity
    duplicate_confirm_threshold: float = 0.95  # cosine similarity
    duplicate_jaccard_threshold: float = 0.5  # token-set Jaccard needed to confirm a duplicate
    exclude_outliers_from_index: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
