"""FastAPI application exposing the search engine."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import Settings, get_settings
from app.evaluation.evaluator import evaluate, load_qrels
from app.logging_config import configure_logging
from app.models.schemas import (
    ClusterAnalysis,
    Document,
    DocumentListResponse,
    IngestionReport,
    IngestRequest,
    SearchRequest,
    SearchResponse,
)
from app.services.search_service import QueryValidationError, SearchEngine

logger = logging.getLogger(__name__)


def create_app(engine_factory: Callable[[Settings], SearchEngine] | None = None, settings: Settings | None = None) -> FastAPI:
    """``engine_factory`` lets tests inject an engine built from lightweight components."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = engine_factory(settings) if engine_factory else SearchEngine(settings)
        engine.bootstrap()
        app.state.engine = engine
        logger.info("api.started", extra={"fields": {"documents": len(engine.documents), "bm25_backend": engine.bm25_backend}})
        yield

    app = FastAPI(
        title="Semantic Search & Ranking Engine",
        version="0.1.0",
        description="Hybrid BM25 + FAISS retrieval with XGBoost intent classification and learning-to-rank.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def get_engine(request: Request) -> SearchEngine:
        return request.app.state.engine

    eval_report_path = settings.processed_dir / "eval_report.json"

    @app.get("/health")
    def health(engine: SearchEngine = Depends(get_engine)) -> dict[str, Any]:
        return {
            "status": "ok",
            "documents": len(engine.documents),
            "indexed": len(engine.faiss_index),
            "bm25_backend": engine.bm25_backend,
            # What was asked for (auto | elasticsearch | memory); lets clients tell a deliberate
            # in-memory setup from an automatic fallback.
            "bm25_backend_requested": settings.bm25_backend,
            # False when the configured backend (Elasticsearch) stopped answering after startup:
            # searches then continue semantic-only.
            "bm25_available": engine.bm25_available(),
            "bm25_documents": engine.lexical.count(),
            "faiss_vectors": len(engine.faiss_index),
            "embedding_model": engine.embedder.model_name,
            "ner_backend": engine.ner.backend,
            "intent_model": "xgboost" if engine.intent.is_trained else "rule_fallback",
            "reranker": engine.reranker.name,
        }

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest, engine: SearchEngine = Depends(get_engine)) -> SearchResponse:
        try:
            return engine.search(request.query, request.top_k, request.filters, request.include_retrieval)
        except QueryValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/documents/ingest", response_model=IngestionReport)
    def ingest(request: IngestRequest, engine: SearchEngine = Depends(get_engine)) -> IngestionReport:
        return engine.ingest(request.documents)

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents(
        offset: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=200),
        category: str | None = None,
        include_excluded: bool = True,
        q: str | None = Query(None, max_length=200, description="substring match on document_id or title"),
        source: str | None = None,
        cluster_id: int | None = None,
        is_duplicate: bool | None = None,
        duplicate_candidate: bool | None = None,
        is_outlier: bool | None = None,
        engine: SearchEngine = Depends(get_engine),
    ) -> DocumentListResponse:
        total, docs = engine.list_documents(
            offset,
            limit,
            category,
            include_excluded,
            q=q,
            source=source,
            cluster_id=cluster_id,
            is_duplicate=is_duplicate,
            duplicate_candidate=duplicate_candidate,
            is_outlier=is_outlier,
        )
        return DocumentListResponse(total=total, offset=offset, limit=limit, documents=docs)

    @app.get("/documents/{document_id}", response_model=Document)
    def get_document(document_id: str, engine: SearchEngine = Depends(get_engine)) -> Document:
        doc = engine.get_document(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document '{document_id}' not found")
        return doc

    @app.get("/analytics")
    def analytics(engine: SearchEngine = Depends(get_engine)) -> dict[str, Any]:
        return engine.analytics()

    @app.get("/clusters", response_model=ClusterAnalysis)
    def clusters(engine: SearchEngine = Depends(get_engine)) -> ClusterAnalysis:
        if engine.analysis is None:
            raise HTTPException(status_code=404, detail="no documents have been analysed yet")
        return engine.analysis

    @app.get("/evaluation")
    def evaluation_report() -> dict[str, Any]:
        """Latest offline evaluation report (written by scripts/evaluate.py or POST /evaluation/run)."""
        if not eval_report_path.exists():
            return {"available": False, "report": None}
        return {"available": True, "report": json.loads(eval_report_path.read_text(encoding="utf-8"))}

    @app.post("/evaluation/run")
    def run_evaluation(engine: SearchEngine = Depends(get_engine)) -> dict[str, Any]:
        """Re-run the held-out evaluation against the current index (takes a few seconds)."""
        report = evaluate(engine, load_qrels(settings.evaluation_qrels_path))
        eval_report_path.parent.mkdir(parents=True, exist_ok=True)
        eval_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"available": True, "report": report}

    return app


app = create_app()  # uvicorn app.api.main:app
