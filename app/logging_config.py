"""Structured JSON logging.

Usage: ``logger.info("search.completed", extra={"fields": {"latency_ms": 12.3}})``
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(getattr(h, "_semantic_search", False) for h in root.handlers):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._semantic_search = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party libraries are chatty at INFO.
    for noisy in ("sentence_transformers", "httpx", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    # Connection failures are reported by our own bm25.* events; avoid duplicate tracebacks.
    logging.getLogger("elastic_transport").setLevel(logging.ERROR)


def log_fields(**fields: object) -> dict[str, dict[str, object]]:
    return {"fields": fields}
