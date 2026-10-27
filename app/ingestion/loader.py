"""Load raw documents from JSON / JSONL files and validate them against ``DocumentIn``."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.models.schemas import DocumentIn, RejectedDocument


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "documents" in data:
        data = data["documents"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or an object with a 'documents' list")
    return data


def parse_records(records: Iterable[dict[str, Any]]) -> tuple[list[DocumentIn], list[RejectedDocument]]:
    docs: list[DocumentIn] = []
    rejected: list[RejectedDocument] = []
    for record in records:
        try:
            docs.append(DocumentIn.model_validate(record))
        except ValidationError as exc:
            doc_id = record.get("document_id") or record.get("id") if isinstance(record, dict) else None
            rejected.append(RejectedDocument(document_id=doc_id, reason=f"schema validation failed: {exc.errors()[0]['msg']}"))
    return docs, rejected


def load_documents(path: str | Path) -> tuple[list[DocumentIn], list[RejectedDocument]]:
    return parse_records(_read_records(Path(path)))
