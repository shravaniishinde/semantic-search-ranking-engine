import json

import pytest
from pydantic import ValidationError

from app.ingestion.loader import load_documents, parse_records
from app.models.schemas import Document, DocumentIn, SearchRequest
from tests.conftest import SAMPLE_DOCS


def test_document_in_accepts_aliases():
    doc = DocumentIn.model_validate({"id": "x1", "title": "T", "text": "body text", "date": "2024-01-02"})
    assert doc.document_id == "x1"
    assert doc.content == "body text"
    assert doc.created_at.isoformat() == "2024-01-02"


def test_document_id_generated_deterministically_when_missing():
    a = DocumentIn(title="Same", content="content")
    b = DocumentIn(title="Same", content="content")
    c = DocumentIn(title="Other", content="content")
    assert a.document_id == b.document_id
    assert a.document_id != c.document_id
    assert a.document_id.startswith("doc-")


def test_document_defaults_for_analysis_flags():
    doc = Document(document_id="d", title="t", content="c")
    assert doc.cluster_id is None
    assert not (doc.is_duplicate or doc.duplicate_candidate or doc.is_outlier or doc.indexed)
    assert doc.full_text == "t. c"


@pytest.mark.parametrize(
    "payload", [{"query": "   "}, {"query": ""}, {"query": "x" * 513}, {"query": "ok", "top_k": 0}, {"query": "ok", "top_k": 51}]
)
def test_search_request_validation(payload):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_sample_corpus_loads():
    docs, rejected = load_documents(SAMPLE_DOCS)
    assert len(docs) >= 70 and not rejected
    assert len({d.document_id for d in docs}) == len(docs)
    categories = {d.category for d in docs}
    assert {
        "cloud computing",
        "cybersecurity",
        "databases",
        "artificial intelligence",
        "software engineering",
        "networking",
        "devops",
    } <= categories


def test_load_json_list_and_wrapped_object(tmp_path):
    records = [{"document_id": "a", "title": "A", "content": "alpha"}]
    (tmp_path / "list.json").write_text(json.dumps(records))
    (tmp_path / "wrapped.json").write_text(json.dumps({"documents": records}))
    for name in ("list.json", "wrapped.json"):
        docs, rejected = load_documents(tmp_path / name)
        assert [d.document_id for d in docs] == ["a"] and not rejected


def test_invalid_records_are_rejected_not_raised():
    docs, rejected = parse_records([{"document_id": "ok", "content": "fine"}, {"document_id": "bad", "created_at": "not-a-date"}])
    assert [d.document_id for d in docs] == ["ok"]
    assert rejected[0].document_id == "bad" and "schema validation failed" in rejected[0].reason
