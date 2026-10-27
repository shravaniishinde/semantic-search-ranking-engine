"""Ranking feature generation for (query, candidate document) pairs.

Kept separate from the model so the same features feed training, inference and the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.classification.intent import INTENTS
from app.models.schemas import Document, Entity, HybridCandidate, IntentPrediction
from app.nlp.ner import entity_keys
from app.preprocessing.text import tokenize

# Extra vocabulary per category so "reduce cloud bill" or "kubectl" can match a category.
CATEGORY_VOCABULARY: dict[str, set[str]] = {
    "cloud computing": {
        "cloud",
        "aws",
        "azure",
        "gcp",
        "serverless",
        "lambda",
        "s3",
        "bucket",
        "vpc",
        "autoscaling",
        "finop",
        "bill",
        "instance",
        "migration",
        "cdn",
        "saas",
    },
    "cybersecurity": {
        "security",
        "attack",
        "attacker",
        "vulnerability",
        "cve",
        "phishing",
        "ransomware",
        "malware",
        "encryption",
        "threat",
        "breach",
        "password",
        "injection",
        "exploit",
        "zero",
        "trust",
    },
    "databases": {
        "database",
        "sql",
        "nosql",
        "query",
        "index",
        "transaction",
        "postgres",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "replica",
        "shard",
        "deadlock",
        "warehouse",
        "olap",
    },
    "artificial intelligence": {
        "ai",
        "model",
        "learning",
        "neural",
        "transformer",
        "bert",
        "embedding",
        "llm",
        "classifier",
        "training",
        "attention",
        "machine",
        "language",
        "overfitting",
    },
    "software engineering": {
        "code",
        "test",
        "testing",
        "refactor",
        "design",
        "pattern",
        "api",
        "rest",
        "review",
        "microservice",
        "monolith",
        "debt",
        "version",
        "agile",
        "scrum",
        "java",
    },
    "networking": {
        "network",
        "tcp",
        "udp",
        "dns",
        "bgp",
        "ip",
        "ipv6",
        "subnet",
        "cidr",
        "latency",
        "packet",
        "router",
        "vpn",
        "firewall",
        "http",
        "quic",
        "load",
        "balancer",
    },
    "devops": {
        "devops",
        "ci",
        "cd",
        "pipeline",
        "docker",
        "container",
        "kubernetes",
        "pod",
        "kubectl",
        "terraform",
        "deploy",
        "deployment",
        "monitoring",
        "prometheus",
        "gitop",
        "sre",
        "slo",
        "secret",
    },
}

FEATURE_NAMES: list[str] = [
    "bm25_score",
    "bm25_score_norm",
    "bm25_rank",
    "in_bm25",
    "semantic_similarity",
    "semantic_rank",
    "in_semantic",
    "hybrid_score",
    "hybrid_rank",
    "query_term_coverage",
    "title_term_coverage",
    "title_phrase_match",
    "entity_overlap",
    "category_match",
    "doc_length_log",
    "query_length",
    *[f"intent_{name}" for name in INTENTS],
    "intent_confidence",
]


@dataclass
class QueryContext:
    query: str
    query_vector: np.ndarray
    entities: list[Entity] = field(default_factory=list)
    intent: IntentPrediction | None = None

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.query)


def category_match(query_tokens: set[str], category: str) -> float:
    vocab = CATEGORY_VOCABULARY.get(category.lower(), set()) | set(tokenize(category))
    return float(bool(query_tokens & vocab))


class RankingFeatureExtractor:
    def __init__(self, candidate_depth: int = 30) -> None:
        # Rank assigned to a document that one retriever did not return at all.
        self.missing_rank = candidate_depth + 1

    def extract(
        self,
        ctx: QueryContext,
        candidates: list[HybridCandidate],
        documents: dict[str, Document],
        doc_vectors: dict[str, np.ndarray],
    ) -> list[dict[str, float]]:
        query_tokens = set(ctx.tokens)
        query_entities = entity_keys(ctx.entities)
        max_bm25 = max((c.bm25_score or 0.0 for c in candidates), default=0.0) or 1.0
        query_vec = ctx.query_vector / max(float(np.linalg.norm(ctx.query_vector)), 1e-12)
        intent_label = ctx.intent.label if ctx.intent else None

        rows = []
        for hybrid_rank, cand in enumerate(candidates, start=1):
            doc = documents[cand.document_id]
            doc_tokens = set(tokenize(doc.full_text))
            title_tokens = set(tokenize(doc.title))
            doc_vec = doc_vectors.get(cand.document_id)
            semantic = float(np.dot(query_vec, doc_vec)) if doc_vec is not None else 0.0
            doc_entities = entity_keys(doc.entities)
            lowered_text = doc.full_text.lower()
            entity_hits = sum(1 for e in query_entities if e in doc_entities or e in lowered_text)

            row = {
                "bm25_score": cand.bm25_score or 0.0,
                "bm25_score_norm": (cand.bm25_score or 0.0) / max_bm25,
                "bm25_rank": float(cand.bm25_rank or self.missing_rank),
                "in_bm25": float(cand.bm25_rank is not None),
                "semantic_similarity": semantic,
                "semantic_rank": float(cand.semantic_rank or self.missing_rank),
                "in_semantic": float(cand.semantic_rank is not None),
                "hybrid_score": cand.hybrid_score,
                "hybrid_rank": float(hybrid_rank),
                "query_term_coverage": len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.0,
                "title_term_coverage": len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0,
                "title_phrase_match": float(bool(ctx.query.strip()) and ctx.query.strip().lower() in doc.title.lower()),
                "entity_overlap": entity_hits / len(query_entities) if query_entities else 0.0,
                "category_match": category_match(query_tokens, doc.category),
                "doc_length_log": math.log1p(len(doc.content.split())),
                "query_length": float(len(ctx.query.split())),
                **{f"intent_{name}": float(intent_label == name) for name in INTENTS},
                "intent_confidence": ctx.intent.confidence if ctx.intent else 0.0,
            }
            rows.append({k: round(float(row[k]), 6) for k in FEATURE_NAMES})
        return rows


def to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32).reshape(
        len(rows), len(FEATURE_NAMES)
    )
