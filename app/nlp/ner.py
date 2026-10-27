"""spaCy named entity recognition for documents and queries.

``en_core_web_sm`` is weak on technology names (it often tags "AWS" or "Kubernetes" as nothing,
or as PERSON), so an EntityRuler with a small technology gazetteer runs *before* the statistical
NER component. Ruler matches take precedence; everything else comes from the pretrained model.
If the spaCy model is not installed, the extractor degrades to gazetteer-only matching so the
rest of the pipeline keeps working.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from app.models.schemas import Entity

logger = logging.getLogger(__name__)

KEPT_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "DATE", "EVENT", "NORP", "LAW", "WORK_OF_ART", "LANGUAGE", "FAC"}
)

# (pattern text, label). Matched case-insensitively.
TECH_GAZETTEER: list[tuple[str, str]] = [
    *[
        (name, "ORG")
        for name in [
            "AWS",
            "Amazon Web Services",
            "Microsoft",
            "Google",
            "Meta",
            "Facebook",
            "HashiCorp",
            "DeepMind",
            "Cloudflare",
            "HackerOne",
            "Splunk",
            "NIST",
            "OWASP",
            "CNCF",
            "IANA",
            "Apache Software Foundation",
        ]
    ],
    *[
        (name, "PRODUCT")
        for name in [
            "AWS Lambda",
            "Amazon S3",
            "S3",
            "S3 Glacier",
            "CloudFront",
            "CloudTrail",
            "GuardDuty",
            "AWS KMS",
            "AWS Organizations",
            "AWS Secrets Manager",
            "Azure",
            "Microsoft Azure",
            "Google Cloud",
            "Entra ID",
            "Microsoft Entra ID",
            "Azure Active Directory",
            "Kubernetes",
            "Docker",
            "Terraform",
            "Vault",
            "HashiCorp Vault",
            "Prometheus",
            "Grafana",
            "Argo CD",
            "Flux",
            "Jenkins",
            "GitHub Actions",
            "GitLab CI",
            "GitHub",
            "PostgreSQL",
            "Postgres",
            "MySQL",
            "InnoDB",
            "MongoDB",
            "Redis",
            "Cassandra",
            "Apache Cassandra",
            "Spanner",
            "Google Spanner",
            "Snowflake",
            "BigQuery",
            "Google BigQuery",
            "Pinecone",
            "Milvus",
            "FAISS",
            "Elasticsearch",
            "Log4j",
            "Apache Log4j",
            "Orion",
            "SolarWinds Orion",
            "NGINX",
            "HAProxy",
            "WireGuard",
            "Wireshark",
            "Sigstore",
            "BeyondCorp",
            "BERT",
            "Sentence-BERT",
            "XGBoost",
            "AlphaGo",
            "AlphaZero",
            "AlexNet",
            "ResNet",
            "ImageNet",
            "Transformer",
            "Lambda",
        ]
    ],
    *[(name, "EVENT") for name in ["Log4Shell", "WannaCry", "SolarWinds incident", "Black Friday"]],
    *[
        (name, "PERSON")
        for name in [
            "Kent Beck",
            "Martin Fowler",
            "Robert C. Martin",
            "Ward Cunningham",
            "Eric Brewer",
            "Tianqi Chen",
            "Lee Sedol",
        ]
    ],
]
# Regex patterns for identifiers spaCy never knows.
_CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_GAZETTEER_LOOKUP = {text.lower(): label for text, label in TECH_GAZETTEER}


def _dedupe(entities: Iterable[Entity]) -> list[Entity]:
    seen: set[tuple[str, str]] = set()
    out: list[Entity] = []
    for ent in entities:
        key = (ent.text.lower(), ent.label)
        if key not in seen:
            seen.add(key)
            out.append(ent)
    return out


class EntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self.nlp = None
        self.backend = "gazetteer"
        try:
            import spacy

            # Only NER is needed; dropping tagger/parser/lemmatizer makes extraction several times faster.
            nlp = spacy.load(model_name, exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "senter"])
            ruler = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
            ruler.add_patterns([{"label": label, "pattern": text} for text, label in TECH_GAZETTEER])
            self.nlp = nlp
            self.backend = "spacy"
        except (ImportError, OSError) as exc:
            logger.warning("ner.spacy_unavailable", extra={"fields": {"model": model_name, "error": str(exc)}})

        self._query_cache: dict[str, list[Entity]] = {}

    def extract(self, text: str) -> list[Entity]:
        """Single-text extraction with a small cache (queries are often seen repeatedly, e.g. once
        by the intent classifier and once by the search pipeline)."""
        if not text:
            return []
        cached = self._query_cache.get(text)
        if cached is None:
            cached = self.extract_batch([text])[0]
            if len(self._query_cache) >= 4096:
                self._query_cache.clear()
            self._query_cache[text] = cached
        return [e.model_copy() for e in cached]

    def extract_batch(self, texts: list[str]) -> list[list[Entity]]:
        if self.nlp is not None:
            per_text = [
                [
                    Entity(text=e.text, label=e.label_, start=e.start_char, end=e.end_char)
                    for e in doc.ents
                    if e.label_ in KEPT_LABELS
                ]
                for doc in self.nlp.pipe(texts, batch_size=32)
            ]
        else:
            per_text = [self._gazetteer_only(t) for t in texts]
        results = []
        for text, ents in zip(texts, per_text):
            cves = [
                Entity(text=m.group(0).upper(), label="VULNERABILITY_ID", start=m.start(), end=m.end())
                for m in _CVE.finditer(text)
            ]
            # A CVE id supersedes whatever the statistical model guessed for the same span.
            ents = [e for e in ents if not any(e.start < c.end and c.start < e.end for c in cves)]
            results.append(_dedupe(ents + cves))
        return results

    @staticmethod
    def _gazetteer_only(text: str) -> list[Entity]:
        found = []
        lowered = text.lower()
        # Longest names first so "AWS Lambda" wins over "AWS".
        for name in sorted(_GAZETTEER_LOOKUP, key=len, reverse=True):
            for match in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", lowered):
                span = (match.start(), match.end())
                if any(span[0] < f.end and f.start < span[1] for f in found):  # type: ignore[operator]
                    continue
                found.append(Entity(text=text[span[0] : span[1]], label=_GAZETTEER_LOOKUP[name], start=span[0], end=span[1]))
        return sorted(found, key=lambda e: e.start or 0)


def entity_keys(entities: Iterable[Entity]) -> set[str]:
    """Lowercased entity surface forms, used for overlap features and filtering."""
    return {e.text.lower() for e in entities}
