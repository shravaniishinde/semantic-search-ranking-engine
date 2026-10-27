"""Query intent classification with XGBoost.

This is a *demonstration* classifier: it is trained on ~125 hand-written queries in
``data/sample/intent_training.jsonl``. Expect it to be useful on queries resembling the training
set and unreliable elsewhere. Cross-validated accuracy is printed by ``scripts/train_models.py``.

Features = hand-crafted signals (length, question words, keyword cues, NER counts) concatenated
with word/bigram TF-IDF.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.models.schemas import IntentPrediction
from app.nlp.ner import EntityExtractor

logger = logging.getLogger(__name__)

INTENTS = ["informational", "technical", "entity_lookup", "troubleshooting", "comparison"]

QUESTION_WORDS = {"what", "why", "how", "when", "where", "which", "who", "explain", "meaning"}
CUES = {
    "troubleshooting": {
        "error",
        "fail",
        "fails",
        "failing",
        "failure",
        "fix",
        "slow",
        "crash",
        "crashloopbackoff",
        "timeout",
        "debug",
        "issue",
        "broken",
        "restarting",
        "stuck",
        "lagging",
        "dropping",
        "deadlock",
        "outofmemoryerror",
        "cannot",
        "keeps",
        "drops",
        "increased",
        "nxdomain",
        "servfail",
    },
    "comparison": {
        "vs",
        "versus",
        "compare",
        "compared",
        "comparison",
        "difference",
        "differences",
        "between",
        "better",
        "or",
        "pros",
        "cons",
    },
    "technical": {
        "configure",
        "set",
        "setup",
        "implement",
        "deploy",
        "create",
        "build",
        "design",
        "enable",
        "store",
        "rotate",
        "plan",
        "steps",
        "architecture",
        "write",
        "migrate",
        "best",
        "practices",
    },
    "informational": {
        "what",
        "explain",
        "overview",
        "introduction",
        "basics",
        "concept",
        "history",
        "meaning",
        "purpose",
        "benefits",
        "why",
    },
}
_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.+]*")


def _words(query: str) -> list[str]:
    return _WORD.findall(query.lower())


@dataclass
class _Artifacts:
    vectorizer: TfidfVectorizer
    label_encoder: LabelEncoder
    model: XGBClassifier


class IntentClassifier:
    HANDCRAFTED = [
        "char_len",
        "token_count",
        "starts_with_question_word",
        "has_question_word",
        "has_question_mark",
        "cue_troubleshooting",
        "cue_comparison",
        "cue_technical",
        "cue_informational",
        "entity_count",
        "product_org_count",
        "entity_token_ratio",
        "capitalized_ratio",
        "has_digit",
    ]

    def __init__(self, entity_extractor: EntityExtractor | None = None) -> None:
        self.entity_extractor = entity_extractor
        self._artifacts: _Artifacts | None = None

    @property
    def is_trained(self) -> bool:
        return self._artifacts is not None

    # ------------------------------------------------------------------ features
    def handcrafted_features(self, query: str) -> np.ndarray:
        words = _words(query)
        raw_tokens = query.split()
        entities = self.entity_extractor.extract(query) if self.entity_extractor else []
        entity_tokens = sum(len(e.text.split()) for e in entities)
        n = max(len(words), 1)
        return np.array(
            [
                len(query),
                len(words),
                float(bool(words) and words[0] in QUESTION_WORDS),
                float(any(w in QUESTION_WORDS for w in words)),
                float("?" in query),
                sum(w in CUES["troubleshooting"] for w in words) / n,
                sum(w in CUES["comparison"] for w in words) / n,
                sum(w in CUES["technical"] for w in words) / n,
                sum(w in CUES["informational"] for w in words) / n,
                len(entities),
                sum(e.label in {"PRODUCT", "ORG", "EVENT", "VULNERABILITY_ID"} for e in entities),
                entity_tokens / n,
                sum(t[:1].isupper() for t in raw_tokens) / max(len(raw_tokens), 1),
                float(any(ch.isdigit() for ch in query)),
            ],
            dtype=np.float32,
        )

    def _matrix(self, queries: list[str], vectorizer: TfidfVectorizer) -> sparse.csr_matrix:
        handcrafted = sparse.csr_matrix(np.stack([self.handcrafted_features(q) for q in queries]))
        return sparse.hstack([handcrafted, vectorizer.transform(queries)]).tocsr()

    @staticmethod
    def _new_model(n_classes: int) -> XGBClassifier:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=150,
            max_depth=4,
            learning_rate=0.15,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            eval_metric="mlogloss",
        )

    # ------------------------------------------------------------------ training
    def fit(self, queries: list[str], labels: list[str]) -> IntentClassifier:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)
        vectorizer.fit(queries)
        encoder = LabelEncoder().fit(labels)
        model = self._new_model(len(encoder.classes_))
        model.fit(self._matrix(queries, vectorizer), encoder.transform(labels))
        self._artifacts = _Artifacts(vectorizer, encoder, model)
        logger.info("intent.trained", extra={"fields": {"examples": len(queries), "classes": list(encoder.classes_)}})
        return self

    def cross_validate(self, queries: list[str], labels: list[str], folds: int = 5) -> dict[str, float]:
        """Stratified k-fold accuracy, refitting the TF-IDF vocabulary inside each fold."""
        y = np.array(labels)
        accuracies = []
        for train_idx, test_idx in StratifiedKFold(folds, shuffle=True, random_state=42).split(queries, y):
            fold = IntentClassifier(self.entity_extractor).fit([queries[i] for i in train_idx], list(y[train_idx]))
            preds = [fold.predict(queries[i]).label for i in test_idx]
            accuracies.append(float(np.mean([p == t for p, t in zip(preds, y[test_idx])])))
        return {"mean_accuracy": float(np.mean(accuracies)), "std_accuracy": float(np.std(accuracies)), "folds": folds}

    @staticmethod
    def load_training_data(path: Path) -> tuple[list[str], list[str]]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r["query"] for r in rows], [r["intent"] for r in rows]

    # ------------------------------------------------------------------ inference
    def predict(self, query: str) -> IntentPrediction:
        if self._artifacts is None:
            return self._rule_fallback(query)
        art = self._artifacts
        probs = art.model.predict_proba(self._matrix([query], art.vectorizer))[0]
        best = int(np.argmax(probs))
        return IntentPrediction(
            label=str(art.label_encoder.classes_[best]),
            confidence=round(float(probs[best]), 4),
            probabilities={str(c): round(float(p), 4) for c, p in zip(art.label_encoder.classes_, probs)},
            source="xgboost",
        )

    def _rule_fallback(self, query: str) -> IntentPrediction:
        """Keyword heuristic used only when no trained model is available."""
        words = _words(query)
        scores = {intent: sum(w in CUES[intent] for w in words) for intent in CUES}
        if self.entity_extractor and not scores["comparison"] and len(words) <= 4:
            if self.entity_extractor.extract(query):
                scores["entity_lookup"] = 2
        best = max(scores, key=scores.get) if any(scores.values()) else "informational"
        return IntentPrediction(label=best, confidence=0.0, probabilities={}, source="rule_fallback")

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        if self._artifacts is None:
            raise RuntimeError("cannot save an untrained classifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._artifacts, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        self._artifacts = joblib.load(path)
        return True
