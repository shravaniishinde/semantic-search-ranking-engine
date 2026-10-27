"""Learning-to-rank reranker (XGBoost ``rank:ndcg``).

Training data comes from graded relevance judgments (``ranking_training_qrels.json``): every
hybrid candidate retrieved for a training query becomes one row, labelled with its judged grade
(0 when unjudged). Queries form the ranking groups. If no model is trained the reranker falls
back to the hybrid RRF order and says so in ``ranker``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRanker

from app.models.schemas import HybridCandidate, RankedResult
from app.ranking.features import FEATURE_NAMES, to_matrix

logger = logging.getLogger(__name__)


@dataclass
class LTRDataset:
    features: np.ndarray
    labels: np.ndarray
    group_sizes: list[int]
    queries: list[str]


class XGBoostReranker:
    def __init__(self) -> None:
        self.model: XGBRanker | None = None
        self.feature_names: list[str] = list(FEATURE_NAMES)

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def name(self) -> str:
        return "xgboost_ltr" if self.is_trained else "hybrid_fallback"

    def fit(self, dataset: LTRDataset) -> XGBoostReranker:
        if dataset.features.shape[0] == 0:
            raise ValueError("empty LTR dataset")
        model = XGBRanker(
            objective="rank:ndcg",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
        )
        model.fit(dataset.features, dataset.labels, group=dataset.group_sizes)
        self.model = model
        logger.info(
            "ranking.trained",
            extra={
                "fields": {
                    "queries": len(dataset.group_sizes),
                    "rows": int(dataset.features.shape[0]),
                    "positives": int((dataset.labels > 0).sum()),
                }
            },
        )
        return self

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        scores = self.model.feature_importances_
        return {name: round(float(s), 4) for name, s in sorted(zip(self.feature_names, scores), key=lambda x: -x[1])}

    def score(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        if not feature_rows:
            return np.zeros(0, dtype=np.float32)
        if self.model is None:
            return np.array([row["hybrid_score"] for row in feature_rows], dtype=np.float32)
        return self.model.predict(to_matrix(feature_rows))

    def rerank(self, candidates: list[HybridCandidate], feature_rows: list[dict[str, float]]) -> list[RankedResult]:
        if len(candidates) != len(feature_rows):
            raise ValueError("candidates and feature rows must align")
        scores = self.score(feature_rows)
        # Stable tie-break on the hybrid order so equal scores keep the retrieval order.
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return [
            RankedResult(
                document_id=candidates[i].document_id,
                rank=position,
                final_score=round(float(scores[i]), 6),
                bm25_score=candidates[i].bm25_score,
                semantic_score=candidates[i].semantic_score,
                hybrid_score=candidates[i].hybrid_score,
                bm25_rank=candidates[i].bm25_rank,
                semantic_rank=candidates[i].semantic_rank,
                features=feature_rows[i],
                ranker=self.name,
            )
            for position, i in enumerate(order, start=1)
        ]

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("cannot save an untrained reranker")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        payload = joblib.load(path)
        if payload["feature_names"] != list(FEATURE_NAMES):
            logger.warning("ranking.model_feature_mismatch", extra={"fields": {"path": str(path)}})
            return False
        self.model, self.feature_names = payload["model"], payload["feature_names"]
        return True
