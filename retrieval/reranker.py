from typing import List
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    def __init__(
        self,
        mode="cross-encoder",
        top_k: int = 5,
        max_length: int = 512,
        batch_size: int = 16
    ):
        self.mode = mode
        self.top_k = top_k
        self.max_length = max_length
        self.batch_size = batch_size

        if mode == "cross-encoder":
            self.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

    # =========================
    # MAIN
    # =========================
    def rerank(self, query: str, docs: List):
        if not docs:
            return []

        docs = docs[:20]

        if self.mode == "cross-encoder":
            return self._cross_encoder_rerank(query, docs)

        return self._simple_rerank(query, docs)

    # =========================
    # SIMPLE (fallback)
    # =========================
    def _simple_rerank(self, query, docs):
        query_words = set(query.lower().split())

        scored = []
        for doc in docs:
            text = doc.page_content.lower()
            words = set(text.split())

            score = len(query_words & words)

            if "we propose" in text:
                score += 3
            if "transformer" in text:
                score += 2


            score += doc.metadata.get("importance", 0)

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:self.top_k]]

    # =========================
    # CROSS ENCODER 
    # =========================
    def _cross_encoder_rerank(self, query, docs):
        pairs = []
        meta_boosts = []

        for doc in docs:
            text = doc.page_content[:self.max_length]

            pairs.append((query, text))

            # ===== metadata boost =====
            boost = 0

            lower = text.lower()

            if "we propose" in lower:
                boost += 1.5

            if "transformer" in lower:
                boost += 1.0

            boost += doc.metadata.get("importance", 0)

            meta_boosts.append(boost)

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        final_scores = np.array(scores) + np.array(meta_boosts)

        scored_docs = list(zip(final_scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:self.top_k]]

    # =========================
    # WITH SCORES
    # =========================
    def rerank_with_scores(self, query: str, docs: List):
        if self.mode != "cross-encoder":
            return []

        pairs = [(query, d.page_content[:self.max_length]) for d in docs]
        scores = self.model.predict(pairs)

        return sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )