from typing import List
import re


class ContextFilter:

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    # =========================
    # LOW INFO FILTER (nới lỏng)
    # =========================
    def _is_low_information(self, text: str) -> bool:
        text = text.strip()

        if len(text) < 10:
            return True

        return False

    # =========================
    # SCORING 
    # =========================
    def score(self, query: str, docs: List):
        query_words = set(self._normalize(query).split())
        scored = []

        for doc in docs:
            raw_text = doc.page_content or ""
            text = self._normalize(raw_text)

            if not text:
                continue

            if self._is_low_information(text):
                continue

            words = set(text.split())

            overlap = len(query_words & words)

            if overlap == 0:
                overlap = 0.5

            if "introduction" in text or "we propose" in text:
                overlap += 2

            phrase_bonus = 1 if query.lower() in text else 0

            final_score = overlap * 2 + phrase_bonus

            scored.append((final_score, doc))

        # sort
        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored]