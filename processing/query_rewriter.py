from langchain_community.llms import Ollama
from retrieval.query_classifier import QueryClassifier


class QueryRewriter:
    def __init__(self):
        self.llm = Ollama(model="phi3:mini", temperature=0)
        self.classifier = QueryClassifier()
        self.max_variants = 3

    def process(self, query: str):
        query = query.strip()
        if not query:
            return []

        try:
            mode = self.classifier.classify(query)
        except Exception:
            mode = "qa"

        variants = [query]

        # ===== Rewrite (controlled) =====
        if len(query.split()) <= 10:
            rewritten = self._rewrite_safe(query, mode)
            if self._is_valid_rewrite(query, rewritten):
                variants.append(rewritten)

        variants.extend(self._expand_intent(query))

        # ===== Light semantic expansion =====
        variants.extend(self._expand_semantic(query))

        variants = self._deduplicate(variants)

        return variants[:self.max_variants]

    # =========================
    # SAFE REWRITE
    # =========================
    def _rewrite_safe(self, query: str, mode: str) -> str:
        prompt = f"""
Rewrite the query for better document retrieval.

Intent: {mode}

Rules:
- Keep exact meaning
- Do NOT add new concepts
- Clarify wording only

Query: {query}

Rewritten:
"""
        try:
            out = self.llm.invoke(prompt).strip()
            return out.replace("\n", " ").strip()
        except Exception:
            return query

    # =========================
    # INTENT EXPANSION
    # =========================
    def _expand_intent(self, query: str):
        variants = []
        q = query.lower()

        # ===== MAIN IDEA / SUMMARY =====
        if "main idea" in q or "main point" in q:
            variants.append("what does the paper propose")
            variants.append("main contribution of the paper")

        if "summary" in q:
            variants.append("what is the paper about")

        # ===== PROPOSAL =====
        if "propose" in q:
            variants.append("what method is introduced")

        return variants

    # =========================
    # SEMANTIC EXPANSION
    # =========================
    def _expand_semantic(self, query: str):
        variants = []
        q = query.lower()

        if "how" in q:
            variants.append(query.replace("how", "method to"))

        if "why" in q:
            variants.append(query.replace("why", "reason for"))

        if "compare" in q or "difference" in q:
            variants.append(query + " comparison")

        return variants

    # =========================
    # VALIDATION
    # =========================
    def _is_valid_rewrite(self, original: str, rewritten: str) -> bool:
        if not rewritten or rewritten == original:
            return False

        if len(rewritten.split()) < 3:
            return False

        if abs(len(rewritten) - len(original)) > 30:
            return False

        if rewritten.lower().startswith("this"):
            return False

        return True

    # =========================
    # DEDUP
    # =========================
    def _deduplicate(self, items):
        seen = set()
        result = []

        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result