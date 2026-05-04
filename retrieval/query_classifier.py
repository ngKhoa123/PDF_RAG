from langchain_community.llms import Ollama


class QueryClassifier:
    def __init__(self):
        self.llm = Ollama(model="phi3:mini", temperature=0)

    # =========================
    # MAIN
    # =========================
    def classify(self, query: str) -> str:
        q = query.lower().strip()

        # ===== RULE-BASED (precise hơn) =====
        if self._is_summary(q):
            return "summary"

        if self._is_compare(q):
            return "compare"

        if self._is_explain(q):
            return "explain"

        if self._is_extract(q):
            return "extract"

        # ===== fallback: QA vs Explain ambiguity =====
        if self._is_definition(q):
            return "qa"

        # ===== LLM fallback =====
        result = self._llm_classify(query)
        if result:
            return result

        return "qa"

    # =========================
    # RULES (refined)
    # =========================
    def _is_summary(self, q):
        return any(x in q for x in [
            "summarize", "summary", "main idea", "overview"
        ])

    def _is_compare(self, q):
        return any(x in q for x in [
            "compare", "difference", "vs", "contrast"
        ])

    def _is_explain(self, q):
        return any(x in q for x in [
            "how does", "how to", "why"
        ])

    def _is_extract(self, q):
        return any(x in q for x in [
            "list", "extract", "show me", "give me a list"
        ])

    def _is_definition(self, q):
        return any(x in q for x in [
            "what is", "define", "meaning of"
        ])

    # =========================
    # LLM CLASSIFIER (safe)
    # =========================
    def _llm_classify(self, query: str):
        prompt = f"""
Classify the query into ONE of these labels:
summary, qa, explain, compare, extract

Rules:
- Output ONLY one word
- Do NOT explain

Query: {query}
Answer:
"""

        try:
            result = self.llm.invoke(prompt)
            result = result.strip().lower()

            # clean
            result = result.replace(".", "").replace("answer:", "").strip()

            # extract first valid token
            for token in result.split():
                if token in ["summary", "qa", "explain", "compare", "extract"]:
                    return token

        except Exception:
            pass

        return None