from typing import List
from langchain_core.documents import Document
import re


class ContextBuilder:

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    # =========================
    # MAIN
    # =========================
    def build(self, docs: List[Document], query: str, top_k: int = 8) -> List[str]:
        if not docs:
            return []

        contexts = []
        total_tokens = 0

        # ===== FORCE INTRO =====
        intro_doc = None
        for d in docs:
            if "we propose" in (d.page_content or "").lower():
                intro_doc = d
                break

        if intro_doc:
            formatted = self._format_doc(intro_doc)
            contexts.append(formatted)
            total_tokens += self._estimate_tokens(formatted)

        # ===== NORMAL FLOW =====
        for doc in docs[:top_k]:
            if doc == intro_doc:
                continue

            text = (doc.page_content or "").strip()
            if not text:
                continue

            lower = text.lower()

            if self._is_noise(lower):
                continue

            formatted = self._format_doc(doc)
            tokens = self._estimate_tokens(formatted)

            if total_tokens + tokens > self.max_tokens:
                break

            contexts.append(formatted)
            total_tokens += tokens

        # ===== FALLBACK =====
        if not contexts:
            print(">>> CONTEXT EMPTY → FALLBACK")
            for d in docs[:3]:
                if d.page_content:
                    contexts.append(self._format_doc(d))

        return contexts

    # =========================
    # NOISE FILTER (CRITICAL)
    # =========================
    def _is_noise(self, text: str) -> bool:
        lower = text.lower()
    
        # ===== reference headers =====
        if re.fullmatch(r"\s*(references|appendix|bibliography)\s*", lower):
            return True
    
        # ===== high citation density =====
        citations = re.findall(r"\[\d+\]", text)
    
        if len(citations) > 5:
            return True
    
        # ===== common boilerplate =====
        boilerplate_patterns = [
            r"\ball rights reserved\b",
            r"\bcopyright\b",
            r"\bproceedings of\b",
        ]
    
        for pattern in boilerplate_patterns:
            if re.search(pattern, lower):
                return True
    
        # ===== too short =====
        if len(text.split()) < 5:
            return True
    
        return False

    # =========================
    # FORMAT
    # =========================
    def _format_doc(self, doc: Document) -> str:
        source = doc.metadata.get("file_name", "unknown")
        chunk_id = doc.metadata.get("chunk_id", 0)

        text = doc.page_content[:800]

        return f"[{source} | chunk {chunk_id}]\n{text}"

    # =========================
    # TOKEN
    # =========================
    def _estimate_tokens(self, text: str):
        return max(1, len(text) // 4)

    # =========================
    # DEBUG
    # =========================
    def debug(self, contexts: List[str]):
        print("\n====== CONTEXT DEBUG ======")
        for i, c in enumerate(contexts):
            print(f"\n--- Context {i} ---")
            print(c[:300])
        print("====== END ======\n")
