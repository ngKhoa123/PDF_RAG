from typing import List, Generator as GenType
from langchain_community.llms import Ollama
from llm.prompt_manager import PromptManager
from retrieval.query_classifier import QueryClassifier
import re


class Generator:
    def __init__(
        self,
        model_name="phi3:mini",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
        )
        self.prompt_manager = PromptManager()
        self.max_tokens = max_tokens
        self.classifier = QueryClassifier()

        self.fallback = "I don't know based on the provided documents."

    # =========================
    # MAIN
    # =========================
    def generate(self, query: str, contexts: List[str], history: str = "") -> str:
        if not contexts:
            return self.fallback

        mode = self.classifier.classify(query)

        contexts = contexts[:5]

        prompt = self.prompt_manager.build(
            query=query,
            contexts=contexts,
            history=history,
            mode=mode
        )

        try:
            response = self.llm.invoke(prompt)

            if isinstance(response, dict):
                response = response.get("output", "")

            return self._postprocess(response)

        except Exception:
            return self.fallback

    # =========================
    # STREAM
    # =========================
    def stream(self, query: str, contexts: List[str], history: str = "") -> GenType[str, None, None]:
        if not contexts:
            yield self.fallback
            return

        prompt = self.prompt_manager.build(
            query=query,
            contexts=contexts[:5],
            history=history
        )

        try:
            for chunk in self.llm.stream(prompt):
                yield chunk
        except Exception:
            yield self.fallback

    # =========================
    # POSTPROCESS (FIXED)
    # =========================
    def _postprocess(self, response: str) -> str:
        if not response:
            return self.fallback

        response = response.strip()
        lower = response.lower()

        # ===== Model refusal =====
        if lower.startswith("i couldn't find") or lower.startswith("i don't know"):
            return self.fallback

        # ===== Clean =====
        response = re.sub(r"\s+", " ", response)
        words = response.split()

        # ===== Too short → reject =====
        if len(words) < 5:
            return self.fallback

        # # ===== Soft citation check (FIX QUAN TRỌNG) =====
        # citations = re.findall(r"\[\d+\]", response)

        # if not citations:
        #     response += "\n\n(Note: This answer is for reference and may not be fully accurate.)"

        # ===== Limit length =====
        # if len(words) > 120:
        #     response = " ".join(words[:120]) + "..."

        return response
