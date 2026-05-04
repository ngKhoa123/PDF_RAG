from typing import List


class PromptManager:

    # =========================
    # LENGTH CONTROL
    # =========================
    def _get_length_instruction(self, mode: str) -> str:
        return {
            "summary": "Answer in 1–2 concise sentences.",
            "qa": "Answer in 2–3 concise sentences.",
            "explain": "Explain clearly in 3–5 sentences.",
            "compare": "Compare using 2–3 key points.",
            "extract": "Return only exact text.",
        }.get(mode, "Be concise.")

    # =========================
    # SYSTEM PROMPT
    # =========================
    def _build_system_prompt(self, mode: str) -> str:
        length_instruction = self._get_length_instruction(mode)

        base_rules = f"""
You are a reliable AI assistant.

{length_instruction}

STRICT RULES:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- Do NOT hallucinate.
- Do NOT include unrelated or out-of-context information.
- Ignore noisy or irrelevant parts of the context.
- Prefer HIGH-LEVEL answers unless explicitly asked for details.
- If the answer is not clearly supported by the context, say:
  "I don't know based on the provided documents."
"""

        # =========================
        # MODE RULES
        # =========================

        if mode == "summary":
            return base_rules + """
Summary Rules:
- Answer what the document/paper is about.
- Focus ONLY on the main idea or purpose.
- Keep the answer high-level.
- Do NOT explain how the system/model works.
- Do NOT include technical details (e.g., layers, attention types, masking).
"""

        elif mode == "qa":
            return base_rules + """
QA Rules:
- Answer directly and clearly.
- Keep it simple.
- Avoid unnecessary technical depth unless asked.
"""

        elif mode == "explain":
            return base_rules + """
Explain Rules:
- Explain clearly and logically.
- You MAY include technical details if needed.
- Stay grounded in the context.
"""

        elif mode == "compare":
            return base_rules + """
Compare Rules:
- Highlight key differences.
- Keep comparison simple and clear.
"""

        elif mode == "extract":
            return """
You are a strict extraction system.

Rules:
- Return ONLY exact text from context.
- Do NOT paraphrase.
- Do NOT add anything.
- If not found, say: "No relevant information found."
"""

        return base_rules

    # =========================
    # BUILD PROMPT
    # =========================
    def build(
        self,
        query: str,
        contexts: List[str],
        history: str = "",
        mode: str = "qa"
    ) -> str:
  
        q_lower = query.lower()
        if "what is this paper about" in q_lower:
            mode = "summary"

        system_prompt = self._build_system_prompt(mode)
        context_block = self._format_context(contexts)

        return f"""
{system_prompt}

Conversation history:
{history}

Context:
{context_block}

Question:
{query}

Answer:
"""

    # =========================
    # FORMAT CONTEXT
    # =========================
    def _format_context(self, contexts: List[str]) -> str:
        if not contexts:
            return "No context provided."

        return "\n\n".join([
            ctx.strip()
            for ctx in contexts
        ])