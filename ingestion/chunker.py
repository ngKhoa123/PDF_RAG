from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import re


class Chunker:
    """
    Section-aware + citation-safe chunker:
    - split theo section (abstract/introduction/...)
    - expand quanh "we propose"
    - loại bỏ citation/reference noise ngay từ ingestion
    - deduplicate
    """

    def __init__(self, chunk_size=900, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # =========================
    # MAIN
    # =========================
    def chunk(self, docs: List[Document], user_id: str, file_name: str):
        results = []
        seen = set()

        full_text = "\n".join([d.page_content for d in docs if d.page_content])
        sections = self._split_sections(full_text)

        chunk_id = 0

        for section_name, section_text in sections:
            importance = self._get_section_importance(section_name)

            sub_docs = [Document(page_content=section_text)]
            chunks = self.splitter.split_documents(sub_docs)

            for chunk in chunks:
                text = (chunk.page_content or "").strip()
                if not text or len(text) < 80:
                    continue

                lower = text.lower()

                if self._is_noise(lower):
                    continue

                if "we propose" in lower or "this paper" in lower:
                    text = self._expand_context(text, section_text)

                key = hashlib.md5(text.encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)

                metadata = {
                    "chunk_id": chunk_id,
                    "user_id": user_id,
                    "file_name": file_name,
                    "doc_id": file_name,
                    "section": section_name,
                    "importance": importance
                }

                results.append(Document(page_content=text, metadata=metadata))
                chunk_id += 1

        print(f"[CHUNKER] Final chunks: {len(results)}")
        return results

    # =========================
    # SECTION SPLIT
    # =========================
    def _split_sections(self, text: str):
        pattern = re.compile(
            r"\n(?=(abstract|introduction|background|method|methods|approach|model|experiment|results|conclusion)s?\b)",
            re.IGNORECASE
        )

        parts = pattern.split(text)

        sections = []
        current = "unknown"
        buffer = ""

        for part in parts:
            p = part.lower().strip()

            if p in ["abstract", "introduction", "background",
                     "method", "methods", "approach",
                     "model", "experiment", "results", "conclusion"]:
                if buffer:
                    sections.append((current, buffer.strip()))
                current = p
                buffer = ""
            else:
                buffer += " " + part

        if buffer:
            sections.append((current, buffer.strip()))

        return sections

    # =========================
    # IMPORTANCE
    # =========================
    def _get_section_importance(self, section: str):
        section = section.lower()

        if section == "abstract":
            return 5
        if section == "introduction":
            return 4
        if section in ["method", "model", "approach"]:
            return 2
        if section in ["experiment", "results"]:
            return 1

        return 0

    # =========================
    # EXPAND CONTEXT
    # =========================
    def _expand_context(self, chunk_text: str, section_text: str, window=600):
        idx = section_text.lower().find(chunk_text.lower()[:50])
        if idx == -1:
            return chunk_text

        start = max(0, idx - window)
        end = min(len(section_text), idx + len(chunk_text) + window)

        return section_text[start:end]

    # =========================
    # HARD NOISE FILTER (CRITICAL)
    # =========================
    def _is_noise(self, text: str):
        # citation patterns
        if re.search(r"\[\d+\]", text):
            return True

        if "et al." in text:
            return True

        if "references" in text:
            return True

        if "proceedings of" in text:
            return True

        if "conference" in text:
            return True

        if "acl" in text:
            return True

        if "copyright" in text or "all rights reserved" in text:
            return True

        return False