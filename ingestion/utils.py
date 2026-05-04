import hashlib
import re
from typing import List
from langchain_core.documents import Document


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    seen = set()
    unique = []

    for chunk in chunks:
        text = chunk.page_content or ""
        norm_text = _normalize_text(text)

        # combine content + source (optional)
        source = chunk.metadata.get("source", "")
        key_raw = f"{source}:{norm_text}"

        key = hashlib.md5(key_raw.encode()).hexdigest()

        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique