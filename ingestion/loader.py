import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)


class DocumentLoader:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # =========================
    # LOAD
    # =========================
    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = self._get_loader(file_path)

        try:
            docs = loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")

        if self.verbose:
            print(f"[LOADER] Raw docs: {len(docs)}")

        valid_docs = []

        for d in docs:
            text = (d.page_content or "").strip()

            if not text:
                continue

            # =========================
            # CLEAN TEXT
            # =========================
            text = self._clean_text(text)

            # skip sau khi clean
            if not text or len(text) < 50:
                continue

            metadata = d.metadata or {}

            metadata.update({
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": self._get_file_type(file_path)
            })

            d.page_content = text
            d.metadata = metadata

            valid_docs.append(d)

        if self.verbose:
            print(f"[LOADER] Valid docs: {len(valid_docs)}")
            if valid_docs:
                print("[LOADER] Sample:", valid_docs[0].page_content[:200])

        return valid_docs

    # =========================
    # CLEAN TEXT
    # =========================
    def _clean_text(self, text: str) -> str:
        lower = text.lower()

        if "provided proper attribution" in lower:
            return ""

        if "attention is all you need" in lower and len(text) < 500:
            return ""

        if "references" in lower:
            return ""

        # normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # =========================
    # LOADER FACTORY
    # =========================
    def _get_loader(self, file_path: str):
        ext = file_path.lower()

        if ext.endswith(".pdf"):
            return PyPDFLoader(file_path)

        elif ext.endswith(".txt"):
            return TextLoader(file_path, encoding="utf-8")

        elif ext.endswith(".docx"):
            return UnstructuredWordDocumentLoader(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    # =========================
    # HELPERS
    # =========================
    def _get_file_type(self, file_path: str) -> str:
        return os.path.splitext(file_path)[-1].replace(".", "").lower()