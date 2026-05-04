from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder

    # =========================
    # BUILD (new index)
    # =========================
    def build(self, docs: List[Document]):
        if not docs:
            raise ValueError("Cannot build vector store with empty documents")

        return FAISS.from_documents(docs, self.embedder)

    # =========================
    # SAVE
    # =========================
    def save(self, vs: FAISS, path: str):
        vs.save_local(path)

    # =========================
    # LOAD
    # =========================
    def load(self, path: str) -> FAISS:
        return FAISS.load_local(
            path,
            self.embedder,
            allow_dangerous_deserialization=True
        )

    # =========================
    # ADD (incremental update)
    # =========================
    def add(self, vs: FAISS, docs: List[Document]):
        if not docs:
            return vs

        vs.add_documents(docs)
        return vs

    # =========================
    # SEARCH
    # =========================
    def search(self, vs: FAISS, query: str, top_k: int = 5):
        return vs.similarity_search(query, k=top_k)

    # =========================
    # SEARCH WITH SCORE
    # =========================
    def search_with_score(self, vs: FAISS, query: str, top_k: int = 5):
        return vs.similarity_search_with_score(query, k=top_k)