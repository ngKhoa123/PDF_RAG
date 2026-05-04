import os
import shutil
import uuid
from typing import List
from indexing.vector_store import VectorStore
from indexing.doc_store import DocStore
from langchain_core.documents import Document


class IndexManager:
    def __init__(self, embedder, base_path="storage/vectors"):
        self.vector_store = VectorStore(embedder)
        self.doc_store = DocStore()
        self.base_path = base_path

        os.makedirs(base_path, exist_ok=True)

        # cache {user_id: vector_store}
        self.cache = {}

    # =========================
    # PATH
    # =========================
    def _get_path(self, user_id: str) -> str:
        return os.path.join(self.base_path, str(user_id))

    # =========================
    # LOAD
    # =========================
    def load(self, user_id: str):
        if user_id in self.cache:
            return self.cache[user_id]

        path = self._get_path(user_id)

        if not os.path.exists(path):
            return None

        vs = self.vector_store.load(path)
        self.cache[user_id] = vs
        return vs

    # =========================
    # SAVE (overwrite)
    # =========================
    def save(self, user_id: str, docs: List[Document]):
        docs = self._attach_metadata(user_id, docs)

        vs = self.vector_store.build(docs)

        path = self._get_path(user_id)
        self.vector_store.save(vs, path)
        self.doc_store.save(user_id, docs)

        self.cache[user_id] = vs

    # =========================
    # UPDATE (append)
    # =========================
    def update(self, user_id: str, new_docs: List[Document]):
        new_docs = self._attach_metadata(user_id, new_docs)

        vs = self.load(user_id)

        if vs:
            vs = self.vector_store.add(vs, new_docs)
        else:
            vs = self.vector_store.build(new_docs)

        path = self._get_path(user_id)
        self.vector_store.save(vs, path)
        self.doc_store.save(user_id, new_docs)

        self.cache[user_id] = vs

    # =========================
    # DELETE
    # =========================
    def delete(self, user_id: str):
        path = self._get_path(user_id)

        if os.path.exists(path):
            shutil.rmtree(path)

        self.doc_store.clear(user_id)

        self.cache.pop(user_id, None)

    # =========================
    # GET DOCS (debug / rerank)
    # =========================
    def get_docs(self, user_id: str) -> List[Document]:
        return self.doc_store.load(user_id)

    # =========================
    # METADATA
    # =========================
    def _attach_metadata(self, user_id: str, docs: List[Document]):
        for doc in docs:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}

            # stable id for traceability
            doc.metadata.setdefault("doc_id", str(uuid.uuid4()))
            doc.metadata.setdefault("user_id", user_id)
            doc.metadata.setdefault("source", "upload")

        return docs