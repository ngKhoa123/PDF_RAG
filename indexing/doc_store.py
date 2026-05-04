import json
import os
import uuid
from typing import List
from langchain_core.documents import Document


class DocStore:
    def __init__(self, base_path="storage/docs"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _path(self, user_id: str) -> str:
        return os.path.join(self.base_path, f"{user_id}.json")

    # =========================
    # SAVE (append + id + dedup)
    # =========================
    def save(self, user_id: str, docs: List[Document]):
        path = self._path(user_id)

        existing_data = []
        existing_ids = set()

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            for item in existing_data:
                if "id" not in item:
                    item["id"] = str(uuid.uuid4())  # assign new id

            existing_ids = {item["id"] for item in existing_data}

        new_data = []

        for doc in docs:
            doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))

            if doc_id in existing_ids:
                continue

            new_data.append({
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        all_data = existing_data + new_data

        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False)

    # =========================
    # LOAD
    # =========================
    def load(self, user_id: str) -> List[Document]:
        path = self._path(user_id)

        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(
                page_content=item["content"],
                metadata={
                    **item["metadata"],
                    "doc_id": item["id"]
                }
            )
            for item in data
        ]

    # =========================
    # DELETE (by doc_id)
    # =========================
    def delete(self, user_id: str, doc_id: str):
        path = self._path(user_id)

        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = [item for item in data if item["id"] != doc_id]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # =========================
    # CLEAR (user reset)
    # =========================
    def clear(self, user_id: str):
        path = self._path(user_id)
        if os.path.exists(path):
            os.remove(path)