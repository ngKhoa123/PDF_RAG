import os
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    def __init__(self, base_path: str = "data/vectors"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # =========================
    # PATH
    # =========================
    def _get_path(self, user_id: str):
        return os.path.join(self.base_path, user_id)

    # =========================
    # SAVE
    # =========================
    def save(self, user_id: str, vector_store):
        path = self._get_path(user_id)
        os.makedirs(path, exist_ok=True)

        vector_store.save_local(path)
        print(f"[VECTOR STORE] Saved at {path}")

    # =========================
    # LOAD
    # =========================
    def load(self, user_id: str, embedder):
        path = self._get_path(user_id)

        if not os.path.exists(path):
            return None

        print(f"[VECTOR STORE] Loaded from {path}")

        return FAISS.load_local(
            path,
            embedder,
            allow_dangerous_deserialization=True
        )

    # =========================
    # DELETE (optional)
    # =========================
    def delete(self, user_id: str):
        path = self._get_path(user_id)

        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
            print(f"[VECTOR STORE] Deleted {path}")