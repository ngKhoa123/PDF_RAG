import os
import shutil
from typing import Dict, Union


class FileStore:
    def __init__(self, base_path: str = "data/files"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # =========================
    # SAVE FILE
    # =========================
    def save(self, user_id: str, file: Union[str, object]) -> Dict:
        user_path = os.path.join(self.base_path, user_id)
        os.makedirs(user_path, exist_ok=True)

        # =========================
        # CASE 1: CLI (string path)
        # =========================
        if isinstance(file, str):
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")

            file_name = os.path.basename(file)
            file_path = os.path.join(user_path, file_name)

            shutil.copy(file, file_path)

            print(f"[FILE STORE] Copied: {file_name}")

            return {
                "path": file_path,
                "file_name": file_name,
                "user_id": user_id
            }

        # =========================
        # CASE 2: Streamlit upload
        # =========================
        file_name = getattr(file, "name", "unknown")
        file_path = os.path.join(user_path, file_name)

        content = file.read()

        if not content:
            raise ValueError("Uploaded file is empty")

        with open(file_path, "wb") as f:
            f.write(content)

        print(f"[FILE STORE] Saved: {file_name}")

        return {
            "path": file_path,
            "file_name": file_name,
            "user_id": user_id
        }

    # =========================
    # LIST FILES
    # =========================
    def list_files(self, user_id: str):
        user_path = os.path.join(self.base_path, user_id)

        if not os.path.exists(user_path):
            return []

        return os.listdir(user_path)