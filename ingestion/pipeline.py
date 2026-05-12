from typing import List
from langchain_core.documents import Document

from storage.file_store import FileStore
from ingestion.loader import DocumentLoader
from ingestion.chunker import Chunker
from ingestion.utils import deduplicate_chunks


class IngestionPipeline:
    def __init__(self, verbose: bool = True):
        self.file_store = FileStore()
        self.loader = DocumentLoader(verbose=verbose)
        self.chunker = Chunker()
        self.verbose = verbose

    # =========================
    # RUN
    # =========================
    def run(self, files, user_id: str) -> List[Document]:
        if not files:
            raise ValueError("No files uploaded")

        all_chunks: List[Document] = []
        file_infos = []

        # ===== SAVE =====
        for f in files:
            try:
                info = self.file_store.save(user_id, f)
                file_infos.append(info)
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Save failed: {getattr(f, 'name', 'unknown')} → {e}")

        if not file_infos:
            raise RuntimeError("All file saves failed")

        # ===== LOAD + CHUNK =====
        for info in file_infos:
            file_path = info["path"]
            file_name = info["file_name"]

            if self.verbose:
                print(f"\n=== PROCESSING {file_name} ===")

            # -------- LOAD --------
            try:
                docs = self.loader.load(file_path)
                if self.verbose:
                    print(f"[PIPELINE] Loaded docs: {len(docs)}")
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Load failed: {file_name} → {e}")
                continue

            if not docs:
                if self.verbose:
                    print("[WARNING] No valid docs")
                continue

            # -------- CHUNK --------
            try:
                chunks = self.chunker.chunk(
                    docs,
                    user_id=user_id,
                    file_name=file_name
                )
                if self.verbose:
                    print(f"[PIPELINE] Raw chunks: {len(chunks)}")
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Chunking failed: {file_name} → {e}")
                continue

            if not chunks:
                if self.verbose:
                    print("[WARNING] No chunks created")
                continue

           # ===== FINAL FILTER =====
        # Cleaning đã được xử lý trong DocumentLoader._clean_text()
        
        clean_chunks = chunks
        if self.verbose:
            print(f"[PIPELINE] Clean chunks: {len(clean_chunks)}")
        
        all_chunks.extend(clean_chunks)

        # ===== DEDUP =====
        all_chunks = deduplicate_chunks(all_chunks)

        if self.verbose:
            print(f"[PIPELINE] Final chunks: {len(all_chunks)}")

        # ===== DEBUG SAMPLE  =====
        if self.verbose and all_chunks:
            print("\n[PIPELINE SAMPLE CHUNK]")
            print(all_chunks[0].page_content[:300])

        if not all_chunks:
            raise ValueError("No valid chunks created from any file")

        return all_chunks
