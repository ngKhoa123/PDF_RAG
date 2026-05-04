import argparse
import os

from ingestion.pipeline import IngestionPipeline
from indexing.embedder import Embedder
from indexing.index_manager import IndexManager


def validate_files(file_paths):
    valid = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue
        valid.append(path)
    return valid


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")

    parser.add_argument("--user_id", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--mode", choices=["update", "overwrite"], default="update")

    args = parser.parse_args()

    user_id = args.user_id
    file_paths = validate_files(args.files)

    if not file_paths:
        raise ValueError("No valid files to ingest")

    print(f"[INFO] Ingesting for user: {user_id}")
    print(f"[INFO] Files: {len(file_paths)}")

    # ===== INIT =====
    pipeline = IngestionPipeline()
    embedder = Embedder().get()
    index_manager = IndexManager(embedder)

    files = file_paths

    # ===== RUN =====
    try:
        chunks = pipeline.run(files, user_id)
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}")

    # ===== INDEX =====
    if args.mode == "overwrite":
        print("[INFO] Overwriting existing index")
        index_manager.save(user_id, chunks)
    else:
        print("[INFO] Updating index")
        index_manager.update(user_id, chunks)

    print(f"[SUCCESS] Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()