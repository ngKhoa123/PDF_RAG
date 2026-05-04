import os


class Settings:
    # ===== APP =====
    APP_NAME = "MiniLM Chatbox"
    ENV = os.getenv("ENV", "dev")

    # ===== PATH =====
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    STORAGE_DIR = os.path.join(BASE_DIR, "storage")

    FILES_DIR = os.path.join(STORAGE_DIR, "files")
    VECTOR_DIR = os.path.join(STORAGE_DIR, "vectors")
    DB_PATH = os.path.join(STORAGE_DIR, "app.db")

    # ===== EMBEDDING =====
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ===== CHUNKING =====
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    CHUNKING_METHOD = "semantic"

    # ===== RETRIEVAL =====
    TOP_K = 10
    ALPHA = 0.5  # hybrid search weight

    # ===== RERANK =====
    USE_RERANKER = True
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K = 5

    # ===== QUERY PROCESSING =====
    USE_QUERY_REWRITE = True
    USE_MULTI_QUERY = True
    NUM_QUERIES = 3

    # ===== LLM =====
    MODEL_NAME = os.getenv("MODEL_NAME", "phi3:mini")
    TEMPERATURE = 0.2

    # ===== MEMORY =====
    MEMORY_TYPE = "summary"
    MAX_HISTORY = 10

    # ===== EVALUATION =====
    EVAL_MODE = False
    LOG_QUERIES = True
    SAVE_RESPONSES = True


settings = Settings()