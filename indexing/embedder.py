from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings


class Embedder:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL

        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": "cpu",   # đổi thành "cuda" nếu có GPU
            },
            encode_kwargs={
                "normalize_embeddings": True,  # rất quan trọng cho cosine
                "batch_size": 32
            }
        )

    def get(self):
        return self.model