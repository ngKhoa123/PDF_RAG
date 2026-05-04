from rank_bm25 import BM25Okapi
from typing import List, Dict
from collections import defaultdict
import re


class HybridRetriever:

    def __init__(self, vector_store, docs: List):
        self.vector_store = vector_store
        self.docs = docs

        self.texts = [doc.page_content for doc in docs]
        self.tokenized = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    # =========================
    # MAIN
    # =========================
    def retrieve(self, queries: List[str], k: int = 5, metadata_filter: Dict = None):
        dense_results = []
        sparse_results = []

        for q in queries:
            dense_results.extend(self._dense_search(q, k))
            sparse_results.extend(self._bm25_search(q, k))

        fused = self._rrf_fusion(dense_results, sparse_results)

        if metadata_filter:
            fused = self._filter_metadata(fused, metadata_filter)

        return fused[:k]

    # =========================
    # TOKENIZE
    # =========================
    def _tokenize(self, text: str):
        text = text.lower()
        return re.findall(r"\w+", text)

    # =========================
    # DENSE
    # =========================
    def _dense_search(self, query, k):
        return self.vector_store.similarity_search(query, k=k)

    # =========================
    # BM25
    # =========================
    def _bm25_search(self, query, k):
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.docs[i] for i in top_idx]

    # =========================
    # RRF FIX (chunk-level key)
    # =========================
    def _rrf_fusion(self, dense, sparse, k=60):
        scores = defaultdict(float)

        def get_key(doc):
            return (
                doc.metadata.get("doc_id"),
                doc.metadata.get("chunk_id"),
                hash(doc.page_content[:100])
            )

        def add_scores(results):
            seen = set()
            for rank, doc in enumerate(results):
                key = get_key(doc)

                if key in seen:
                    continue
                seen.add(key)

                scores[key] += 1 / (k + rank)

        add_scores(dense)
        add_scores(sparse)

        # map unique docs
        unique_docs = {}
        for doc in dense + sparse:
            key = get_key(doc)
            if key not in unique_docs:
                unique_docs[key] = doc

        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda d: scores[get_key(d)],
            reverse=True
        )

        return sorted_docs

    # =========================
    # FILTER
    # =========================
    def _filter_metadata(self, docs, metadata_filter):
        filtered = []

        for doc in docs:
            meta = getattr(doc, "metadata", {})
            if all(meta.get(k) == v for k, v in metadata_filter.items()):
                filtered.append(doc)

        return filtered