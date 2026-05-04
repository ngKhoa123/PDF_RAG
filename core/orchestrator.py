from typing import List, Optional, Dict
from collections import defaultdict

from config.settings import settings

# ingestion
from ingestion.pipeline import IngestionPipeline

# indexing
from indexing.embedder import Embedder
from indexing.index_manager import IndexManager

# retrieval
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

# processing
from processing.query_rewriter import QueryRewriter
from processing.context_builder import ContextBuilder
from processing.context_filter import ContextFilter

# llm
from llm.generator import Generator

# memory
from memory.memory_manager import MemoryManager


class RAGOrchestrator:
    def __init__(self):
        self.embedder = Embedder().get()
        self.index_manager = IndexManager(self.embedder)

        self.rewriter = QueryRewriter()
        self.builder = ContextBuilder()
        self.context_filter = ContextFilter()

        self.reranker = Reranker(mode="cross-encoder")

        self.generator = Generator(
            model_name=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE
        )

        self.memory = MemoryManager()

    # =========================
    # INGEST
    # =========================
    def ingest(self, files, user_id: str):
        pipeline = IngestionPipeline()
        chunks = pipeline.run(files, user_id)

        print(f"[ORCHESTRATOR] Total chunks: {len(chunks)}")
        self.index_manager.update(user_id, chunks)
        return len(chunks)

    # =========================
    # SELECT DOC WITH BOOST
    # =========================
    def _select_doc_id(self, ranked_docs, forced_doc_id=None):
        if forced_doc_id:
            return forced_doc_id

        score_map: Dict[str, float] = defaultdict(float)

        for i, doc in enumerate(ranked_docs):
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                continue

            text = (doc.page_content or "").lower()
            score = (len(ranked_docs) - i)

            if "we propose" in text:
                score += 10
            if "transformer" in text:
                score += 6

            if "we trained" in text or "gpu" in text:
                score -= 3

            score_map[doc_id] += score

        if not score_map:
            return None

        return max(score_map, key=score_map.get)

    # =========================
    # QUERY
    # =========================
    def query(self, query: str, user_id: str, doc_id: Optional[str] = None) -> str:
        try:
            print("\n========== DEBUG ==========")
            print("Query:", query)

            # ===== MEMORY =====
            history = self.memory.get_context(user_id, max_turns=3)

            # ===== REWRITE =====
            queries = self.rewriter.process(query)

            # ===== LOAD =====
            vs = self.index_manager.load(user_id)
            if vs is None:
                return "No documents found."

            all_docs = self.index_manager.doc_store.load(user_id)
            if not all_docs:
                return "No indexed documents found."

            print("All docs:", len(all_docs))

            # ===== RETRIEVE =====
            retriever = HybridRetriever(vs, all_docs)

            retrieved_docs = retriever.retrieve(
                queries,
                k=settings.TOP_K
            )

            print("retrieved_docs:", len(retrieved_docs))

            if not retrieved_docs:
                return "I don't know based on the provided documents."

            # =========================================
            # FORCE INTRO CHUNK (GLOBAL)
            # =========================================
            intro_docs = [
                d for d in all_docs
                if "we propose" in (d.page_content or "").lower()
            ]

            if intro_docs:
                print(">>> FORCE ADD INTRO")
                retrieved_docs.append(intro_docs[0])

            # ===== CHECK INTRO =====
            has_intro = any(
                "we propose" in (d.page_content or "").lower()
                for d in retrieved_docs
            )
            print("HAS INTRO:", has_intro)

            # ===== RERANK =====
            ranked_docs = self.reranker.rerank(query, retrieved_docs)
            print("ranked_docs:", len(ranked_docs))

            # ===== SELECT DOC =====
            selected_doc_id = self._select_doc_id(
                ranked_docs,
                forced_doc_id=doc_id
            )

            print("Selected doc_id:", selected_doc_id)

            candidate_docs = ranked_docs

            # ===== FILTER =====
            filtered_docs = self.context_filter.score(query, candidate_docs)
            print("filtered_docs:", len(filtered_docs))

            if not filtered_docs:
                print(">>> FILTER TOO STRONG → fallback")
                filtered_docs = candidate_docs[:8]

            # ===== DEBUG =====
            print("\n====== RAW DOC DEBUG ======")
            for i, doc in enumerate(filtered_docs[:5]):
                txt = doc.page_content or ""
                print(f"\n--- DOC {i} ---")
                print(txt[:200])
            print("====== END ======\n")

            # ===== BUILD CONTEXT =====
            contexts = self.builder.build(filtered_docs[:8], query)

            print("contexts:", len(contexts))
            self.builder.debug(contexts)
            
            if not contexts:
                print(">>> CONTEXT EMPTY → HARD FALLBACK")
                contexts = [
                    d.page_content for d in ranked_docs[:3]
                    if d.page_content
                ]

            if not contexts:
                return "I don't know based on the provided documents."

            # ===== GENERATE =====
            answer = self.generator.generate(
                query=query,
                contexts=contexts,
                history=history
            )

            # ===== MEMORY =====
            self.memory.add_turn(user_id, query, answer)

            return answer

        except Exception as e:
            return f"[SYSTEM ERROR] {str(e)}"