from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import time

from RAG import sparse_retrieval, dense_retrieval, rrf, rerank


# LOAD PICKLE FILES
def load_pickle(path, filetype: str):
    print(f"\nChecking if {path} exists")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print("File found!")

    try:
        print(f"\nLoading: {filetype}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
            print(f'Filetype: {type(obj)}')
            return obj
    except Exception as e:
        print(f"\nERROR loading {filetype}:")
        print(f"  {e}")
        raise


# LOAD FAISS INDEX
def load_faiss(path):
    print(f"\nChecking if {path} exists")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print("File found!")

    try:
        print(f"\nLoading: FAISS Index")

        dense_index = faiss.read_index(path)
        return dense_index

    except Exception as e:
        print(f"\nERROR loading FAISS index:")
        print(f"  {e}")
        raise


def fetch_documents(reranked_docs: list, chunks):

    evidence = []

    for retrieved_dict in reranked_docs:
        evidence.append(
            {
            "chunk": chunks[retrieved_dict['chunk_index']],
            "chunk_index": retrieved_dict['chunk_index'],
            "score": retrieved_dict['score']
            } 
        )

    return evidence


# WHY CLASS? WHY CONCURRENCY?
# As it may become clear from the older versions, originally the files took ~16 seconds to load sequentially
# And another ~3 seconds per query sequentially executed.
# Which is why I implemented concurrency using Threadpoolexecutor first
# Then I noticed a separate problem - for every query we were loading the models, loading the files and then querying.
# So I split the huge orchestration function into 2 - one to load the files and one to query.
# That way whenever needed I could maintain the loaded files in session
# However that caused a lot of ugly function calls and return statements 
# so I implemented a small class.
# Ultimately both these fixes were necessary since the planner agent would go back and forth with the retrieval process and query multiple times

class RAGTool:

    def __init__(self):
        self.embedding_model: SentenceTransformer
        self.reranker: CrossEncoder
        self.chunks: list
        self.dense_index: faiss.Index
        self.bm25: BM25Okapi

    def load(self):

        with ThreadPoolExecutor(max_workers=5) as executor:

            embedding_future = executor.submit(
                SentenceTransformer,
                "BAAI/bge-base-en-v1.5"
            )

            reranker_future = executor.submit(
                CrossEncoder,
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

            chunks_future = executor.submit(
                load_pickle,
                "src\\RAG\\chunk_store.pkl",
                "Chunks"
            )

            dense_index_future = executor.submit(
                load_faiss,
                "src\\RAG\\dense_index.faiss"
            )

            bm25_future = executor.submit(
                load_pickle,
                "src\\RAG\\bm25_index.pkl",
                "BM25"
            )

            self.embedding_model = embedding_future.result()
            self.reranker_model = reranker_future.result()
            self.chunks = chunks_future.result()
            self.dense_index = dense_index_future.result()
            self.bm25 = bm25_future.result()

    # This method runs dense+sparse retrieval in parallel then 
    # runs rrf and reranking in sequence.
    # At the end we are left with top_k documents for a query.
    def query(self, user_query: str):

        with ThreadPoolExecutor(max_workers=2) as executor:

            dense_future = executor.submit(
                dense_retrieval.dense_retrieval_orchestration,
                user_query,
                self.embedding_model,
                self.chunks,
                self.dense_index
            )

            sparse_future = executor.submit(
                sparse_retrieval.sparse_retrieval_orchestration,
                user_query,
                self.bm25
            )

            dense_candidates = dense_future.result()
            sparse_candidates = sparse_future.result()

        rrf_candidates = rrf.reciprocal_rank_fusion(
            dense_candidates,
            sparse_candidates
        )

        reranked_candidates = rerank.rerank_candidates(
            user_query,
            rrf_candidates,
            self.chunks,
            self.reranker_model
        )

        return fetch_documents(
            reranked_candidates,
            self.chunks
        )



if __name__ == "__main__":

    rag = RAGTool()

    # Load all the files and models into session
    rag.load()

    # Since files are loaded, multiple quries do not pose much of an issue.
    # queries = ("Lionel Messi")
    # for query in queries:
    #     final_evidence = rag.query(query)
    #     print(final_evidence)

    query = "Steve Jobs"
    final_evidence = rag.query(query)
    print(final_evidence)
    

