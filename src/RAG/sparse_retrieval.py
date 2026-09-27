from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from pathlib import Path
import numpy as np
import pickle
import os


# LOAD PICKLE FILES IF NEEDED
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


def sparse_retrieve(query: str, bm25: BM25Okapi, top_k: int = 40):

    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    sparse_candidates = []

    for index in top_indices:

        sparse_candidates.append(
            {
                "chunk_index": int(index),
                "score": float(scores[index])
            }
        )

    return sparse_candidates


def sparse_retrieval_orchestration(user_query, bm25: BM25Okapi):

    # INITATE/CONTROL PARAMETERS
    top_k = 40

    # CONTROL FLOW
    evidence_candidates = sparse_retrieve(query=user_query, bm25=bm25, top_k=top_k)
    # evidence_candidates = fetch_documents(sparse_retrieval, chunks)

    # RETURN EVIDENCE
    return evidence_candidates




if __name__ == "__main__":

    # LOAD CHUNKS/EMBEDDDINGS
    
    bm25 = load_pickle("src\\RAG\\bm25_index.pkl", "BM25")

    # GET USER QUESTION
    user_query = input("What would you like to ask: ")

    # DISPLAY RESULTS
    evidence = sparse_retrieval_orchestration(user_query, bm25)
    print(evidence)



