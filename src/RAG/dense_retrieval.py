from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from pathlib import Path
import numpy as np
import pickle
import os

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

    
# CONVERT USER QUERY INTO EMBEDDING
def embed_query(query: str, embedding_model):

    query_embedding = embedding_model.encode(
    query,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
    )

    return query_embedding


# USING QUERY EMBEDDING RETRIEVE FROM FAISS INDEX
def dense_retrieve(query_embedding: np.ndarray, dense_index: faiss.Index, top_k: int = 40):

    evidence_candidates = []

    distances, indices = dense_index.search(query_embedding.reshape(1, -1),top_k)

    # FAISS returns a list of lists so we must extract the first element
    # i.e., our list of similarity scores and document indices

    similarity_scores = distances[0].tolist()
    chunk_indices = indices[0].tolist()

    for idx, score in enumerate(similarity_scores):
        evidence_candidates.append({'chunk_index': chunk_indices[idx], 'score': score})

    return evidence_candidates


# USING RETRIEVED INDICES RETRIEVE ACTUAL TEXT CHUNKS FROM CHUNKS PICLE FILE

def fetch_documents(dense_vectors: tuple, chunks):

    evidence = []
    similarity_scores = dense_vectors[0]
    indices = dense_vectors[1]

    # SOMETIMES FAISS RETURNS -1! HAD TO LEARN THAT THE HARD WAY :(

    for idx, chunk_index in enumerate(indices):

        if chunk_index == -1:
            continue

        evidence.append(
            {
            "chunk": chunks[chunk_index],
            "chunk_index": chunk_index,
            "score": similarity_scores[idx]
            } 
        )

    return evidence


def dense_retrieval_orchestration(user_query, embedding_model, chunks, dense_index):

    # INITATE/CONTROL PARAMETERS

    top_k = 40

    # CONTROL FLOW

    query_embedding = embed_query(user_query, embedding_model=embedding_model)
    evidence_candidates = dense_retrieve(query_embedding, dense_index, top_k=top_k)
    # evidence_candidates = fetch_documents(dense_retrieval, chunks)

    # RETURN EVIDENCE
    return evidence_candidates



if __name__ == "__main__":

    # SET MODEL IF TESTING MODULE

    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # LOAD CHUNKS/EMBEDDDINGS

    chunks = load_pickle("src\\RAG\\chunk_store.pkl", "Chunks")
    dense_index = load_faiss("src\\RAG\\dense_index.faiss")

    user_query = input("Enter your query:\n")

    # RESULTS

    evidence = dense_retrieval_orchestration(user_query, embedding_model, chunks, dense_index)
    print(evidence)