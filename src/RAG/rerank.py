from sentence_transformers import CrossEncoder


def rerank_candidates(
    query: str,
    candidates: list,
    chunks: list,
    reranker: CrossEncoder,
    top_k: int = 5
):

    pairs = [
        (query, chunks[candidate["chunk_index"]]["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        reranked.append(
            {
                "chunk_index": candidate["chunk_index"],
                "score": float(score)
            }
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked[:top_k]