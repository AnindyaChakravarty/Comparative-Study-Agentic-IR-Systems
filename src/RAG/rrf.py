from collections import defaultdict

def reciprocal_rank_fusion(
    dense_candidates: list,
    sparse_candidates: list,
    rrf_k: int = 60
):

    rrf_scores = defaultdict(float)

    for rank, candidate in enumerate(dense_candidates, start=1):
        chunk_index = candidate["chunk_index"]
        rrf_scores[chunk_index] += 1 / (rrf_k + rank)

    for rank, candidate in enumerate(sparse_candidates, start=1):
        chunk_index = candidate["chunk_index"]
        rrf_scores[chunk_index] += 1 / (rrf_k + rank)

    fused_candidates = [
        {
            "chunk_index": chunk_index,
            "score": score
        }
        for chunk_index, score in rrf_scores.items()
    ]

    fused_candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    return fused_candidates