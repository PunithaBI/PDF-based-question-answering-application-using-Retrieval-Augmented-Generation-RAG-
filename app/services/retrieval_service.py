from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import (
    CANDIDATE_K,
    TOP_K,
    ENABLE_RERANKER,
    RERANKER_MODEL
)

from app.services.vector_store import search_chunks
from app.services.document_service import (
    get_chunks,
    get_document
)


@lru_cache(maxsize=1)
def get_reranker():
    """
    Load the cross-encoder only when reranking is enabled.
    """

    if not ENABLE_RERANKER:
        return None

    return CrossEncoder(
        RERANKER_MODEL,
        device="cpu"
    )


def dense_retrieval(
    document_id: str,
    question: str,
    limit: int = CANDIDATE_K
) -> list[dict]:

    return search_chunks(
        document_id=document_id,
        question=question,
        limit=limit
    )


def keyword_retrieval(
    document_id: str,
    question: str,
    limit: int = CANDIDATE_K
) -> list[dict]:

    chunks = get_chunks(document_id)

    if not chunks:
        return []

    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = question.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for chunk, score in ranked[:limit]:

        results.append({
            **chunk,
            "bm25_score": float(score)
        })

    return results


def reciprocal_rank_fusion(
    dense_results: list[dict],
    keyword_results: list[dict],
    limit: int = CANDIDATE_K,
    k: int = 60
) -> list[dict]:
    """
    Combine rankings without mixing their raw score scales.
    """

    combined = {}

    for rank, item in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = item["chunk_id"]

        combined.setdefault(
            chunk_id,
            {
                **item,
                "rrf_score": 0.0,
                "retrieval_methods": []
            }
        )

        combined[chunk_id]["rrf_score"] += (
            1 / (k + rank)
        )

        combined[chunk_id]["retrieval_methods"].append(
            "dense"
        )

    for rank, item in enumerate(
        keyword_results,
        start=1
    ):

        chunk_id = item["chunk_id"]

        if chunk_id not in combined:

            combined[chunk_id] = {
                **item,
                "rrf_score": 0.0,
                "retrieval_methods": []
            }

        combined[chunk_id]["rrf_score"] += (
            1 / (k + rank)
        )

        combined[chunk_id]["retrieval_methods"].append(
            "bm25"
        )

        combined[chunk_id]["bm25_score"] = item.get(
            "bm25_score", 0.0
        )

    results = sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return results[:limit]


def rerank_results(
    question: str,
    results: list[dict]
) -> list[dict]:
    """
    Reorder retrieved candidates using a cross-encoder.
    """

    model = get_reranker()

    if model is None or not results:
        return results

    pairs = [
        [question, item["text"]]
        for item in results
    ]

    scores = model.predict(pairs)

    for item, score in zip(results, scores):
        item["rerank_score"] = float(score)

    return sorted(
        results,
        key=lambda item: item["rerank_score"],
        reverse=True
    )


def retrieve_evidence(
    document_id: str,
    question: str,
    top_k: int = TOP_K
) -> list[dict]:

    document = get_document(document_id)

    if not document:
        return []

    if (
        document.get("status") != "indexed"
        or document.get("is_active", True) is False
    ):
        return []

    candidate_limit = max(
        CANDIDATE_K,
        top_k
    )

    dense_results = dense_retrieval(
        document_id,
        question,
        candidate_limit
    )

    keyword_results = keyword_retrieval(
        document_id,
        question,
        candidate_limit
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        keyword_results,
        candidate_limit
    )

    reranked_results = rerank_results(
        question,
        fused_results
    )

    return reranked_results[:top_k]