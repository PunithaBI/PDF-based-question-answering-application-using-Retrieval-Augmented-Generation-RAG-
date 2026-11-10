import uuid

from qdrant_client import QdrantClient, models

from app.config import (
    QDRANT_PATH,
    QDRANT_COLLECTION
)

from app.services.embedding_service import (
    embed_texts,
    get_model
)


_client = None


def get_client() -> QdrantClient:
    """
    Return one shared local Qdrant client.
    """
    global _client

    if _client is None:
        _client = QdrantClient(
            path=str(QDRANT_PATH)
        )

    return _client


def ensure_collection():
    """
    Create the collection if it does not exist.
    """
    client = get_client()

    if client.collection_exists(QDRANT_COLLECTION):
        return

    dimension = get_model().get_embedding_dimension()

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE
        )
    )


def upsert_chunks(
    document_id: str,
    chunks: list[dict],
    vectors: list[list[float]]
):
    """
    Save document chunks and their embeddings.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            "Chunk count and vector count do not match."
        )

    ensure_collection()

    points = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document_id}:{chunk['chunk_id']}"
            )
        )

        payload = {
            "document_id": document_id,
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "page": chunk["page"],
            "text": chunk["text"],
            "token_count_estimate": chunk.get(
                "token_count_estimate", 0
            ),
            "char_count": chunk.get(
                "char_count", len(chunk["text"])
            )
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    if points:
        get_client().upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True
        )

    return len(points)


def search_chunks(
    document_id: str,
    question: str,
    limit: int = 5
) -> list[dict]:
    """
    Search chunks belonging to one document.
    """
    ensure_collection()

    vectors, _ = embed_texts(
        [question],
        is_query=True
    )

    query_vector = vectors[0]

    response = get_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    )
                )
            ]
        ),
        limit=limit,
        with_payload=True
    )

    results = []

    for point in response.points:
        results.append({
            **point.payload,
            "dense_score": float(point.score)
        })

    return results


def count_document(document_id: str) -> int:
    """
    Count indexed chunks for one document.
    """
    ensure_collection()

    result = get_client().count(
        collection_name=QDRANT_COLLECTION,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    )
                )
            ]
        ),
        exact=True
    )

    return result.count


def delete_document(document_id: str):
    """
    Delete all vectors belonging to a document.
    """
    ensure_collection()

    get_client().delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        )
                    )
                ]
            )
        ),
        wait=True
    )