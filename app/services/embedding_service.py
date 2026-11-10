import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load the embedding model once.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Load the tokenizer once.
    """

    return AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )


def token_count(text: str) -> int:
    """
    Count tokens without truncation.
    """

    tokenizer = get_tokenizer()

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


def model_limit() -> int:
    """
    Return the embedding model's maximum sequence length.
    """

    return int(
        get_model().max_seq_length
    )


def embed_texts(
    texts: list[str],
    is_query: bool = False
) -> tuple[list[list[float]], dict]:

    if not texts:
        return [], {
            "model": EMBEDDING_MODEL,
            "dimension": None,
            "input_count": 0,
            "embedding_time_ms": 0
        }

    model = get_model()
    tokenizer = get_tokenizer()

    max_length = int(
        model.max_seq_length
    )

    prefix = ""

    # BGE recommends an instruction on the query side.
    if (
        is_query
        and EMBEDDING_MODEL.lower().startswith("baai/bge")
    ):
        prefix = (
            "Represent this sentence for searching "
            "relevant passages: "
        )

    model_inputs = [
        prefix + text
        for text in texts
    ]

    token_counts = [
        len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False
            )
        )
        for text in model_inputs
    ]

    oversized = [
        count
        for count in token_counts
        if count > max_length
    ]

    if oversized:
        raise ValueError(
            f"{len(oversized)} input(s) exceed the embedding "
            f"model limit of {max_length} tokens. "
            "Reduce the chunk size before indexing."
        )

    started = time.perf_counter()

    vectors = model.encode(
        model_inputs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32
    )

    elapsed_ms = round(
        (time.perf_counter() - started) * 1000,
        2
    )

    diagnostics = {
        "model": EMBEDDING_MODEL,
        "dimension": int(vectors.shape[1]),
        "model_max_sequence_length": max_length,
        "input_count": len(texts),
        "input_token_total": sum(token_counts),
        "input_token_max": max(token_counts),
        "embedding_time_ms": elapsed_ms,
        "query_instruction_used": bool(prefix)
    }

    return vectors.tolist(), diagnostics