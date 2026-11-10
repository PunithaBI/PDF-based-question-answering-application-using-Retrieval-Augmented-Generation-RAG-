
import re

from app.config import (
    CHUNK_TARGET_TOKENS,
    CHUNK_OVERLAP_TOKENS
)

from app.services.embedding_service import (
    get_tokenizer,
    model_limit
)


def token_count(text: str) -> int:
    """Count tokens using the actual embedding tokenizer."""

    if not text:
        return 0

    return len(
        get_tokenizer().encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


def split_into_blocks(text: str) -> list[str]:
    """Split Markdown into logical paragraphs and blocks."""

    blocks = re.split(
        r"\n\s*\n",
        text
    )

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]


def split_oversized_block(
    text: str,
    max_tokens: int
) -> list[str]:
    """
    Split an oversized block using tokenizer boundaries.
    Markdown is retained as decoded text where possible.
    """

    tokenizer = get_tokenizer()

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    if not token_ids:
        return []

    parts = []

    for start in range(
        0,
        len(token_ids),
        max_tokens
    ):

        token_slice = token_ids[
            start:start + max_tokens
        ]

        part = tokenizer.decode(
            token_slice,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        if part:
            parts.append(part)

    return parts


def chunk_pages(
    pages: list[dict],
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[dict]:
    """
    Create page-aware chunks.

    Uses the embedding tokenizer for sizing and ensures
    each chunk remains below the model's token limit.
    """

    if target_tokens < 1:
        raise ValueError(
            "target_tokens must be greater than zero."
        )

    if overlap_tokens < 0:
        raise ValueError(
            "overlap_tokens cannot be negative."
        )

    # Reserve space for special tokens.
    safe_model_limit = model_limit() - 4

    if safe_model_limit < 1:
        raise ValueError(
            "Embedding model token limit is invalid."
        )

    effective_target = min(
        target_tokens,
        safe_model_limit
    )

    effective_overlap = min(
        overlap_tokens,
        effective_target - 1
    )

    chunks = []

    for page in pages:

        page_number = int(page["page"])
        page_text = page.get("text", "")

        if not page_text.strip():
            continue

        blocks = split_into_blocks(page_text)

        current_blocks = []
        current_tokens = 0

        def save_current_chunk():
            nonlocal current_blocks
            nonlocal current_tokens

            if not current_blocks:
                return

            chunk_text = "\n\n".join(
                current_blocks
            ).strip()

            if chunk_text:
                actual_tokens = token_count(chunk_text)

                if actual_tokens > safe_model_limit:
                    raise ValueError(
                        "Chunk exceeds the embedding model limit "
                        "after chunking."
                    )

                chunks.append({
                    "page": page_number,
                    "text": chunk_text,
                    "token_count_estimate": actual_tokens
                })

            current_blocks = []
            current_tokens = 0

        for block in blocks:

            block_tokens = token_count(block)

            # Split blocks that are too large to fit
            # inside one chunk.
            if block_tokens > effective_target:

                save_current_chunk()

                pieces = split_oversized_block(
                    block,
                    max_tokens=effective_target - 4
                )

                for piece in pieces:

                    piece_tokens = token_count(piece)

                    if piece_tokens > safe_model_limit:
                        raise ValueError(
                            "An oversized PDF block could not "
                            "be split within the model limit."
                        )

                    chunks.append({
                        "page": page_number,
                        "text": piece,
                        "token_count_estimate": piece_tokens
                    })

                continue

            # Flush before adding a block that exceeds
            # the target size.
            if (
                current_blocks
                and current_tokens + block_tokens
                > effective_target
            ):

                previous_blocks = current_blocks.copy()

                save_current_chunk()

                # Retain recent blocks for overlap.
                overlap_blocks = []
                overlap_count = 0

                for previous_block in reversed(
                    previous_blocks
                ):

                    previous_tokens = token_count(
                        previous_block
                    )

                    if (
                        overlap_count + previous_tokens
                        > effective_overlap
                    ):
                        break

                    overlap_blocks.insert(
                        0,
                        previous_block
                    )

                    overlap_count += previous_tokens

                current_blocks = overlap_blocks
                current_tokens = overlap_count

            current_blocks.append(block)
            current_tokens += block_tokens

        save_current_chunk()

    # Assign stable IDs within the document.
    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk["chunk_id"] = (
            f"chunk-{index:06d}"
        )

        chunk["chunk_index"] = index - 1

        chunk["char_count"] = len(
            chunk["text"]
        )

    return chunks
