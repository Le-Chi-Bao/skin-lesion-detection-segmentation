"""Structure-aware recursive splitting with lossless offsets into canonical block text."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from app.rag.models import Chunk, Document


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 384
    overlap: int = 48
    # UTF-8 bytes are a conservative token budget, independent of model downloads.
    # Dense additionally checks its real tokenizer and never silently truncates passages.
    token_metric: str = "utf8-bytes-v1"
    version: str = "recursive-blocks-1"

    def __post_init__(self):
        if (type(self.max_tokens) is not int or type(self.overlap) is not int
                or self.max_tokens < 4 or not 0 <= self.overlap < self.max_tokens
                or self.token_metric != "utf8-bytes-v1"):
            raise ValueError("invalid_chunk_config")


def token_count(text: str) -> int:
    return len(text.encode("utf-8"))


class RecursiveChunker:
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    def _split(self, text: str, start: int, end: int, depth: int = 0) -> list[tuple[int, int]]:
        if token_count(text[start:end]) <= self.config.max_tokens:
            return [(start, end)]
        separators = (r"\n\n", r"\n", r"(?<=[.!?。])\s+", r"\s+")
        if depth == len(separators):
            # Last resort: split only between Unicode code points, never drop characters.
            pieces, left, size = [], start, 0
            for pos in range(start, end):
                width = token_count(text[pos])
                if size + width > self.config.max_tokens:
                    pieces.append((left, pos))
                    left, size = pos, 0
                size += width
            return pieces + [(left, end)]
        pieces, left = [], start
        for match in re.finditer(separators[depth], text[start:end]):
            right = start + match.end()
            if right > left:
                pieces.extend(self._split(text, left, right, depth + 1))
            left = right
        if left < end:
            pieces.extend(self._split(text, left, end, depth + 1))
        return pieces

    def _windows(self, text: str, start: int, end: int):
        pieces = self._split(text, start, end)
        left, right = pieces[0]
        for next_left, next_right in pieces[1:]:
            if token_count(text[left:next_right]) <= self.config.max_tokens:
                right = next_right
                continue
            yield left, right
            # Prefer whole trailing structural fragments for overlap. Fall back to no
            # overlap when a complete list/table row cannot fit the configured budget.
            candidates = [a for a, b in pieces if left < a < right
                          and token_count(text[a:right]) <= self.config.overlap
                          and token_count(text[a:next_right]) <= self.config.max_tokens]
            left = min(candidates) if candidates else next_left
            right = next_right
        yield left, right

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        record = document.record
        blocks = record["content"]
        if not blocks:
            return ()
        text = "\n\n".join(b["text"] for b in blocks)
        block_spans, offset = [], 0
        for block in blocks:
            block_spans.append((offset, offset + len(block["text"])))
            offset += len(block["text"]) + 2
        # A heading starts a new section and retains the current hierarchy in citations.
        sections, headings, section_start = [], [], 0
        current_path = ()
        for index, block in enumerate(blocks):
            if block["kind"] != "heading":
                continue
            position = block_spans[index][0]
            if position > section_start:
                sections.append((section_start, position, current_path))
            level = block["level"] or 1
            headings = [(n, t) for n, t in headings if n < level]
            headings.append((level, block["text"]))
            current_path = tuple(t for n, t in headings)
            section_start = position
        sections.append((section_start, len(text), current_path))
        config = asdict(self.config)
        chunks = []
        for start, end, heading_path in sections:
            for left, right in self._windows(text, start, end):
                covered = [i for i, (a, b) in enumerate(block_spans) if a < right and b > left]
                if not covered or not text[left:right].strip():
                    continue
                identity = [document.document_id, document.record_hash, config, left, right]
                chunk_id = "chunk:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                ranges = [{"block_index": i, "start": max(left, block_spans[i][0]) - block_spans[i][0],
                           "end": min(right, block_spans[i][1]) - block_spans[i][0]} for i in covered]
                citation = {"document_id": document.document_id,
                            "original_sha256": record["original"]["sha256"], "record_sha256": document.record_hash,
                            "source": record["source"], "title": record["metadata"]["title"],
                            "metadata": record["metadata"], "block_range": [min(covered), max(covered) + 1],
                            "block_char_ranges": ranges, "char_range": [left, right],
                            "char_basis": "content blocks joined by two newlines; Unicode code points; end exclusive",
                            "heading_path": list(heading_path), "batch_id": document.batch_id,
                            "event_id": document.event_id, "provenance_sha256": document.provenance_hash,
                            "rights": document.rights, "record_rights": record["rights"],
                            "extraction_status": record["extraction"]["status"],
                            "synthetic": document.synthetic, "chunk_config": config}
                chunks.append(Chunk(chunk_id, document.document_id, text[left:right], citation))
        return tuple(chunks)
