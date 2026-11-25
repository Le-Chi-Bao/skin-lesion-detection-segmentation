from copy import deepcopy
from dataclasses import replace

import pytest

from app.rag.chunking import ChunkConfig, RecursiveChunker, token_count
from rag_support import synthetic_batch


@pytest.mark.parametrize("budget,overlap", [(64, 0), (96, 24), (384, 48)])
def test_lossless_deterministic_ranges_and_citations(tmp_path, budget, overlap):
    manifest, _, importer = synthetic_batch(tmp_path)
    document = importer.import_manifest(manifest).documents[0]
    # Stress headings, nested lists, repeated text, table rows and very long unbroken Unicode.
    record = deepcopy(document.record)
    record["content"] += [
        {"kind": "heading", "level": 2, "text": "SYNTHETIC DEMO child"},
        {"kind": "paragraph", "level": None, "text": "đỏ xanh. " * 50},
        {"kind": "list_item", "level": 2, "text": "nested item\n" * 30},
        {"kind": "table", "level": None, "text": "A | B\n" + "cell | value\n" * 40},
        {"kind": "text", "level": None, "text": "🧪" * 100},
    ]
    document = replace(document, record=record)
    chunker = RecursiveChunker(ChunkConfig(budget, overlap))
    chunks = chunker.chunk(document)
    assert chunks == chunker.chunk(document)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    canonical = "\n\n".join(b["text"] for b in record["content"])
    covered = set()
    for chunk in chunks:
        citation = chunk.citation
        start, end = citation["char_range"]
        assert chunk.text == canonical[start:end]
        assert token_count(chunk.text) <= budget
        covered.update(range(start, end))
        for span in citation["block_char_ranges"]:
            assert 0 <= span["start"] < span["end"] <= len(record["content"][span["block_index"]]["text"])
        for required in ("source", "metadata", "original_sha256", "record_sha256", "batch_id", "event_id",
                         "provenance_sha256", "rights", "heading_path"):
            assert required in citation
        assert "page" not in citation and "doi" not in citation
    assert all(pos in covered for pos, character in enumerate(canonical) if not character.isspace())
    assert any(len(c.citation["heading_path"]) == 2 for c in chunks)
    changed = RecursiveChunker(ChunkConfig(budget + 1, overlap)).chunk(document)
    assert not {c.chunk_id for c in chunks} & {c.chunk_id for c in changed}
    revised = chunker.chunk(replace(document, record_hash="f" * 64))
    assert not {c.chunk_id for c in chunks} & {c.chunk_id for c in revised}


def test_overlap_keeps_complete_fragments(tmp_path):
    path, _, importer = synthetic_batch(tmp_path)
    doc = importer.import_manifest(path).documents[0]
    record = deepcopy(doc.record)
    record["content"] = [{"kind": "paragraph", "level": None, "text": "abcdefgh " * 100}]
    chunks = RecursiveChunker(ChunkConfig(64, 20)).chunk(replace(doc, record=record))
    assert any(a.citation["char_range"][1] > b.citation["char_range"][0] for a, b in zip(chunks, chunks[1:]))


def test_table_stays_whole_when_it_fits(tmp_path):
    path, _, importer = synthetic_batch(tmp_path)
    doc = importer.import_manifest(path).documents[1]
    chunks = RecursiveChunker().chunk(doc)
    assert any(doc.record["content"][-1]["text"] in c.text for c in chunks)


@pytest.mark.parametrize("budget,overlap", [(0, 0), (10, 10), (100, -1)])
def test_invalid_config(budget, overlap):
    with pytest.raises(ValueError):
        ChunkConfig(budget, overlap)
