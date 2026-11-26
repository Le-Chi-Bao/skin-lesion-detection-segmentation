"""Local kit integrity, real Chroma persistence and live revocation regressions."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from app.rag.chunking import RecursiveChunker
from app.rag.local_corpus import LocalKitImporter, consent_for
from app.rag.models import Chunk, RAGError
from app.rag.service import RAGService
from app.rag.vector_store import ChromaDenseRetriever


@pytest.fixture
def local_kit(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    original = Path("bo_tai_lieu_y_khoa_tieng_viet_pdf_kit")
    shutil.copytree(original / "metadata", kit / "metadata")
    shutil.copytree(original / "pdf_originals", kit / "pdf_originals")
    markdown = tmp_path / "markdown"
    shutil.copytree(Path("data/rag/markdown"), markdown)
    approval = tmp_path / "consent.json"
    approval.write_text(json.dumps(consent_for(kit, markdown)), encoding="utf-8")
    return kit, markdown, approval


class CountingEmbedder:
    config = SimpleNamespace(batch_size=16, dimensions=384)
    persistent_fingerprint = "test-local-384-v1"

    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] = 1
            vectors[i, 1] = len(text) / 100
        return vectors

    def embed_query(self, text):
        return np.array([1, 0] + [0] * 382, dtype=np.float32)


def test_eight_documents_preserve_page_heading_and_review(local_kit):
    kit, markdown, approval = local_kit
    importer = LocalKitImporter(kit, markdown, approval)
    result = importer.import_manifest(approval)
    assert result.status == "VERIFIED" and len(result.documents) == 8
    for doc in result.documents:
        chunks = RecursiveChunker().chunk(doc)
        assert chunks == RecursiveChunker().chunk(doc)
        assert {c.citation["pdf_page"] for c in chunks} == set(range(1, doc.record["metadata"]["pdf_pages"] + 1))
        for chunk in chunks:
            citation = chunk.citation
            assert citation["human_review_status"] == "needs_human_review"
            assert citation["conversion_status"] == "NEEDS_MANUAL_REVIEW"
            assert citation["rights"]["scope"] == "local_only"
            assert citation["rights"]["page_review_claim"] == "none"
            for interval in citation["block_char_ranges"]:
                assert doc.record["content"][interval["block_index"]]["pdf_page"] == citation["pdf_page"]
            if citation["heading_path"]:
                assert citation["heading_path"][-1] in "\n".join(b["text"] for b in doc.record["content"])


@pytest.mark.parametrize("change", ["markdown", "pdf", "metadata", "revocation", "extra", "missing"])
def test_local_integrity_and_consent_fail_closed(local_kit, change):
    kit, markdown, approval = local_kit
    importer = LocalKitImporter(kit, markdown, approval)
    service = RAGService()
    assert service.ingest(approval, importer).status == "VERIFIED"
    assert service.search("viêm da", "BM25").hits
    if change == "markdown":
        file = next(markdown.glob("*.md"))
        file.write_bytes(file.read_bytes() + b"\nchanged")
    elif change == "pdf":
        file = next((kit / "pdf_originals").glob("*.pdf"))
        file.write_bytes(file.read_bytes() + b"changed")
    elif change == "metadata":
        file = kit / "metadata/documents.json"
        file.write_bytes(file.read_bytes() + b"\n")
    elif change == "revocation":
        grant = json.loads(approval.read_bytes())
        grant["indexing"] = "denied"
        approval.write_text(json.dumps(grant), encoding="utf-8")
    elif change == "extra":
        (markdown / "extra.md").write_text("unexpected")
    else:
        next(markdown.glob("*.md")).rename(markdown / "missing.txt")
    assert importer.import_manifest(approval).status == "INVALID"
    assert service.search("viêm da", "Hybrid").status == "EMPTY_INDEX"
    assert not service.bm25.chunks and not service.dense.chunks


def sample(identity, text):
    return Chunk(identity, "doc", text, {"source": {"locator": "https://example.org/article"},
                                        "title": "Example", "pdf_page": 2})


def test_real_chroma_reuses_vectors_and_filters_inactive_ids(tmp_path):
    first = CountingEmbedder()
    path = tmp_path / "chroma"
    index = ChromaDenseRetriever(first, path)
    chunks = [sample("one", "viêm da"), sample("two", "mụn trứng cá"), sample("three", "viêm da")]
    index.index(chunks)
    assert len(first.texts) == 2 and index.collection.count() == 3
    assert index.embedded_count == 2
    second = CountingEmbedder()
    reopened = ChromaDenseRetriever(second, path)
    reopened.index(chunks)
    assert not second.texts and reopened.reused_count == 3
    assert reopened.collection.count() == 3
    assert len(reopened.search("viêm da", 10)) == 3
    # Same text from a new source/chunk uses cached embedding; old rows stay on disk.
    current = [sample("new-version", "viêm da")]
    reopened.index(current)
    assert not second.texts and reopened.collection.count() == 4
    assert [h.chunk.chunk_id for h in reopened.search("viêm da", 10)] == ["new-version"]
    reopened.clear()
    assert reopened.search("viêm da") == ()


def test_chroma_fingerprint_separates_models(tmp_path):
    first = ChromaDenseRetriever(CountingEmbedder(), tmp_path / "chroma")
    first.index([sample("one", "viêm da")])
    embedder = CountingEmbedder()
    embedder.persistent_fingerprint = "test-local-384-v2"
    second = ChromaDenseRetriever(embedder, tmp_path / "chroma")
    second.index([sample("one", "viêm da")])
    assert len(embedder.texts) == 1 and second.collection.name != first.collection.name


def test_local_config_disallows_cloud(tmp_path):
    from app.rag.runtime import create_service

    values = json.loads(Path("config/rag.local.json").read_bytes())
    values["generation"] = {"enabled": True}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(RAGError, match="local_consent_does_not_allow_cloud_processing"):
        create_service(config_path=path)


def test_actual_e5_tokenizer_rechunks_without_truncation(local_kit):
    pytest.importorskip("transformers")
    from app.rag.chunking import E5BudgetChunker
    from app.rag.config import ChunkConfig, EmbeddingConfig
    from app.rag.retrieval import LocalE5Embedder

    model = Path("model_store/embeddings/multilingual-e5-small")
    if not model.is_dir():
        pytest.skip("E5 Small local tokenizer not installed")
    kit, markdown, approval = local_kit
    documents = LocalKitImporter(kit, markdown, approval).import_manifest(approval).documents
    chunker = E5BudgetChunker(ChunkConfig(max_tokens=1400, overlap=160),
                              LocalE5Embedder(EmbeddingConfig(model=str(model), dimensions=384)))
    raw_max, final_max = 0, 0
    for document in documents:
        chunks = chunker.chunk(document)
        raw = RecursiveChunker(chunker.config).chunk(document)
        lengths = lambda items: [len(ids) for ids in chunker._tokenizer(
            ["passage: " + c.text.strip() for c in items], truncation=False, verbose=False)["input_ids"]]
        raw_max = max(raw_max, max(lengths(raw)))
        final_max = max(final_max, max(lengths(chunks)))
        covered = {index: set() for index in range(len(document.record["content"]))}
        for chunk in chunks:
            for span in chunk.citation["block_char_ranges"]:
                covered[span["block_index"]].update(range(span["start"], span["end"]))
        for index, block in enumerate(document.record["content"]):
            assert covered[index] == set(range(len(block["text"])))
    assert raw_max > 512 and final_max <= 512
