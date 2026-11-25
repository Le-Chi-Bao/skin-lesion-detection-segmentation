import math
import os
import sys
from types import SimpleNamespace

import pytest

from app.rag.chunking import RecursiveChunker
from app.rag.models import Chunk, ModelUnavailable, RAGError, SearchHit
from app.rag.retrieval import (BM25Retriever, DenseRetriever, EmbeddingConfig, HybridRetriever,
                               LocalE5Embedder, RRFusion, tokenize)
from app.rag.service import RAGService
from rag_support import FakeEmbedder, synthetic_batch


def corpus(tmp_path):
    path, _, importer = synthetic_batch(tmp_path)
    return tuple(c for d in importer.import_manifest(path).documents for c in RecursiveChunker().chunk(d))


def test_bm25_vietnamese_english_topk_and_unicode(tmp_path):
    chunks = corpus(tmp_path)
    retriever = BM25Retriever(chunks)
    assert "tổn" in tokenize("TỔN THƯƠNG")
    assert tokenize("to\u0302\u0309n") == tokenize("tổn")
    for query in ("tổn thương da", "skin color border"):
        results = retriever.search(query, 2)
        assert results and "Quan sát" in results[0].chunk.citation["title"]
        assert all(hit.score > 0 for hit in results)
        assert len(results) <= 2
        assert results == retriever.search(query, 2)
    assert retriever.search("no_such_token_xyz") == ()
    assert retriever.search("") == ()


def test_dense_cosine_and_hybrid_with_fake_embeddings_only(tmp_path):
    chunks = corpus(tmp_path)
    dense = DenseRetriever(FakeEmbedder())
    dense.index(chunks)
    results = dense.search("skin skin da", 2)
    assert len(results) == 2 and results[0].score >= results[1].score
    assert all(-1.000001 <= h.score <= 1.000001 for h in results)
    hybrid = HybridRetriever(BM25Retriever(chunks), dense)
    results = hybrid.search("source nguồn", 3)
    assert results and len({h.chunk.chunk_id for h in results}) == len(results)
    assert all(h.mode == "Hybrid" for h in results)


def test_rrf_formula_duplicates_and_stable_ties():
    a, b = Chunk("a", "doc-a", "A", {}), Chunk("b", "doc-b", "B", {})
    ra, rb = SearchHit(a, 100, "BM25"), SearchHit(b, 0.01, "Dense")
    fused = RRFusion(60).fuse([[ra, ra, rb], [rb, ra]])
    assert [h.chunk.chunk_id for h in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == fused[0].score
    assert RRFusion().fuse([]) == ()


def test_empty_indexes_do_not_load_embeddings():
    class Forbidden:
        def embed_documents(self, texts):
            pytest.fail("Empty index must not load embeddings")

    dense = DenseRetriever(Forbidden())
    dense.index(())
    assert dense.search("skin") == ()
    assert BM25Retriever().search("skin") == ()


def test_non_finite_embeddings_rejected_and_old_index_cleared(tmp_path):
    dense = DenseRetriever(FakeEmbedder())
    chunks = corpus(tmp_path)
    dense.index(chunks)
    dense.embedder = SimpleNamespace(embed_documents=lambda texts: [[math.nan]] * len(texts))
    with pytest.raises(RAGError, match="invalid_embeddings"):
        dense.index(chunks)
    assert dense.chunks == () and dense.vectors is None


@pytest.mark.real_e5
def test_real_e5_local_smoke(tmp_path):
    # Actual E5 only. Strict mode makes a missing local model a failed verification gate.
    import numpy as np

    embedder = LocalE5Embedder(EmbeddingConfig(batch_size=2))
    manifest, _, importer = synthetic_batch(tmp_path)
    imported = importer.import_manifest(manifest)
    assert imported.status == "VERIFIED"
    documents = {d.document_id: d for d in imported.documents}
    chunks = tuple(c for d in imported.documents for c in RecursiveChunker().chunk(d))
    try:
        docs = embedder.embed_documents([c.text for c in chunks])
        query = embedder.embed_query("màu sắc da")
    except ModelUnavailable as exc:
        message = f"BLOCKED: real multilingual-e5-base NOT VERIFIED: {exc}"
        if os.environ.get("RAG_REQUIRE_REAL_E5") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    assert docs.shape == (len(chunks), 768) and query.shape == (768,)
    assert np.isfinite(docs).all() and np.isfinite(query).all()
    assert np.linalg.norm(docs, axis=1) == pytest.approx(np.ones(len(chunks)), abs=1e-5)
    assert np.linalg.norm(query) == pytest.approx(1, abs=1e-5)

    service = RAGService(embedder=embedder)
    try:
        assert service.ingest(manifest, importer).status == "VERIFIED"
        for text, title in (("skin lesion color border size", "Quan sát"),
                            ("nguồn tài liệu và bằng chứng truy xuất", "Nguồn"),
                            ("image segmentation mask bounding box", "Phân đoạn")):
            query_vector = embedder.embed_query(text)
            for mode in ("Dense", "Hybrid"):
                first = service.search(text, mode, 3)
                second = service.search(text, mode, 3)
                assert first.status == second.status == "OK"
                assert first.synthetic and len(first.hits) == 3
                assert title in first.hits[0].chunk.citation["title"]
                assert [h.chunk.chunk_id for h in first.hits] == [h.chunk.chunk_id for h in second.hits]
                assert [h.score for h in first.hits] == pytest.approx([h.score for h in second.hits], abs=1e-6)
                assert len({h.chunk.chunk_id for h in first.hits}) == 3
                for hit in first.hits:
                    doc = documents[hit.chunk.document_id]
                    canonical = "\n\n".join(b["text"] for b in doc.record["content"])
                    citation = hit.chunk.citation
                    start, end = citation["char_range"]
                    assert hit.chunk.text == canonical[start:end]
                    assert citation["record_sha256"] == doc.record_hash
                    assert citation["original_sha256"] == doc.record["original"]["sha256"]
                    assert citation["source"] == doc.record["source"]
                    assert citation["rights"] == doc.rights and citation["synthetic"]
                if mode == "Dense":
                    expected = service.dense.vectors @ query_vector
                    ordered = sorted(zip(service.dense.chunks, expected), key=lambda pair: (-pair[1], pair[0].chunk_id))[:3]
                    assert [h.chunk.chunk_id for h in first.hits] == [c.chunk_id for c, _ in ordered]
                    assert [h.score for h in first.hits] == pytest.approx([float(s) for _, s in ordered], abs=1e-6)
                else:
                    count = max(3, service.hybrid.candidates)
                    scores = {}
                    for ranking in (service.bm25.search(text, count), service.dense.search(text, count)):
                        for rank, hit in enumerate(ranking, 1):
                            identity = hit.chunk.chunk_id
                            scores[identity] = scores.get(identity, 0) + 1 / (60 + rank)
                    expected_ids = sorted(scores, key=lambda identity: (-scores[identity], identity))[:3]
                    assert [h.chunk.chunk_id for h in first.hits] == expected_ids
                    assert [h.score for h in first.hits] == pytest.approx([scores[i] for i in expected_ids])
    finally:
        service.close()


@pytest.mark.parametrize("k", [1, 60, 80.5])
def test_rrf_unequal_rankings_duplicates_and_cross_ranking_contributions(k):
    a, b, c, d = [SearchHit(Chunk(key, "doc-" + key, key, {}), 1, "BM25") for key in "abcd"]
    rankings = [[a, a, b, c, c, d], [c, a], []]
    fused = RRFusion(k).fuse(rankings, 10)
    expected = {"a": 1 / (k + 1) + 1 / (k + 2), "b": 1 / (k + 2),
                "c": 1 / (k + 3) + 1 / (k + 1), "d": 1 / (k + 4)}
    assert [h.chunk.chunk_id for h in fused] == sorted(expected, key=lambda key: (-expected[key], key))
    assert {h.chunk.chunk_id: h.score for h in fused} == pytest.approx(expected)
    assert RRFusion(k).fuse(rankings, 2) == fused[:2]


def test_rrf_ties_use_chunk_id_regardless_of_ranking_order():
    a, b = [SearchHit(Chunk(key, key, key, {}), 1, "Dense") for key in "ab"]
    fusion = RRFusion()
    assert fusion.fuse([[b, a], [a, b]]) == fusion.fuse([[a, b], [b, a]])
    assert [h.chunk.chunk_id for h in fusion.fuse([[b], [a]])] == ["a", "b"]


def test_e5_prefixes_pooling_offline_options_and_no_passage_truncation(monkeypatch):
    import numpy as np
    torch = pytest.importorskip("torch", reason="Install CV/Dense dependencies for E5 pooling unit test")

    observed = {"inputs": [], "options": []}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            observed["inputs"].append((texts, kwargs))
            if not kwargs.get("return_tensors"):
                return {"input_ids": [[1] * (600 if "TOO_LONG" in t else 3) for t in texts]}
            return Batch(input_ids=torch.tensor([[1, 2, 0]] * len(texts)),
                         attention_mask=torch.tensor([[1, 1, 0]] * len(texts)))

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[2., 0.], [0., 2.], [999., 999.]]]
                                                                  * len(kwargs["input_ids"])))

    def load_tokenizer(name, **kwargs):
        observed["options"].append(kwargs)
        return Tokenizer()

    def load_model(name, **kwargs):
        observed["options"].append(kwargs)
        return Model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModel=SimpleNamespace(from_pretrained=load_model)))
    embedder = LocalE5Embedder()
    docs = embedder.embed_documents(["  tiếng Việt  "])
    embedder.embed_query("  skin  ")
    assert np.allclose(docs[0], [1 / math.sqrt(2)] * 2)
    inputs = [text for texts, options in observed["inputs"] for text in texts]
    assert "passage: tiếng Việt" in inputs and "query: skin" in inputs
    assert all(o["local_files_only"] is True and o["trust_remote_code"] is False for o in observed["options"])
    assert observed["options"][1]["use_safetensors"] is True
    with pytest.raises(RAGError, match="rechunk_required"):
        embedder.embed_documents(["TOO_LONG"])


def test_missing_model_is_explicit_and_cannot_download(monkeypatch):
    calls = []

    def missing(name, **options):
        calls.append(options)
        raise OSError("not installed")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=missing), AutoModel=object()))
    with pytest.raises(ModelUnavailable, match="unavailable_local"):
        LocalE5Embedder().embed_query("synthetic")
    assert calls == [{"local_files_only": True, "trust_remote_code": False}]
