import math
import sys
from types import SimpleNamespace

import pytest

from app.rag.chunking import RecursiveChunker
from app.rag.models import Chunk, ModelUnavailable, RAGError, SearchHit
from app.rag.retrieval import (BM25Retriever, DenseRetriever, HybridRetriever,
                               LocalE5Embedder, RRFusion, tokenize)
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


def test_real_e5_local_smoke():
    # No network/download. Skip explicitly when dependencies or a complete local model are absent.
    embedder = LocalE5Embedder()
    try:
        docs = embedder.embed_documents(["SYNTHETIC DEMO skin color", "SYNTHETIC DEMO source citation"])
        query = embedder.embed_query("màu sắc da")
    except ModelUnavailable as exc:
        pytest.skip(f"Real multilingual-e5-base NOT VERIFIED: {exc}")
    assert docs.shape == (2, 768) and query.shape == (768,)
    assert float((query ** 2).sum()) == pytest.approx(1, abs=1e-5)


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
