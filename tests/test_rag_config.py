"""Operator configuration boundaries and integration with existing WP6A contracts."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.rag.chunking import ChunkConfig as LegacyChunkConfig
from app.rag.config import ChunkConfig, EmbeddingConfig, RAGConfig
from app.rag.models import RAGError
from app.rag.retrieval import EmbeddingConfig as LegacyEmbeddingConfig
from app.rag.runtime import create_service
from app.rag.service import RAGService
from rag_support import reviewed_batch


def test_defaults_legacy_imports_and_old_wp6a_json():
    assert LegacyChunkConfig is ChunkConfig
    assert LegacyEmbeddingConfig is EmbeddingConfig
    config = RAGConfig.from_dict({"chunking": {"max_tokens": 384, "overlap": 48},
                                  "embedding": {"allow_download": False},
                                  "rrf_k": 60, "candidates": 30})
    assert config == RAGConfig()
    example = RAGConfig.load(Path("config/rag.example.json"))
    assert example.paths is not None
    assert example.chunking == config.chunking
    assert example.bm25 == config.bm25 and example.retrieval == config.retrieval
    assert example.embedding == config.embedding
    # Preserve the direct service defaults and response contract.
    assert RAGService().search("skin").mode == "Hybrid"


@pytest.mark.parametrize("value", [
    [], None, "config", {"api_key": "forbidden"}, {"unknown": 1},
    {"manifest": "file.json"}, {"paths": {}},
    {"chunking": None}, {"chunking": []}, {"chunking": {"budget_typo": 500}},
    {"chunking": {"max_tokens": True}}, {"chunking": {"max_tokens": 384.0}},
    {"chunking": {"max_tokens": 3}}, {"chunking": {"overlap": -1}},
    {"chunking": {"overlap": 384}}, {"chunking": {"overlap": "48"}},
    {"chunking": {"strategy": "tokenizer"}}, {"chunking": {"token_metric": "tokens"}},
    {"chunking": {"version": "unsupported"}},
    {"bm25": {"k1": True}}, {"bm25": {"k1": 0}}, {"bm25": {"k1": "1.5"}},
    {"bm25": {"b": -0.1}}, {"bm25": {"b": 1.1}}, {"bm25": {"b": float("nan")}},
    {"embedding": {"model": 123}}, {"embedding": {"model": " "}},
    {"embedding": {"allow_download": "false"}}, {"embedding": {"device": []}},
    {"embedding": {"device": "mps"}}, {"embedding": {"query_prefix": None}},
    {"embedding": {"passage_prefix": 1}}, {"embedding": {"max_length": True}},
    {"embedding": {"max_length": 512.0}}, {"embedding": {"max_length": 513}},
    {"embedding": {"max_length": 7}}, {"embedding": {"batch_size": 0}},
    {"embedding": {"batch_size": 1.5}},
    {"retrieval": {"mode": []}}, {"retrieval": {"mode": "dense"}},
    {"retrieval": {"top_k": True}}, {"retrieval": {"top_k": 0}},
    {"retrieval": {"top_k": 101}}, {"retrieval": {"top_k": "5"}},
    {"rrf_k": 0}, {"rrf_k": True}, {"rrf_k": float("inf")}, {"rrf_k": "60"},
    {"rrf_k": 10 ** 400},
    {"candidates": 0}, {"candidates": True}, {"candidates": 30.5},
])
def test_invalid_config_is_rejected(value):
    with pytest.raises(RAGError):
        RAGConfig.from_dict(value)


@pytest.mark.parametrize("text", ['{', '[]', '{"rrf_k": NaN}', '{"rrf_k": Infinity}',
                                  '{"rrf_k": 60, "rrf_k": 10}',
                                  '{"embedding": {"device": "cpu", "device": "cuda"}}'])
def test_invalid_json_and_ambiguous_keys_fail_closed(tmp_path, text):
    path = tmp_path / "rag.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RAGError):
        RAGConfig.load(path)


@pytest.mark.parametrize("field", ["manifest", "artifacts", "provenance", "registry", "approvals"])
@pytest.mark.parametrize("value", [None, 42, [], "", " ", "a\0b"])
def test_invalid_path_types_and_empty_paths(field, value):
    settings = dict.fromkeys(["manifest", "artifacts", "provenance", "registry", "approvals"], "valid")
    settings[field] = value
    with pytest.raises(RAGError):
        RAGConfig.from_dict(settings)


def test_runtime_applies_settings_paths_rights_and_cli_overrides(tmp_path, monkeypatch, capsys):
    from app.rag.__main__ import main

    manifest, bindings, _, registry, approvals = reviewed_batch(tmp_path)
    config_dir = tmp_path / "settings"
    config_dir.mkdir()
    path = config_dir / "rag.json"
    settings = {"manifest": "../" + manifest.relative_to(tmp_path).as_posix(),
                "artifacts": str(bindings.artifacts), "provenance": str(bindings.provenance),
                "registry": "../registry.json", "approvals": "../approvals.json",
                "chunking": {"max_tokens": 200, "overlap": 20},
                "bm25": {"k1": 2.1, "b": 0.4}, "retrieval": {"mode": "BM25", "top_k": 1},
                "embedding": {"batch_size": 2, "max_length": 256,
                              "query_prefix": "q: ", "passage_prefix": "p: "},
                "rrf_k": 20, "candidates": 9}
    path.write_text(json.dumps(settings), encoding="utf-8")
    service = create_service(config_path=path)
    try:
        assert service.document_count == 3
        assert service.chunker.config == ChunkConfig(max_tokens=200, overlap=20)
        assert (service.bm25.k1, service.bm25.b) == (2.1, 0.4)
        assert service.hybrid.fusion.k == 20 and service.hybrid.candidates == 9
        assert service.dense.embedder.config == EmbeddingConfig(batch_size=2, max_length=256,
                                                                query_prefix="q: ", passage_prefix="p: ")
        assert service.dense.embedder._model is None  # Config/BM25 must not load/download E5.
        assert service.settings.paths.resolve("registry", path) == registry
        assert service.settings.paths.resolve("approvals", path) == approvals
        assert service.search("skin", "BM25", 2).status == "OK"
    finally:
        service.close()
    monkeypatch.setattr("sys.argv", ["rag", "--config", str(path), "--query", "SYNTHETIC DEMO"])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "BM25" and len(result["hits"]) == 1
    settings["retrieval"] = {"mode": "Dense", "top_k": 1}
    path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["rag", "--config", str(path), "--query", "SYNTHETIC DEMO",
                                    "--mode", "BM25", "--top-k", "2"])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "BM25" and len(result["hits"]) == 2


def test_retrieval_defaults_initialize_ui(tmp_path, monkeypatch):
    import gradio as gr
    from app.ui.rag_page import render_rag_page

    path = tmp_path / "rag.json"
    path.write_text(json.dumps({"retrieval": {"mode": "Hybrid", "top_k": 42}}), encoding="utf-8")
    monkeypatch.setenv("RAG_CONFIG", str(path))
    with gr.Blocks() as demo:
        render_rag_page()
    try:
        components = {c["props"].get("label"): c["props"] for c in demo.config["components"]}
        assert components["Retrieval mode"]["value"] == "Hybrid"
        assert components["Top-K"]["value"] == 42
    finally:
        demo.close()


def test_runtime_requires_all_production_paths(tmp_path):
    path = tmp_path / "rag.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RAGError, match="incomplete_rag_paths"):
        create_service(config_path=path)


def test_config_boundary_values():
    settings = RAGConfig.from_dict({"chunking": {"max_tokens": 4, "overlap": 3},
                                   "bm25": {"k1": 0.1, "b": 0},
                                   "embedding": {"max_length": 8, "batch_size": 1,
                                                 "query_prefix": "", "passage_prefix": ""},
                                   "retrieval": {"mode": "Hybrid", "top_k": 100},
                                   "rrf_k": 0.1, "candidates": 1})
    assert settings.chunking.overlap == 3 and settings.embedding.query_prefix == ""
    assert asdict(settings.bm25) == {"k1": 0.1, "b": 0}
