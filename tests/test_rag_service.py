from copy import deepcopy

import pytest

from app.rag.models import ModelUnavailable
from app.rag.runtime import create_service
from app.rag.service import RAGService
from rag_support import FakeEmbedder, approve_current, read, reviewed_batch, synthetic_batch, update_record, write


def test_three_modes_and_empty_index(tmp_path):
    service = RAGService(embedder=FakeEmbedder())
    assert service.search("skin", "Dense").status == "EMPTY_INDEX"
    manifest, _, importer = synthetic_batch(tmp_path)
    assert service.ingest(manifest, importer).status == "VERIFIED"
    for mode in ("BM25", "Dense", "Hybrid"):
        response = service.search("skin", mode)
        assert response.status == "OK" and response.hits and response.synthetic
        assert response.elapsed_ms >= 0
    assert service.search("", "Hybrid").status == "EMPTY_QUERY"
    assert service.search("unmatched_token_xyz", "BM25").status == "NO_MATCH"


def test_exact_duplicates_are_deduplicated(tmp_path):
    manifest, _, importer = synthetic_batch(tmp_path)
    data = read(manifest)
    duplicate = deepcopy(data["records"][0])
    duplicate.update(input_index=3, record_path="records/3.json")
    (manifest.parent / "records/3.json").write_bytes((manifest.parent / "records/0.json").read_bytes())
    data["records"].append(duplicate)
    data["summary"].update(total=4, exported=4, extraction={"success": 4, "partial": 0, "failed": 0})
    write(manifest, data)
    result = importer.import_manifest(manifest)
    assert result.status == "VERIFIED" and len(result.documents) == 3
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(manifest, importer)
    assert service.document_count == 3
    service.ingest(manifest, importer)
    assert service.document_count == 3


def test_conflicting_duplicates_fail_entire_batch(tmp_path):
    manifest, _, importer = synthetic_batch(tmp_path)
    data = read(manifest)
    duplicate = deepcopy(data["records"][0])
    duplicate.update(input_index=3, record_path="records/3.json")
    (manifest.parent / "records/3.json").write_bytes((manifest.parent / "records/0.json").read_bytes())
    data["records"].append(duplicate)
    data["summary"].update(total=4, exported=4, extraction={"success": 4, "partial": 0, "failed": 0})
    write(manifest, data)
    update_record(manifest, 3, lambda r: r["content"][0].update(text="SYNTHETIC DEMO conflicting version"))
    result = importer.import_manifest(manifest)
    assert result.status == "INVALID" and result.error_code == "conflicting_duplicate_document"
    assert not result.documents
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(manifest, importer)
    assert service.document_count == 0


@pytest.mark.parametrize("mode", ["BM25", "Dense", "Hybrid"])
def test_update_requires_new_hash_approval_and_removes_old_chunks(tmp_path, mode):
    manifest, _, importer, _, approvals = reviewed_batch(tmp_path)
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(manifest, importer)
    original = service.search("skin", mode).hits
    document_id = read(manifest)["records"][0]["document_id"]
    old_ids = {h.chunk.chunk_id for h in original if h.chunk.document_id == document_id}
    assert old_ids
    update_record(manifest, 0, lambda r: r["content"][1].update(text="SYNTHETIC DEMO newrevisiontoken skin"))
    without_review = service.search("newrevisiontoken", mode)
    assert all(h.chunk.document_id != document_id for h in without_review.hits)
    approve_current(manifest, approvals)
    updated = service.search("newrevisiontoken", mode)
    assert any("newrevisiontoken" in h.chunk.text for h in updated.hits)
    assert all(h.chunk.chunk_id not in old_ids for h in updated.hits)


@pytest.mark.parametrize("revoke", ["registry", "approval", "original", "provenance", "manifest", "remove_record"])
def test_live_revocation_evicts_all_indexes(tmp_path, revoke):
    manifest, bindings, importer, registry, approvals = reviewed_batch(tmp_path)
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(manifest, importer)
    assert service.search("skin", "Hybrid").hits
    entries = read(manifest)["records"]
    target = entries[0]["document_id"]
    if revoke == "registry":
        data = read(registry)
        data["sources"][0]["rights"]["html_text"]["ai_use"]["status"] = "denied"
        write(registry, data)
    elif revoke == "approval":
        data = read(approvals)
        data["documents"][0]["indexing"] = "revoked"
        write(approvals, data)
    elif revoke == "original":
        (bindings.artifacts / entries[0]["original_sha256"]).unlink()
    elif revoke == "provenance":
        (bindings.provenance / (entries[0]["event_id"] + ".json")).unlink()
    elif revoke == "manifest":
        manifest.write_text("invalid")
    else:
        data = read(manifest)
        data["records"] = data["records"][1:]
        data["omissions"] = [{"input_index": 0, "document_id": target, "event_id": entries[0]["event_id"],
                              "status": "skipped", "code": "missing_access_basis"}]
        data["summary"].update(exported=2, skipped=1, extraction={"success": 2, "partial": 0, "failed": 0})
        (manifest.parent / "records/0.json").unlink()
        write(manifest, data)
    response = service.search("skin", "Hybrid")
    assert all(h.chunk.document_id != target for h in response.hits)
    assert all(c.document_id != target for c in service.bm25.chunks)
    assert all(c.document_id != target for c in service.dense.chunks)


def test_explicit_removal_does_not_reappear_and_sessions_are_independent(tmp_path):
    manifest, _, importer = synthetic_batch(tmp_path)
    a, b = RAGService(embedder=FakeEmbedder()), RAGService(embedder=FakeEmbedder())
    a.ingest(manifest, importer)
    assert b.search("skin").status == "EMPTY_INDEX"
    b.ingest(manifest, importer)
    target = a.search("skin", "BM25").hits[0].chunk.document_id
    a.remove_document(target)
    assert not a.search("skin", "BM25").hits
    assert b.search("skin", "BM25").hits
    a.ingest(manifest, importer)
    assert a.search("skin", "BM25").hits


def test_new_receipt_never_falls_back_to_old_authorized_version(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    old, _, old_importer, _, _ = reviewed_batch(old_root)
    new, _, new_importer, _, approvals = reviewed_batch(new_root)
    update_record(new, 0, lambda r: r["content"][1].update(text="SYNTHETIC DEMO newreceipt skin"))
    approve_current(new, approvals)
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(old, old_importer)
    target = read(old)["records"][0]["document_id"]
    service.ingest(new, new_importer)
    assert any("newreceipt" in h.chunk.text for h in service.search("skin", "Hybrid").hits)
    approve_current(new, approvals, indexing="revoked")
    assert all(h.chunk.document_id != target for h in service.search("skin", "Hybrid").hits)


def test_live_evidence_refresh_does_not_leave_stale_dense_citations(tmp_path):
    manifest, _, importer, _, approvals = reviewed_batch(tmp_path)
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(manifest, importer)
    first = service.search("skin", "Dense")
    target = read(manifest)["records"][0]["document_id"]
    before = next(h for h in first.hits if h.chunk.document_id == target)
    approve_current(manifest, approvals, evidence="SYNTHETIC DEMO updated review evidence")
    after = next(h for h in service.search("skin", "Dense").hits if h.chunk.document_id == target)
    assert before.chunk.chunk_id == after.chunk.chunk_id
    assert after.chunk.citation["rights"]["evidence"] == "SYNTHETIC DEMO updated review evidence"


def test_returned_metadata_cannot_mutate_internal_corpus(tmp_path):
    manifest, _, importer = synthetic_batch(tmp_path)
    service = RAGService(embedder=FakeEmbedder())
    service.ingest(manifest, importer)
    hit = service.search("skin", "BM25").hits[0]
    hit.chunk.citation["source"]["locator"] = "forged"
    hit.chunk.citation["rights"].clear()
    again = service.search("skin", "BM25").hits[0]
    assert again.chunk.citation["source"]["locator"].startswith("synthetic://")
    assert again.chunk.citation["rights"]["ai_use"] == "allowed"


def test_unavailable_dense_does_not_silently_fall_back(tmp_path):
    class Missing:
        def embed_documents(self, texts):
            raise ModelUnavailable("embedding_model_unavailable_local")

    manifest, _, importer = synthetic_batch(tmp_path)
    service = RAGService(embedder=Missing())
    service.ingest(manifest, importer)
    for mode in ("Dense", "Hybrid"):
        response = service.search("skin", mode)
        assert response.status == "embedding_model_unavailable_local" and response.hits == ()
    assert service.search("skin", "BM25").hits


def test_runtime_demo_and_real_config_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        create_service(demo=True, config_path=tmp_path / "config.json")
    service = create_service(demo=True)
    try:
        assert service.search("skin", "BM25").synthetic
    finally:
        service.close()
    assert service.document_count == 0
