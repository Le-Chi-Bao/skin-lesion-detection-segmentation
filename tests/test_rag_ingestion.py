import hashlib
from copy import deepcopy

import pytest

from app.rag.importer import DocumentImporter, DocumentValidator, safe_file
from app.rag.models import RAGError
from app.rag.synthetic import SyntheticRights, json_bytes
from rag_support import approve_current, read, reviewed_batch, synthetic_batch, update_record, write


def test_v01_parsing_external_stores_and_verification(tmp_path):
    manifest, bindings, importer = synthetic_batch(tmp_path)
    assert not bindings.artifacts.is_relative_to(manifest.parent)
    result = importer.import_manifest(manifest)
    assert result.status == "VERIFIED" and len(result.documents) == 3
    for doc in result.documents:
        DocumentValidator().validate(doc.record)
        assert doc.synthetic and len(doc.provenance_hash) == 64
        assert doc.record_hash in {r["record_sha256"] for r in read(manifest)["records"]}


@pytest.mark.parametrize("field", ["schema_version", "metadata", "rights", "extraction", "unexpected"])
def test_record_schema_is_strict(tmp_path, field):
    manifest, _, importer = synthetic_batch(tmp_path)
    record = read(manifest.parent / "records/0.json")
    if field == "unexpected":
        record[field] = True
    else:
        record[field] = "invalid"
    with pytest.raises(RAGError, match="record_validation_error"):
        importer.validator.validate(record)


@pytest.mark.parametrize("target", ["record_hash", "record_size", "original", "original_size", "provenance",
                                     "manifest_version", "summary", "path", "unlisted", "missing_event"])
def test_integrity_corruption_never_marks_verified(tmp_path, target):
    path, bindings, importer = synthetic_batch(tmp_path)
    manifest = read(path)
    entry = manifest["records"][0]
    if target == "record_hash":
        entry["record_sha256"] = "a" * 64
    elif target == "record_size":
        entry["record_byte_size"] += 1
    elif target == "original":
        (bindings.artifacts / entry["original_sha256"]).write_bytes(b"corrupted SYNTHETIC DEMO")
    elif target == "original_size":
        update_record(path, 0, lambda r: r["original"].update(byte_length=1))
        manifest = read(path)
    elif target == "provenance":
        event_path = bindings.provenance / (entry["event_id"] + ".json")
        event = read(event_path)
        event["locator"] = "synthetic://forged"
        write(event_path, event)
    elif target == "manifest_version":
        manifest["manifest_version"] = "0.2"
    elif target == "summary":
        manifest["summary"]["total"] += 1
    elif target == "path":
        entry["record_path"] = "../records/0.json"
    elif target == "unlisted":
        (path.parent / "records/extra.json").write_text("{}")
    elif target == "missing_event":
        (bindings.provenance / (entry["event_id"] + ".json")).unlink()
    write(path, manifest)
    result = importer.import_manifest(path)
    assert result.status == "INVALID" and result.documents == () and result.error_code


@pytest.mark.parametrize("relative", ["../secret", "/secret", "C:/secret", "records\\0.json",
                                       "records/../secret", "records/0.json:stream", "./secret"])
def test_path_traversal_rejected(tmp_path, relative):
    with pytest.raises(RAGError, match="unsafe_path"):
        safe_file(tmp_path, relative)


def test_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("SYNTHETIC DEMO")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(RAGError, match="unsafe_path"):
        safe_file(tmp_path, "link")


def test_synthetic_permission_is_bound_to_exact_authored_content(tmp_path):
    manifest, bindings, importer = synthetic_batch(tmp_path)
    update_record(manifest, 0, lambda r: r["content"][0].update(text="SYNTHETIC DEMO label on arbitrary content"))
    result = importer.import_manifest(manifest)
    assert result.status == "BLOCKED" and len(result.documents) == 2
    assert result.rejected[0][1] == "rights_or_partial_review_denied"
    # A production reviewed record cannot enter through SyntheticRights either.
    real, stores, _, _, _ = reviewed_batch(tmp_path)
    assert not DocumentImporter(stores, SyntheticRights()).import_manifest(real).documents


def test_production_registry_does_not_authorize_synthetic_scheme(tmp_path):
    path, bindings, _ = synthetic_batch(tmp_path)
    _, _, importer, _, _ = reviewed_batch(tmp_path)
    result = DocumentImporter(bindings, importer.rights).import_manifest(path)
    assert result.status == "BLOCKED" and not result.documents


@pytest.mark.parametrize("change", ["ai_unknown", "reuse_denied", "indexing_denied", "expired", "missing_evidence",
                                     "wrong_hash", "missing_registry", "metadata_only", "duplicate_grant"])
def test_rights_fail_closed_despite_record_reuse_permitted(tmp_path, change):
    manifest, _, importer, registry_path, approval_path = reviewed_batch(tmp_path)
    assert importer.import_manifest(manifest).status == "VERIFIED"
    registry, approvals = read(registry_path), read(approval_path)
    grant = approvals["documents"][0]
    if change == "ai_unknown":
        registry["sources"][0]["rights"]["html_text"]["ai_use"] = {"status": "unknown", "evidence_url": None}
    elif change == "reuse_denied":
        registry["sources"][0]["rights"]["html_text"]["reuse"]["status"] = "denied"
    elif change == "indexing_denied":
        grant["indexing"] = "denied"
    elif change == "expired":
        grant["review_again_at"] = "2026-02-01T00:00:00Z"
    elif change == "missing_evidence":
        grant["evidence"] = ""
    elif change == "wrong_hash":
        grant["record_sha256"] = "f" * 64
    elif change == "metadata_only":
        grant["content_scope"] = "metadata_only"
    elif change == "duplicate_grant":
        approvals["documents"].append(deepcopy(grant))
    write(registry_path, registry)
    write(approval_path, approvals)
    if change == "missing_registry":
        registry_path.unlink()
    result = importer.import_manifest(manifest)
    assert result.status == "BLOCKED" and all(d.document_id != grant["document_id"] for d in result.documents)


@pytest.mark.parametrize("status,allow_partial,accepted", [("success", False, True), ("partial", False, False),
                                                           ("partial", True, True), ("failed", True, False)])
def test_extraction_review(tmp_path, status, allow_partial, accepted):
    manifest, _, importer, _, approvals = reviewed_batch(tmp_path)

    def change(record):
        record["extraction"].update(status=status, warnings=["SYNTHETIC DEMO incomplete"] if status == "partial" else [],
                                    errors=["SYNTHETIC DEMO failed"] if status == "failed" else [])
        if status == "failed":
            record["content"] = []

    record, _ = update_record(manifest, 0, change)
    approve_current(manifest, approvals, partial="allowed" if allow_partial else "denied")
    result = importer.import_manifest(manifest)
    assert (record["document_id"] in {d.document_id for d in result.documents}) == accepted


def test_pubmed_never_becomes_fulltext(tmp_path):
    manifest, _, importer, registry, approvals = reviewed_batch(tmp_path)
    record, _ = update_record(manifest, 0, lambda r: r["source"].update(source_id="pubmed-demo"))
    data = read(registry)
    data["sources"][0]["source_id"] = "pubmed-demo"
    write(registry, data)
    approve_current(manifest, approvals, source_id="pubmed-demo")
    from medical_data_crawler.source_policy import allows_asset_use
    from rag_support import NOW

    # Registry can authorize an API source; WP6A still must not treat it as full-text.
    assert allows_asset_use(data, "pubmed-demo", record["source"]["locator"], asset="html_text", action="ai_use", at=NOW)
    assert importer.rights.evidence(record, hashlib.sha256(json_bytes(record)).hexdigest()) is None


def test_missing_verifier_cannot_mark_batch_verified(tmp_path, monkeypatch):
    import builtins

    manifest, _, importer = synthetic_batch(tmp_path)
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("medical_data_crawler"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    result = importer.import_manifest(manifest)
    assert result.status == "INVALID" and not result.documents
    assert result.error_code == "crawler_dependency_missing"


def test_verification_limits_and_missing_store_are_closed(tmp_path):
    manifest, bindings, importer = synthetic_batch(tmp_path)
    bounded = DocumentImporter(bindings, importer.rights, max_file_bytes=32)
    assert bounded.import_manifest(manifest).status == "INVALID"
    from app.rag.importer import StoreBindings

    missing = DocumentImporter(StoreBindings(tmp_path / "does-not-exist", bindings.provenance), importer.rights)
    assert missing.import_manifest(manifest).status == "INVALID"
    assert not (tmp_path / "does-not-exist").exists()
