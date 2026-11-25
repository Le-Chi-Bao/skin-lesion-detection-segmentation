"""Authored SYNTHETIC DEMO fixtures only; no downloaded medical documents."""

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from app.rag.importer import DocumentImporter
from app.rag.rights import RegistryRights
from app.rag.synthetic import SyntheticRights, json_bytes, write_demo

NOW = datetime(2026, 11, 25, tzinfo=UTC)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_bytes(json_bytes(value))


def update_record(manifest_path, index, transform):
    manifest = read(manifest_path)
    entry = manifest["records"][index]
    path = manifest_path.parent / entry["record_path"]
    record = read(path)
    transform(record)
    payload = json_bytes(record)
    path.write_bytes(payload)
    entry.update(record_sha256=hashlib.sha256(payload).hexdigest(), record_byte_size=len(payload),
                 extraction_status=record["extraction"]["status"], reuse=record["rights"]["reuse"])
    manifest["summary"]["extraction"] = {s: sum(e["extraction_status"] == s for e in manifest["records"])
                                          for s in ("success", "partial", "failed")}
    write(manifest_path, manifest)
    return record, entry


def synthetic_batch(tmp_path):
    manifest, bindings = write_demo(tmp_path / "demo")
    return manifest, bindings, DocumentImporter(bindings, SyntheticRights())


def reviewed_batch(tmp_path):
    manifest, bindings = write_demo(tmp_path / "reviewed")
    sources, grants = [], []
    for i in range(3):
        source_id = f"wp6a-demo-{chr(97 + i)}"
        locator = f"https://example.org/wp6a-demo-{chr(97 + i)}"

        def change(record):
            record["source"].update(source_id=source_id, locator=locator)

        record, entry = update_record(manifest, i, change)
        event_path = bindings.provenance / (entry["event_id"] + ".json")
        event = read(event_path)
        event.update(source_id=source_id, locator=locator)
        write(event_path, event)
        allowed = {"status": "allowed", "evidence_url": "https://example.org/synthetic-policy"}
        rights = {a: deepcopy(allowed) for a in ("read", "internal_storage", "reuse", "ai_use")}
        sources.append({"source_id": source_id, "publisher": "Synthetic Demo", "domain": "example.org",
                        "candidate_url": locator, "document_type": "html", "audience_candidate": "unknown",
                        "scope_candidates": [], "priority": "standard", "status": "approved_for_manual_pilot",
                        "rights": {"html_text": rights, "original_bytes": deepcopy(rights)},
                        "copyright_policy_urls": ["https://example.org/synthetic-policy"],
                        "access_terms_urls": ["https://example.org/synthetic-policy"],
                        "terms": {"manual": "allowed", "automated": "unknown",
                                  "evidence_url": "https://example.org/synthetic-policy"},
                        "robots": {"status": "not_checked", "user_agent": None, "evidence_url": None},
                        "automated_acquisition": "unknown",
                        "embedded_assets": {"status": "cleared_for_internal_storage",
                                            "evidence_url": "https://example.org/synthetic-policy"},
                        "review": {"reviewer_id": "reviewer-synthetic", "reviewed_at": "2026-01-01T00:00:00Z",
                                   "review_again_at": "2027-01-01T00:00:00Z"}, "pending_decisions": []})
        grants.append({"document_id": record["document_id"], "record_sha256": entry["record_sha256"],
                       "original_sha256": record["original"]["sha256"], "source_id": source_id,
                       "locator": locator, "indexing": "allowed", "content_scope": "full_text", "partial": "denied",
                       "reviewer_id": "reviewer-synthetic", "evidence": "Authored SYNTHETIC DEMO test only",
                       "reviewed_at": "2026-01-01T00:00:00Z", "review_again_at": "2027-01-01T00:00:00Z"})
    registry, approvals = tmp_path / "registry.json", tmp_path / "approvals.json"
    write(registry, {"schema_version": "0.1", "sources": sources})
    write(approvals, {"version": "wp6a-indexing-1", "documents": grants})
    policy = RegistryRights(registry, approvals, clock=lambda: NOW)
    return manifest, bindings, DocumentImporter(bindings, policy), registry, approvals


def approve_current(manifest, approvals, index=0, **changes):
    current = read(manifest)["records"][index]
    data = read(approvals)
    data["documents"][index].update(record_sha256=current["record_sha256"], **changes)
    write(approvals, data)


class FakeEmbedder:
    """Unit test geometry only; never evidence that a real model has run."""

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, query):
        return self._vector(query)

    @staticmethod
    def _vector(text):
        text = text.casefold()
        return [1 + text.count("skin") + text.count("da"),
                1 + text.count("source") + text.count("nguồn"),
                1 + text.count("mask") + text.count("phân đoạn")]
