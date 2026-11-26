"""Live rights decisions from operator-configured, reviewed files, never batch flags."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from app.rag.models import Document


def utc_now() -> datetime:
    return datetime.now(UTC)


class RegistryRights:
    """Trust boundary: registry and approvals must be controlled by the reviewer.

    Crawler's registry authorizes reuse/AI independently. WP6A's sidecar additionally
    binds indexing and partial review to exact document/record/original hashes.
    Both files are re-read on every decision, including before returning search hits.
    """

    synthetic = False

    def __init__(self, registry_path: Path, approvals_path: Path, *, clock=utc_now):
        self.registry_path = Path(registry_path)
        self.approvals_path = Path(approvals_path)
        self.clock = clock

    def evidence(self, record: dict, record_hash: str) -> dict | None:
        try:
            from medical_data_crawler.source_policy import allows_asset_use, validate_registry

            registry_bytes = self.registry_path.read_bytes()
            approval_bytes = self.approvals_path.read_bytes()
            registry = json.loads(registry_bytes)
            approvals = json.loads(approval_bytes)
            validate_registry(registry)
            if approvals["version"] != "wp6a-indexing-1":
                return None
            source = record["source"]
            asset = {"text/html": "html_text", "application/pdf": "pdf_text"}.get(
                record["original"]["media_type"]
            )
            # Metadata/abstract APIs are never full-text evidence in WP6A.
            from urllib.parse import urlsplit

            host = urlsplit(source["locator"]).hostname or ""
            if not asset or host == "pubmed.ncbi.nlm.nih.gov" or "pubmed" in source["source_id"].lower():
                return None
            now = self.clock()
            if now.utcoffset() is None:
                return None
            if record["rights"]["reuse"] != "permitted":
                return None
            for action in ("read", "internal_storage", "reuse", "ai_use"):
                if not allows_asset_use(registry, source["source_id"], source["locator"],
                                        asset=asset, action=action, at=now):
                    return None
            matches = [item for item in approvals["documents"]
                       if item["document_id"] == record["document_id"]
                       and item["record_sha256"] == record_hash]
            if len(matches) != 1:
                return None
            grant = matches[0]
            if (grant["original_sha256"] != record["original"]["sha256"]
                    or grant["source_id"] != source["source_id"]
                    or grant["locator"] != source["locator"]
                    or grant["indexing"] != "allowed"
                    or grant["content_scope"] != "full_text"
                    or not isinstance(grant["reviewer_id"], str) or not grant["reviewer_id"].strip()
                    or not isinstance(grant["evidence"], str) or not grant["evidence"].strip()
                    or not datetime.fromisoformat(grant["reviewed_at"]) <= now
                    < datetime.fromisoformat(grant["review_again_at"])):
                return None
            if record["extraction"]["status"] == "partial" and grant.get("partial") != "allowed":
                return None
            reviewed_source = next(s for s in registry["sources"] if s["source_id"] == source["source_id"])
            if asset != reviewed_source["document_type"] + "_text":
                return None
            return {"reuse": "permitted", "ai_use": "allowed", "indexing": "allowed",
                    "asset": asset, "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
                    "approvals_sha256": hashlib.sha256(approval_bytes).hexdigest(),
                    "reviewer_id": grant["reviewer_id"], "evidence": grant["evidence"],
                    "review_again_at": grant["review_again_at"],
                    "source_review": reviewed_source["review"],
                    "asset_rights": reviewed_source["rights"][asset]}
        except Exception:
            # Missing dependency, malformed/expired/revoked decisions all close the gate.
            return None

    def allows(self, document: Document) -> bool:
        return not document.synthetic and self.evidence(document.record, document.record_hash) is not None

    def external_evidence(self, document: Document, action: str, model: str, provider: str | None):
        """Additional reviewed sidecar decision. Reuse/AI/indexing never imply outbound rights."""
        try:
            if not provider or not self.allows(document):
                return None
            approvals = json.loads(self.approvals_path.read_bytes())
            matches = [g for g in approvals["documents"]
                       if g["document_id"] == document.document_id
                       and g["record_sha256"] == document.record_hash]
            if len(matches) != 1:
                return None
            grant = matches[0]["external_processing"]
            now = self.clock()
            if (grant["status"] != "allowed" or grant["patient_identifiers"] != "absent_reviewed"
                    or grant["destination"] != "https://openrouter.ai/api/v1"
                    or not isinstance(grant["reviewer_id"], str) or not grant["reviewer_id"].strip()
                    or not isinstance(grant["evidence"], str) or not grant["evidence"].strip()
                    or not datetime.fromisoformat(grant["reviewed_at"]) <= now
                    < datetime.fromisoformat(grant["review_again_at"])):
                return None
            routes = [r for r in grant["routes"] if r == {
                "action": action, "model": model, "provider": provider}]
            return grant if len(routes) == 1 else None
        except Exception:
            return None
