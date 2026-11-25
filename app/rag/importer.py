"""Verified v0.1 ingestion using the pinned Crawler verifier and external store bindings."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.rag.models import Document, ImportResult, RAGError


@dataclass(frozen=True)
class StoreBindings:
    artifacts: Path
    provenance: Path


def safe_file(root: Path, relative: str) -> Path:
    """Reject traversal, Windows drives/ADS, symlinks and junctions below a bound root."""
    if not relative or "\\" in relative or ":" in relative:
        raise RAGError("unsafe_path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RAGError("unsafe_path")
    base = root.resolve(strict=True)
    candidate = base.joinpath(*parts)
    if candidate.resolve(strict=True) != candidate or not candidate.is_file():
        raise RAGError("unsafe_path")
    if not candidate.is_relative_to(base):
        raise RAGError("unsafe_path")
    return candidate


class DocumentValidator:
    def validate(self, record: object) -> None:
        try:
            from medical_data_crawler.contract import validate_record

            validate_record(record)
        except ImportError:
            raise RAGError("crawler_dependency_missing") from None
        except Exception:
            raise RAGError("record_validation_error") from None


class DocumentImporter:
    def __init__(self, bindings: StoreBindings, rights, *, max_file_bytes: int = 16 * 1024 * 1024):
        self.bindings = bindings
        self.rights = rights
        self.max_file_bytes = max_file_bytes
        self.validator = DocumentValidator()

    def import_manifest(self, manifest_path: Path) -> ImportResult:
        try:
            from medical_data_crawler.export import verify_batch
            from medical_data_crawler.provenance import ProvenanceStore
            from medical_data_crawler.storage import ArtifactStore

            # Require existing stores; importing must not create/guess roots.
            artifact_root = Path(self.bindings.artifacts).resolve(strict=True)
            provenance_root = Path(self.bindings.provenance).resolve(strict=True)
            if not artifact_root.is_dir() or not provenance_root.is_dir():
                raise RAGError("store_binding_invalid")
            manifest_path = Path(manifest_path).absolute()
            safe_file(manifest_path.parent, manifest_path.name)
            artifacts = ArtifactStore(artifact_root)
            provenance = ProvenanceStore(provenance_root, artifacts)
            verified = verify_batch(manifest_path, provenance, max_file_bytes=self.max_file_bytes)
            if not verified.valid:
                raise RAGError(verified.error_code)
            manifest = verified.manifest
            documents, rejected, observed = [], [], []
            seen = {}
            for entry in manifest["records"]:
                path = safe_file(manifest_path.parent, entry["record_path"])
                payload = path.read_bytes()
                # Bind the exact bytes used below, even if files changed since verify_batch.
                if (len(payload) != entry["record_byte_size"]
                        or hashlib.sha256(payload).hexdigest() != entry["record_sha256"]):
                    raise RAGError("record_integrity_error")
                record = json.loads(payload)
                self.validator.validate(record)
                checksum = record["original"]["sha256"]
                safe_file(artifact_root, checksum)
                event_bytes = safe_file(provenance_root, entry["event_id"] + ".json").read_bytes()
                # Recheck provenance against the actual record consumed (without private verifier APIs).
                event = json.loads(event_bytes)
                if (event.get("document_id") != record["document_id"]
                        or event.get("sha256") != checksum
                        or event.get("event_id") != entry["event_id"]
                        or event.get("event_version") != "0.1"
                        or event.get("status") != "accepted" or event.get("error_code") is not None
                        or event.get("artifact_reference") != record["original"]["reference"]
                        or event.get("byte_size") != record["original"]["byte_length"]
                        or any(event.get(k) != record["source"][k]
                               for k in ("source_id", "locator", "acquired_at"))):
                    raise RAGError("provenance_integrity_error")
                artifact = artifacts.verify(checksum)
                if artifact.byte_size != record["original"]["byte_length"]:
                    raise RAGError("original_integrity_error")
                identity = record["document_id"]
                observed.append(identity)
                if identity in seen:
                    if seen[identity] != entry["record_sha256"]:
                        raise RAGError("conflicting_duplicate_document")
                    continue
                seen[identity] = entry["record_sha256"]
                if record["extraction"]["status"] == "failed":
                    rejected.append((identity, "extraction_failed"))
                    continue
                evidence = self.rights.evidence(record, entry["record_sha256"])
                if evidence is None:
                    rejected.append((identity, "rights_or_partial_review_denied"))
                    continue
                documents.append(Document(record, entry["record_sha256"], manifest["batch_id"],
                                          entry["event_id"], hashlib.sha256(event_bytes).hexdigest(),
                                          evidence, self.rights.synthetic))
            observed.extend(o["document_id"] for o in manifest["omissions"] if o["document_id"])
            return ImportResult("VERIFIED" if not rejected else "BLOCKED", tuple(documents),
                                tuple(rejected), observed_ids=tuple(sorted(set(observed))))
        except ImportError:
            return ImportResult("INVALID", error_code="crawler_dependency_missing")
        except RAGError as exc:
            return ImportResult("INVALID", error_code=str(exc))
        except Exception:
            return ImportResult("INVALID", error_code="batch_verification_failed")
