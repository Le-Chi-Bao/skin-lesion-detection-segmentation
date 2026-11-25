"""Fixed, authored SYNTHETIC DEMO corpus. No arbitrary fixture authorization switch."""

import hashlib
import json
from pathlib import Path

from app.rag.importer import StoreBindings
from app.rag.models import Document

LABEL = "SYNTHETIC DEMO"
# These are search tokens/examples, not clinical evidence or treatment instructions.
CORPUS = (
    ("skin-observation", "Quan sát tổn thương da / Skin observation", [
        {"kind": "heading", "level": 1, "text": "SYNTHETIC DEMO — Quan sát ảnh"},
        {"kind": "paragraph", "level": None, "text":
         "SYNTHETIC DEMO: Bộ từ khóa quan sát tổn thương da gồm màu sắc, đường viền và kích thước. "
         "Đây là câu tự tạo để kiểm thử truy xuất, không phải tài liệu y khoa."},
        {"kind": "list_item", "level": 1, "text": "SYNTHETIC DEMO: skin lesion color border size observation."},
    ]),
    ("retrieval-evidence", "Nguồn và bằng chứng / Source evidence", [
        {"kind": "heading", "level": 1, "text": "SYNTHETIC DEMO — Bằng chứng truy xuất"},
        {"kind": "paragraph", "level": None, "text":
         "SYNTHETIC DEMO: Truy xuất văn bản giữ nguồn tài liệu và mã đoạn. "
         "Source provenance and citation metadata identify the evidence used by retrieval."},
        {"kind": "table", "level": None, "text": "SYNTHETIC DEMO\nField | Meaning\nchunk_id | mã đoạn\nsource | nguồn"},
    ]),
    ("image-segmentation", "Phân đoạn ảnh / Image segmentation", [
        {"kind": "heading", "level": 1, "text": "SYNTHETIC DEMO — Phân đoạn"},
        {"kind": "paragraph", "level": None, "text":
         "SYNTHETIC DEMO: Mặt nạ phân đoạn ảnh là ví dụ từ vựng cho tìm kiếm. "
         "Image segmentation mask and bounding box are synthetic search examples, not clinical findings."},
    ]),
)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _data():
    records, originals, events = [], {}, {}
    for i, (key, title, content) in enumerate(CORPUS):
        original = (LABEL + "\n" + "\n".join(b["text"] for b in content)).encode("utf-8")
        checksum = hashlib.sha256(original).hexdigest()
        doc_id = "doc:" + hashlib.sha256(("wp6a-synthetic:" + key).encode()).hexdigest()
        event_id = hashlib.sha256(("wp6a-event:" + key).encode()).hexdigest()[:32]
        source = {"source_id": "wp6a-synthetic", "locator": "synthetic://wp6a/" + key,
                  "acquired_at": "2026-01-01T00:00:00Z", "context": "provenance_event:" + event_id}
        record = {"schema_version": "0.1", "document_id": doc_id, "source": source,
                  "original": {"reference": checksum, "media_type": "text/html",
                               "byte_length": len(original), "sha256": checksum},
                  "content": content, "metadata": {"title": LABEL + " — " + title, "language": "vi-en",
                                                  "authors": None, "published_on": None},
                  "scope": "unknown", "scope_evidence": None, "audience": "unknown",
                  "audience_evidence": None,
                  "extraction": {"status": "success", "method": "authored-synthetic", "version": "1",
                                 "configuration": {"profile": "wp6a-fixed-demo-v1"}, "warnings": [], "errors": []},
                  "rights": {"access_basis": "Project-authored SYNTHETIC DEMO", "reuse": "permitted",
                             "evidence": "wp6a-fixed-demo-v1"}}
        records.append(record)
        originals[checksum] = original
        events[event_id] = {"event_version": "0.1", "event_id": event_id, "document_id": doc_id,
                            "source_id": source["source_id"], "locator": source["locator"],
                            "acquired_at": source["acquired_at"], "artifact_reference": checksum,
                            "sha256": checksum, "byte_size": len(original), "status": "accepted", "error_code": None}
    return records, originals, events


class SyntheticRights:
    """Only exact code-owned records qualify. Labels or source names alone never qualify."""

    synthetic = True

    def __init__(self):
        self._allowed = {hashlib.sha256(json_bytes(r)).hexdigest(): r for r in _data()[0]}

    def evidence(self, record: dict, record_hash: str) -> dict | None:
        expected = self._allowed.get(record_hash)
        if expected is None or expected != record:
            return None
        return {"reuse": "permitted", "ai_use": "allowed", "indexing": "allowed",
                "evidence": "Project-authored fixed corpus: wp6a-fixed-demo-v1", "label": LABEL}

    def allows(self, document: Document) -> bool:
        return document.synthetic and self.evidence(document.record, document.record_hash) is not None


def write_demo(root: Path) -> tuple[Path, StoreBindings]:
    """Generate a fresh isolated layout; stores deliberately live outside the batch."""
    records, originals, events = _data()
    batch_id = hashlib.sha256(b"wp6a-fixed-demo-v1").hexdigest()[:32]
    root = Path(root)
    batch = root / "batches" / batch_id
    bindings = StoreBindings(root / "artifacts", root / "provenance")
    for directory in (batch / "records", bindings.artifacts, bindings.provenance):
        directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for checksum, payload in originals.items():
        (bindings.artifacts / checksum).write_bytes(payload)
    for event_id, event in events.items():
        (bindings.provenance / (event_id + ".json")).write_bytes(json_bytes(event))
    for i, record in enumerate(records):
        payload = json_bytes(record)
        (batch / "records" / f"{i}.json").write_bytes(payload)
        entries.append({"input_index": i, "document_id": record["document_id"],
                        "event_id": record["source"]["context"].split(":")[1], "record_path": f"records/{i}.json",
                        "record_sha256": hashlib.sha256(payload).hexdigest(), "record_byte_size": len(payload),
                        "original_reference": record["original"]["sha256"],
                        "original_sha256": record["original"]["sha256"], "extraction_status": "success", "reuse": "permitted"})
    manifest = {"manifest_version": "0.1", "batch_id": batch_id, "created_at": "2026-01-01T00:00:00Z",
                "structured_raw_version": "0.1", "records": entries, "omissions": [],
                "summary": {"total": len(entries), "exported": len(entries), "failed": 0, "skipped": 0,
                            "extraction": {"success": len(entries), "partial": 0, "failed": 0}}}
    path = batch / "manifest.json"
    path.write_bytes(json_bytes(manifest))
    return path, bindings
