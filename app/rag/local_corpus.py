"""Exact local kit ingestion. Local indexing consent is not a page-review claim.

The existing crawler/RegistryRights route remains independent and unchanged.
This adapter only accepts a hash-bound operator consent for eight local PDFs/MDs.
"""

import hashlib
import json
import re
from pathlib import Path

import yaml

from app.rag.importer import safe_file
from app.rag.models import Document, ImportResult, RAGError


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def read_kit(kit: Path, markdown: Path):
    metadata_bytes = safe_file(kit, "metadata/documents.json").read_bytes()
    entries = json.loads(metadata_bytes)
    if (not isinstance(entries, list) or len(entries) != 8
            or len({e["document_id"] for e in entries}) != 8):
        raise RAGError("local_kit_requires_exactly_eight_documents")
    expected = {Path(e["markdown_filename"]).name for e in entries}
    if {p.name for p in markdown.glob("*.md")} != expected:
        raise RAGError("local_markdown_set_mismatch")
    snapshots = []
    for entry in entries:
        name = entry["markdown_filename"]
        if name != "markdown/" + Path(name).name:
            raise RAGError("unsafe_path")
        md = safe_file(markdown, Path(name).name)
        pdf = safe_file(kit, "pdf_originals/" + entry["pdf_filename"])
        payload, original = md.read_bytes(), pdf.read_bytes()
        if not original.startswith(b"%PDF-"):
            raise RAGError("local_pdf_invalid")
        text = payload.decode("utf-8")
        front = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
        if not front:
            raise RAGError("local_markdown_frontmatter_missing")
        header = yaml.safe_load(front[1])
        source = entry["source_record"]
        if (header["document_id"] != entry["document_id"]
                or header["original_sha256"] != digest(original)
                or entry["download_record"]["sha256"] != digest(original)
                or header["pdf_pages"] != entry["pdf_pages"]
                or header["source_url"] != source["article_url"]
                or header["license"] != source["license"]
                or header["conversion_status"] != entry["conversion_status"]
                or header["human_review_status"] != entry["human_review_status"]):
            raise RAGError("local_source_integrity_mismatch")
        body = text[front.end():]
        markers = list(re.finditer(r"^<!-- source_page: (\d+) -->\s*$", body, re.M))
        if [int(m[1]) for m in markers] != list(range(1, entry["pdf_pages"] + 1)):
            raise RAGError("local_pdf_page_markers_invalid")
        pages = {p["pdf_page"]: p for p in entry["pages"]}
        blocks = []
        for i, marker in enumerate(markers):
            page = int(marker[1])
            section = body[marker.end():markers[i + 1].start() if i + 1 < len(markers) else len(body)]
            section = re.sub(r"<!--.*?-->", "", section, flags=re.S).strip()
            # Paragraphs, table rows, and headings retain Markdown; no medical rewriting.
            for part in re.split(r"\n\s*\n|(?=^#{1,6} )", section, flags=re.M):
                if not part.strip():
                    continue
                heading = re.match(r"^(#{1,6})\s+", part.strip())
                blocks.append({"text": part.strip(), "kind": "heading" if heading else "paragraph",
                               "level": len(heading[1]) if heading else None, "pdf_page": page,
                               "printed_page": pages[page].get("printed_page_number", "")})
        if not blocks or {b["pdf_page"] for b in blocks} != set(pages):
            raise RAGError("local_empty_or_missing_pages")
        binding = {"document_id": entry["document_id"], "markdown_sha256": digest(payload),
                   "original_sha256": digest(original), "metadata_sha256": digest(metadata_bytes),
                   "source_url": source["article_url"], "license": source["license"]}
        snapshots.append((binding, header, blocks, md, pdf))
    return snapshots


def consent_for(kit: Path, markdown: Path):
    """Called only by the operator prepare command after explicit user authorization."""
    snapshots = read_kit(kit, markdown)
    return {"version": "wp6d-local-consent-1", "indexing": "allowed", "scope": "local_only",
            "evidence": "User explicitly accepted these eight documents for local indexing in the WP6D request.",
            "page_review_claim": "none; original conversion and review statuses retained",
            "documents": [s[0] for s in snapshots]}


class LocalKitRights:
    synthetic = False

    def __init__(self, approval):
        self.approval = Path(approval)

    def evidence(self, binding):
        try:
            grant = json.loads(self.approval.read_bytes())
            if (grant["version"] != "wp6d-local-consent-1" or grant["indexing"] != "allowed"
                    or grant["scope"] != "local_only" or not grant["evidence"].strip()
                    or len(grant["documents"]) != 8 or grant["documents"].count(binding) != 1):
                return None
            return {"indexing": "allowed", "scope": "local_only", "evidence": grant["evidence"],
                    "approval_sha256": digest(self.approval.read_bytes()), "page_review_claim": "none"}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def allows(self, document):
        return not document.synthetic and self.evidence(document.record["local_binding"]) is not None


class LocalKitImporter:
    def __init__(self, kit, markdown, approval):
        self.kit, self.markdown = Path(kit), Path(markdown)
        self.rights = LocalKitRights(approval)

    def import_manifest(self, manifest_path):
        try:
            if Path(manifest_path).resolve() != self.rights.approval.resolve():
                raise RAGError("local_approval_path_mismatch")
            documents = []
            for binding, header, blocks, md, pdf in read_kit(self.kit, self.markdown):
                evidence = self.rights.evidence(binding)
                if evidence is None:
                    raise RAGError("local_indexing_consent_missing_or_changed")
                metadata = {**header, "markdown_path": str(md), "pdf_path": str(pdf)}
                record = {"document_id": binding["document_id"], "content": blocks,
                          "metadata": metadata, "source": {"source_id": binding["document_id"],
                          "locator": binding["source_url"]}, "original": {"sha256": binding["original_sha256"],
                          "media_type": "application/pdf"}, "rights": {"license": binding["license"],
                          "scope": "local_only"}, "extraction": {"status": header["conversion_status"]},
                          "local_binding": binding}
                record_hash = digest(canonical(record))
                documents.append(Document(record, record_hash, "wp6d-local-kit", "local-consent",
                                          binding["metadata_sha256"], evidence))
            # VERIFIED describes hash/page-marker/consent checks, never content or page review.
            return ImportResult("VERIFIED", tuple(documents), observed_ids=tuple(d.document_id for d in documents))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
            return ImportResult("INVALID", error_code="local_kit_integrity_or_consent_failed")
