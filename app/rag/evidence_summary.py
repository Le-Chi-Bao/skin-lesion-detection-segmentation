"""Deterministic extractive summaries that never claim to be LLM answers."""

import re

from app.rag.retrieval import tokenize


STOP_WORDS = {
    "va", "la", "nhung", "cac", "mot", "co", "trong", "cua", "den", "duoc", "nao",
    "gi", "the", "doi", "voi", "ve", "hay", "khong", "tai", "tu", "o", "khi",
    "bệnh", "viêm", "da", "cơ", "địa",
}


def _sentences(text):
    parts = (part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text))
    sentences, in_references = [], False
    for part in parts:
        if not part:
            continue
        heading = re.sub(r"^[#\s*_`]+|[#\s*_`]+$", "", part).casefold()
        if re.search(r"\b(tài liệu tham khảo|references|bibliography)\b", heading):
            in_references = True
        if (in_references or re.match(r"^(?:#{1,6}\s|<table\b|\*\*\s*\d+[.)])", part, re.I)
                or re.match(r"^\*\*[_ ]*(?:bảng|table)\b", part, re.I)
                or re.match(r"^mối liên quan giữa .*\(n\s*=\s*\d+\)", part, re.I)):
            continue
        sentences.append(part)
    return sentences


def select_evidence(query, hits, limit=5):
    """Return exact source sentences with lexical support from the question."""
    query_tokens = [term for term in tokenize(query) if len(term) > 1 and term not in STOP_WORDS]
    terms = set(query_tokens)
    phrases = tuple(zip(query_tokens, query_tokens[1:]))
    selected, seen = [], set()
    for hit in hits:
        for sentence in _sentences(hit.chunk.text):
            sentence_tokens = tokenize(sentence)
            overlap = terms & set(sentence_tokens)
            identity = (hit.chunk.chunk_id, sentence)
            if not overlap or identity in seen:
                continue
            seen.add(identity)
            sentence_phrases = set(zip(sentence_tokens, sentence_tokens[1:]))
            phrase_matches = len(set(phrases) & sentence_phrases)
            selected.append((len(overlap) + phrase_matches * 2, hit, sentence))
    selected.sort(key=lambda item: (-item[0], -item[1].score, item[1].chunk.chunk_id))
    return tuple((hit, sentence) for _, hit, sentence in selected[:limit])


def summarize_evidence(query, hits, limit=5):
    excerpts = select_evidence(query, hits, limit)
    if not excerpts:
        return "", ()
    lines, sources = [], []
    for index, (hit, sentence) in enumerate(excerpts, 1):
        evidence_id = f"E{index}"
        lines.append(f"- “{sentence}” [{evidence_id}]")
        sources.append({
            "evidence_id": evidence_id,
            "chunk_id": hit.chunk.chunk_id,
            "citation": hit.chunk.citation,
            "excerpt": sentence,
        })
    answer = (
        "Các đoạn được truy xuất có nội dung liên quan sau đây. Đây là trích đoạn được "
        "chọn theo từ khóa, không phải câu trả lời do LLM tạo hay kết luận y khoa.\n\n"
        + "\n".join(lines)
    )
    return answer, tuple(sources)
