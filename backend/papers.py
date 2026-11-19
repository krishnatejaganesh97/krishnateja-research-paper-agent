# Paper handling — load, extract, chunk, retrieve.
#
# What Teja learns here (Product 1's RAG pipeline, applied to papers):
#   1. PDF extraction — pypdf turns a PDF into plain text. PDFs are messy
#      (columns, headers, page numbers), so extraction is never perfect.
#   2. Section parsing — research papers have structure (Problem, Method,
#      Findings, Limitations). Parsing sections makes summaries reliable:
#      the agent reads the Findings section for findings, not the whole
#      paper at once.
#   3. Chunking — same as Product 1 (LangChain splitter, ~500 chars).
#   4. Retrieval — keyword scoring over chunks in mock mode (deterministic,
#      works offline). With a real embedding model you would rank by vector
#      similarity instead; the interface stays the same.
#
# Papers live in two places:
#   data/sample_papers/ — the 4 invented sample notes (loaded at startup).
#   data/uploads/       — papers Teja uploads (PDF / .txt / .md), indexed
#                         the same way at upload time.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "data" / "sample_papers"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

SECTION_HEADINGS = ["problem", "method", "findings", "limitations", "key claims"]

STOPWORDS = {
    "what", "where", "when", "who", "how", "does", "did", "do", "the",
    "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "for", "to", "and", "his", "her", "he", "she", "it", "with",
    "summarize", "compare", "tell", "me", "about", "list", "give",
}


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str = ""


@dataclass
class Paper:
    source: str          # file name — this is what citations show
    title: str
    text: str
    sections: dict = field(default_factory=dict)  # problem/method/...
    chunks: List[Chunk] = field(default_factory=list)


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF file's bytes (pypdf)."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def parse_sections(text: str) -> dict:
    """Split a paper into sections by its ## headings (or ALL-CAPS lines).

    Returns {section_name: text}. Unknown text goes under "body".
    Research notes use headings like '## Findings'; plain PDFs may have
    none, in which case the whole text is 'body' and summarization falls
    back to sentence extraction.
    """
    sections: dict[str, list[str]] = {}
    current = "body"
    sections[current] = []
    for line in text.splitlines():
        stripped = line.strip()
        heading = None
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
        elif re.fullmatch(r"[A-Z][A-Z ]{3,}", stripped or " "):
            heading = stripped.lower()
        if heading:
            key = next((h for h in SECTION_HEADINGS if h in heading), None)
            if key:
                current = key
                sections.setdefault(current, [])
                continue
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def chunk_text(text: str, source: str, section: str = "", chunk_size: int = 500,
               chunk_overlap: int = 50) -> List[Chunk]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = splitter.split_text(text)
    except Exception:
        pieces = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        Chunk(id=f"{source}#{i}", text=p.strip(), source=source, section=section)
        for i, p in enumerate(pieces) if p.strip()
    ]


def make_paper(source: str, text: str) -> Paper:
    """Build a Paper (title + sections + chunks) from raw text."""
    title = source
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break
    sections = parse_sections(text)
    chunks: List[Chunk] = []
    for name, body in sections.items():
        chunks.extend(chunk_text(body, source=source, section=name))
    if not chunks:
        chunks = chunk_text(text, source=source)
    return Paper(source=source, title=title, text=text, sections=sections, chunks=chunks)


def load_sample_papers() -> List[Paper]:
    papers: List[Paper] = []
    for path in sorted(SAMPLE_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        papers.append(make_paper(path.name, path.read_text(encoding="utf-8")))
    return papers


def load_uploaded_papers() -> List[Paper]:
    papers: List[Paper] = []
    for path in sorted(UPLOAD_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            text = extract_pdf_text(path.read_bytes())
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        papers.append(make_paper(path.name, text))
    return papers


def load_all_papers() -> List[Paper]:
    """Sample papers + anything uploaded so far."""
    return load_sample_papers() + load_uploaded_papers()


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def retrieve(query: str, papers: List[Paper], k: int = 5) -> List[Chunk]:
    """Keyword retrieval over all paper chunks (mock-mode signal).

    Scores each chunk by shared content words with the query, then
    round-robins across papers so no single paper crowds out the others
    — a comparison question needs evidence from BOTH sides.
    """
    words = _content_words(query)
    scored: list[tuple[int, Chunk]] = []
    for paper in papers:
        for chunk in paper.chunks:
            score = sum(1 for w in words if w in chunk.text.lower())
            scored.append((score, chunk))
    scored.sort(key=lambda p: p[0], reverse=True)
    by_source: dict[str, list[Chunk]] = {}
    for score, chunk in scored:
        by_source.setdefault(chunk.source, []).append(chunk)
    ranked: List[Chunk] = []
    sources = sorted(by_source, key=lambda s: -max(
        s2 for s2, c in scored if c.source == s))
    round_no = 0
    while len(ranked) < k:
        added = False
        for src in sources:
            chunks = [c for s, c in scored if c.source == src and s > 0]
            if round_no < len(chunks):
                ranked.append(chunks[round_no])
                added = True
            if len(ranked) >= k:
                break
        if not added:
            break
        round_no += 1
    return ranked[:k] or [c for _, c in scored[:1]]
