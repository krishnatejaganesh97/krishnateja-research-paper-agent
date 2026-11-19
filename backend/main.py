"""FastAPI backend for the Teja Research Paper Agent.

Endpoints (same pattern as Products 1 and 2, so Teja can compare all three):
  GET  /health    — is the service up, which mode, how many papers loaded?
  GET  /papers    — list the papers in the collection (sample + uploaded)
  POST /upload    — upload a paper (.pdf / .txt / .md), extract + index it
  POST /review    — {topic} -> full literature review (summaries, comparison,
                    outline) run through the LangGraph pipeline
  POST /ask       — {question} -> grounded answer over the papers, with citations
  GET  /tasks     — sample topics and questions to try
  GET  /          — serves the frontend UI (frontend/index.html)

Run from the project root:
  python -m uvicorn backend.main:app --reload --port 8002
  then open http://localhost:8002
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from backend import agent, papers  # noqa: E402
from backend.llm import detect_mode  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Teja Research Paper Agent",
    description="PhD research paper agent: upload papers, get summaries, contradictions, gaps, and a literature review outline with citations. Mock mode runs without an API key.",
    version="1.0.0",
)

SAMPLE_TOPICS = [
    "RAG retrieval, evaluation, and agentic workflows",
    "How to evaluate a RAG system",
    "Do multi-step agents actually help?",
]

SAMPLE_QUESTIONS = [
    "What chunk size works best for RAG?",
    "Do agentic workflows help on multi-step tasks?",
    "Does self-critique reduce errors?",
    "What should I measure when evaluating a RAG system?",
]


class ReviewRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500,
                       description="Topic of the literature review.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Question about the paper collection.")


@app.get("/health")
def health():
    all_papers = papers.load_all_papers()
    return {
        "status": "ok",
        "mode": detect_mode(),
        "graph": "plan -> read_papers -> compare -> draft_outline -> critique -> (revise | finalize)",
        "papers": len(all_papers),
        "paper_sources": [p.source for p in all_papers],
    }


@app.get("/papers")
def list_papers():
    all_papers = papers.load_all_papers()
    return {"papers": [
        {"source": p.source, "title": p.title, "sections": sorted(p.sections.keys()),
         "chunks": len(p.chunks)} for p in all_papers]}


@app.get("/tasks")
def tasks():
    return {"topics": SAMPLE_TOPICS, "questions": SAMPLE_QUESTIONS}


@app.post("/review")
def review(req: ReviewRequest):
    return agent.run_review(req.topic)


@app.post("/ask")
def ask(req: AskRequest):
    return agent.ask_question(req.question)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a paper (.pdf / .txt / .md), extract its text, index it.

    What this teaches: PDF text extraction (pypdf) + the same
    chunk -> index pipeline as the sample papers. Uploaded papers are
    included in the next /review and /ask calls.
    """
    if not file.filename or not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Please upload a .pdf, .txt or .md file.")
    raw = await file.read()
    dest = UPLOAD_DIR / Path(file.filename).name
    dest.write_bytes(raw)
    try:
        if dest.suffix.lower() == ".pdf":
            text = papers.extract_pdf_text(raw)
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read that file: {e}")
    if len(text.strip()) < 50:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "No readable text found in that file (scanned image PDFs have no text layer).")
    paper = papers.make_paper(dest.name, text)
    return {"filename": dest.name, "title": paper.title,
            "sections": sorted(paper.sections.keys()),
            "chunks_indexed": len(paper.chunks),
            "total_papers": len(papers.load_all_papers())}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
