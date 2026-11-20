# Teja Research Paper Agent — Product 3
# A PhD research paper agent built with LangGraph.
#
# What Teja learns here (Products 1 + 2 combined, applied to real work):
#   1. RAG over papers — chunking + retrieval from Product 1, but the
#      documents are research papers with structure (Problem / Method /
#      Findings / Limitations), so the agent reads sections, not blobs.
#   2. LangGraph — the same graph shape as Product 2, with new nodes:
#      plan -> read_papers -> compare -> draft_outline -> critique ->
#      (revise | finalize). "Research" becomes "read the papers",
#      "draft" becomes "write the literature review outline".
#   3. Structured outputs — every step must match a Pydantic model:
#      PaperSummary, Comparison, LiteratureReview. A malformed step is
#      rejected and revised, never shown as-is.
#   4. Comparison — the agent does not just summarize each paper alone;
#      it compares them and flags contradictions and gaps. That is the
#      part a plain chatbot cannot do, and the part a PhD student needs.
#   5. Evals — backend/eval.py runs summarize / compare / outline / Q&A
#      and checks structure, citations, and contradiction detection.
#
# Modes (same as Products 1 and 2 — no code change to switch):
#   mock   — no key set. Deterministic, labelled mock: extractive
#            summaries from paper sections + rule-based comparison.
#            Runs fully offline.
#   openai — OPENAI_API_KEY set: real LLM summaries / comparison / outline.
#   azure  — AZURE_OPENAI_* set: same, via Azure OpenAI.
# Never hard-code a key. See .env.example.

from __future__ import annotations

import re
from operator import add
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from backend import llm, papers  # noqa: E402

MAX_REVISIONS = 2

# ---------------------------------------------------------------------------
# Structured outputs — the contract for every step.
# ---------------------------------------------------------------------------
class PaperSummary(BaseModel):
    source: str = Field(..., description="Paper file name, e.g. 01-rag-retrieval-note.md")
    title: str
    problem: str = Field(..., min_length=10)
    method: str = Field(..., min_length=10)
    findings: str = Field(..., min_length=10)
    limitations: str = Field(..., min_length=10)
    key_claims: list[str] = Field(..., min_length=1)
    citations: list[str] = Field(..., min_length=1)


class Contradiction(BaseModel):
    topic: str
    paper_a: str
    claim_a: str
    paper_b: str
    claim_b: str
    note: str = Field(..., description="What a researcher should do about it.")


class Gap(BaseModel):
    topic: str
    description: str
    papers_mentioning: list[str] = Field(default_factory=list)


class Comparison(BaseModel):
    agreements: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)


class OutlineSection(BaseModel):
    heading: str
    summary: str = Field(..., min_length=10)
    papers_cited: list[str] = Field(..., min_length=1)
    key_points: list[str] = Field(..., min_length=1)


class LiteratureReview(BaseModel):
    title: str
    introduction: str = Field(..., min_length=20)
    sections: list[OutlineSection] = Field(..., min_length=2)
    conclusion: str = Field(..., min_length=20)
    citations: list[str] = Field(..., min_length=1)


class ReviewOutput(BaseModel):
    topic: str
    summaries: list[PaperSummary] = Field(..., min_length=1)
    comparison: Comparison
    outline: LiteratureReview
    mode: str = Field("mock", description="mock | openai | azure")


# ---------------------------------------------------------------------------
# Mock comparison knowledge — the heuristic the mock mode uses.
# These are phrase pairs that mean opposite things. The mock comparer
# matches them against each paper's key claims. In real mode the LLM
# does the comparison instead; this table is only the offline fallback,
# and it is labelled as such everywhere it is used.
# ---------------------------------------------------------------------------
OPPOSITION_PAIRS = [
    ("larger chunks improve", "smaller chunks improve", "chunk size for RAG"),
    ("self-critique reduces errors", "self-critique provides limited benefit",
     "value of self-critique in agents"),
]

AGREEMENT_PHRASES = [
    "cite their sources",
    "citation precision",
    "faithfulness",
]

GAP_TOPICS = [
    ("human evaluation", ["human evaluation"],
     "No paper in the collection reports a human evaluation of its claims."),
    ("multilingual documents", ["multilingual"],
     "No paper tests its findings on multilingual documents."),
]


# ---------------------------------------------------------------------------
# State — the shared blackboard (same pattern as Product 2).
# ---------------------------------------------------------------------------
class ReviewState(dict):
    pass


def _log(node: str, detail: str) -> dict:
    return {"log": [{"node": node, "detail": detail}]}


def _first_sentences(text: str, n: int = 2) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!])\s+", text) if len(s.strip()) > 15]
    return " ".join(sentences[:n])


# ---------------------------------------------------------------------------
# Mock summarization — read the paper's own sections, don't invent.
# ---------------------------------------------------------------------------
def _mock_summarize(paper: "papers.Paper") -> dict:
    sec = paper.sections
    claims: list[str] = []
    if "key claims" in sec:
        claims = [l.strip("- ").strip() for l in sec["key claims"].splitlines()
                  if l.strip().startswith("-") and len(l.strip()) > 3]
    if not claims:  # papers without a Key claims section: take finding sentences
        claims = [_first_sentences(sec.get("findings", paper.text), 2)]
    return {
        "source": paper.source,
        "title": paper.title,
        "problem": _first_sentences(sec.get("problem", sec.get("body", paper.text)), 2),
        "method": _first_sentences(sec.get("method", sec.get("body", paper.text)), 2),
        "findings": _first_sentences(sec.get("findings", sec.get("body", paper.text)), 2),
        "limitations": _first_sentences(sec.get("limitations", "Not stated in the paper."), 2),
        "key_claims": claims,
        "citations": [paper.source],
    }


def _llm_summarize(paper: "papers.Paper") -> Optional[dict]:
    try:
        payload = llm.llm_json(
            "Summarize this research paper as JSON matching this shape:\n"
            '{"source": "<file name>", "title": "...", "problem": "...", '
            '"method": "...", "findings": "...", "limitations": "...", '
            '"key_claims": ["...", "..."], "citations": ["<file name>"]}\n'
            "Use ONLY what is in the paper. Do not invent results.\n\n"
            f"Source file: {paper.source}\n\n{paper.text[:6000]}"
        )
        payload["source"] = paper.source
        payload["citations"] = [paper.source]
        return PaperSummary(**payload).model_dump()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mock comparison — phrase-pair matching + shared-claim and gap scans.
# ---------------------------------------------------------------------------
def _mock_compare(summaries: list[dict], all_papers: list["papers.Paper"]) -> dict:
    contradictions: list[dict] = []
    for phrase_a, phrase_b, topic in OPPOSITION_PAIRS:
        hit_a = hit_b = None
        for s in summaries:
            claims = " ".join(s["key_claims"]).lower()
            if phrase_a in claims:
                hit_a = s
            if phrase_b in claims:
                hit_b = s
        if hit_a and hit_b and hit_a["source"] != hit_b["source"]:
            claim_a = next(c for c in hit_a["key_claims"] if phrase_a in c.lower())
            claim_b = next(c for c in hit_b["key_claims"] if phrase_b in c.lower())
            contradictions.append({
                "topic": topic,
                "paper_a": hit_a["source"], "claim_a": claim_a,
                "paper_b": hit_b["source"], "claim_b": claim_b,
                "note": (
                    "These papers disagree. Do not cite only one side — read both "
                    "papers' methods and check which setting each one tested before "
                    "deciding which claim applies to your work."
                ),
            })
    agreements: list[str] = []
    for phrase in AGREEMENT_PHRASES:
        holders = [s["source"] for s in summaries
                   if phrase in " ".join(s["key_claims"]).lower()]
        if len(holders) >= 2:
            agreements.append(
                f"At least {len(holders)} papers agree: {phrase} matters "
                f"({', '.join(holders)})."
            )
    gaps: list[dict] = []
    findings_text = " ".join(
        p.sections.get("findings", "") for p in all_papers).lower()
    limitations_text = " ".join(
        p.sections.get("limitations", "") for p in all_papers).lower()
    for topic, keywords, description in GAP_TOPICS:
        in_limitations = any(k in limitations_text for k in keywords)
        in_findings = any(k in findings_text for k in keywords)
        if in_limitations and not in_findings:
            mentioners = [p.source for p in all_papers
                          if any(k in p.sections.get("limitations", "").lower()
                                 for k in keywords)]
            gaps.append({"topic": topic, "description": description,
                         "papers_mentioning": mentioners})
    return {"agreements": agreements, "contradictions": contradictions, "gaps": gaps}


def _llm_compare(summaries: list[dict]) -> Optional[dict]:
    try:
        context = "\n\n".join(
            f"[{s['source']}] {s['title']}\nClaims: " + "; ".join(s["key_claims"])
            for s in summaries)
        payload = llm.llm_json(
            "Compare these papers and reply with JSON matching this shape:\n"
            '{"agreements": ["..."], "contradictions": [{"topic": "...", '
            '"paper_a": "...", "claim_a": "...", "paper_b": "...", '
            '"claim_b": "...", "note": "..."}], '
            '"gaps": [{"topic": "...", "description": "...", '
            '"papers_mentioning": ["..."]}]}\n'
            "Only report contradictions and gaps supported by the claims below.\n\n"
            f"{context}"
        )
        return Comparison(**payload).model_dump()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mock outline — one section per theme, built from the summaries.
#
# Citation rule (enforced by scripts/check_citations.py and the critique
# node): every key point ends with [<file name>] tags naming ONLY the
# papers that actually state that point. A section-level `papers_cited`
# list is not enough — a reader must be able to trace each individual
# claim to its source. The conclusion likewise names the papers behind
# any agreement claim instead of saying "the papers agree".
# ---------------------------------------------------------------------------
def _cite(point: str, sources: list[str]) -> str:
    """Append per-point source tags, e.g. '... [01-rag-retrieval-note.md]'."""
    return point + " " + " ".join(f"[{s}]" for s in sources)


def _mock_outline(topic: str, summaries: list[dict], comparison: dict) -> dict:
    by_source = {s["source"]: s for s in summaries}

    def theme_section(heading: str, sources: list[str], points: list[str]) -> dict:
        present = [s for s in sources if s in by_source]
        summary = " ".join(
            _first_sentences(by_source[s]["findings"], 1) for s in present)
        return {"heading": heading, "summary": summary or "See the papers cited.",
                "papers_cited": present,
                "key_points": points or [f"See {s}." for s in present]}

    sections = [
        theme_section(
            "Retrieval for RAG",
            ["01-rag-retrieval-note.md", "02-rag-evaluation-note.md"],
            [_cite("Chunk size changes what retrieval can return — and the notes "
                   "disagree on which size is best.",
                   ["01-rag-retrieval-note.md", "02-rag-evaluation-note.md"]),
             _cite("Dense retrieval wins on paraphrased questions; keyword retrieval "
                   "wins on exact terms.",
                   ["01-rag-retrieval-note.md"]),
             _cite("Re-ranking gives the best supported answers.",
                   ["01-rag-retrieval-note.md"])]),
        theme_section(
            "Evaluating RAG systems",
            ["02-rag-evaluation-note.md"],
            [_cite("Faithfulness is the measure to track first.",
                   ["02-rag-evaluation-note.md"]),
             _cite("Citation precision depends on chunk size.",
                   ["02-rag-evaluation-note.md"]),
             _cite("Every citation should be checked against its passage.",
                   ["02-rag-evaluation-note.md"])]),
        theme_section(
            "Agentic workflows and self-critique",
            ["03-agentic-workflows-note.md", "04-self-critique-limits-note.md"],
            [_cite("Multi-step agents beat single-shot answers on multi-hop questions.",
                   ["03-agentic-workflows-note.md"]),
             _cite("The notes disagree on how much self-critique helps.",
                   ["03-agentic-workflows-note.md", "04-self-critique-limits-note.md"]),
             _cite("Agents cost more per question than single-shot answers.",
                   ["03-agentic-workflows-note.md"])]),
        theme_section(
            "Contradictions and open questions",
            [s["source"] for s in summaries],
            [_cite(f"Contradiction — {c['topic']}: {c['paper_a']} says "
                   f"'{c['claim_a']}' vs {c['paper_b']} says '{c['claim_b']}'.",
                   [c["paper_a"], c["paper_b"]])
             for c in comparison["contradictions"]]
            + [_cite(f"Gap — {g['topic']}: {g['description']}",
                     g["papers_mentioning"])
               for g in comparison["gaps"]]),
    ]
    return {
        "title": f"Literature review: {topic}",
        "introduction": (
            f"This review covers {len(summaries)} papers on {topic}. It summarizes "
            "each paper, compares their claims, and flags where they agree, where "
            "they contradict each other, and what none of them covers."
        ),
        "sections": sections,
        # "The papers agree" would be wrong here: only 01, 02 and 03 say RAG
        # answers must cite their sources — 04 never makes that claim. So
        # the conclusion names the three agreeing notes explicitly.
        "conclusion": (
            "Three of the four notes — [01-rag-retrieval-note.md], "
            "[02-rag-evaluation-note.md] and [03-agentic-workflows-note.md] — "
            "agree that RAG answers must cite their sources "
            "([04-self-critique-limits-note.md] does not make that claim). "
            "The notes disagree on chunk size and on the value of self-critique. "
            "The main open questions are human evaluation and multilingual "
            "documents, which no note in this collection studies."
        ),
        "citations": [s["source"] for s in summaries],
    }


def _llm_outline(topic: str, summaries: list[dict], comparison: dict) -> Optional[dict]:
    try:
        context = "\n\n".join(
            f"[{s['source']}] {s['title']}\nFindings: {s['findings']}" for s in summaries)
        contra = "; ".join(
            f"{c['topic']}: {c['paper_a']} vs {c['paper_b']}" for c in comparison["contradictions"])
        payload = llm.llm_json(
            "Write a literature review outline as JSON matching this shape:\n"
            '{"title": "...", "introduction": "...", '
            '"sections": [{"heading": "...", "summary": "...", '
            '"papers_cited": ["<file name>"], "key_points": ["..."]}], '
            '"conclusion": "...", "citations": ["<file name>", ...]}\n'
            "Use ONLY the papers below. Cite the file name for every claim. "
            "End EVERY key point with [<file name>] tags naming ONLY the "
            "papers that actually state that point — never cite a paper for "
            "a claim it does not make, and never write 'the papers agree' "
            "without naming which papers agree. "
            f"Cover the contradictions: {contra or 'none found'}.\n\n"
            f"Topic: {topic}\n\n{context}"
        )
        return LiteratureReview(**payload).model_dump()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Nodes — one function per graph step.
# ---------------------------------------------------------------------------
def plan_node(state: dict) -> dict:
    n = len(state["papers"])
    plan = [
        f"Read all {n} papers and summarize each one (problem, method, findings, limitations).",
        "Compare the summaries: find agreements, contradictions, and gaps.",
        "Draft a literature review outline from the summaries and comparison.",
        "Critique the outline: every claim cited, every citation real, contradictions covered.",
        "Revise if the critique finds issues, then finalize.",
    ]
    return {"plan": plan, **_log("plan", f"Planned {len(plan)} steps for {n} papers.")}


def read_node(state: dict) -> dict:
    mode = llm.detect_mode()
    summaries: list[dict] = []
    for paper in state["paper_objs"]:
        summary = None
        if mode != "mock":
            summary = _llm_summarize(paper)
        if summary is None:
            summary = _mock_summarize(paper)
            summary = PaperSummary(**summary).model_dump()
        summaries.append(summary)
    return {"summaries": summaries, **_log(
        "read_papers",
        f"Summarized {len(summaries)} papers "
        f"({'LLM, ' + mode if mode != 'mock' else 'mock — extractive from sections'}).")}


def compare_node(state: dict) -> dict:
    mode = llm.detect_mode()
    comparison = None
    if mode != "mock":
        comparison = _llm_compare(state["summaries"])
    if comparison is None:
        comparison = _mock_compare(state["summaries"], state["paper_objs"])
        comparison = Comparison(**comparison).model_dump()
    return {"comparison": comparison, **_log(
        "compare",
        f"Compared papers: {len(comparison['agreements'])} agreement(s), "
        f"{len(comparison['contradictions'])} contradiction(s), "
        f"{len(comparison['gaps'])} gap(s).")}


def draft_node(state: dict) -> dict:
    mode = llm.detect_mode()
    outline = None
    if mode != "mock":
        outline = _llm_outline(state["topic"], state["summaries"], state["comparison"])
    if outline is None:
        outline = _mock_outline(state["topic"], state["summaries"], state["comparison"])
        outline = LiteratureReview(**outline).model_dump()
    return {"outline": outline, **_log(
        "draft_outline",
        f"Drafted literature review outline: {len(outline['sections'])} sections.")}


def critique_node(state: dict) -> dict:
    """Check the review. Issues decide whether the graph loops back.

    Besides structure and real citations, this checks citation integrity
    per claim: every key point must carry its own [<file>] citation to a
    real paper, every contradiction must quote a real claim from each
    paper it names, every gap must trace to a paper that mentions it,
    and any agreement claim in the conclusion must name its papers.
    """
    outline, summaries = state["outline"], state["summaries"]
    known = {s["source"] for s in summaries}
    claims_by_source = {s["source"]: " ".join(s.get("key_claims", [])).lower()
                        for s in summaries}
    issues: list[str] = []
    for cite in outline.get("citations", []):
        if cite not in known:
            issues.append(f"Outline cites [{cite}], which is not in the paper collection.")
    for sec in outline.get("sections", []):
        if not sec.get("papers_cited"):
            issues.append(f"Section '{sec['heading']}' cites no papers.")
        for cite in sec.get("papers_cited", []):
            if cite not in known:
                issues.append(f"Section '{sec['heading']}' cites unknown paper [{cite}].")
        if len(sec.get("key_points", [])) < 1:
            issues.append(f"Section '{sec['heading']}' has no key points.")
        for point in sec.get("key_points", []):
            cited = re.findall(r"\[([A-Za-z0-9_.\-]+\.(?:md|txt|pdf))\]", point)
            if not cited:
                issues.append(f"Key point in '{sec['heading']}' has no citation: {point!r}")
            for cite in cited:
                if cite not in known:
                    issues.append(f"Key point in '{sec['heading']}' cites unknown paper [{cite}].")
    for s in summaries:
        if not s.get("citations"):
            issues.append(f"Summary of [{s['source']}] has no citations.")
        for f in ("problem", "method", "findings", "limitations"):
            if len(s.get(f, "")) < 10:
                issues.append(f"Summary of [{s['source']}] has an empty '{f}'.")
    for c in state["comparison"].get("contradictions", []):
        for paper_key, claim_key in (("paper_a", "claim_a"), ("paper_b", "claim_b")):
            src, claim = c.get(paper_key, ""), c.get(claim_key, "")
            if src not in known:
                issues.append(f"Contradiction on '{c.get('topic')}' names unknown paper [{src}].")
            elif re.sub(r"\s+", " ", claim.lower()).strip() not in claims_by_source.get(src, ""):
                issues.append(f"Contradiction on '{c.get('topic')}': the quoted claim "
                              f"is not a claim of [{src}].")
    for g in state["comparison"].get("gaps", []):
        for src in g.get("papers_mentioning", []):
            if src not in known:
                issues.append(f"Gap '{g.get('topic')}' mentions unknown paper [{src}].")
    conclusion = outline.get("conclusion", "")
    for sent in re.split(r"(?<=[.!])\s+", conclusion):
        if re.search(r"\bagrees?\b", sent.lower()):
            named = [c for c in re.findall(r"\[([A-Za-z0-9_.\-]+\.(?:md|txt|pdf))\]", sent)
                     if c in known]
            if not named:
                issues.append(f"Conclusion claims agreement without naming the papers: {sent!r}")
    contra_topics = {c["topic"] for c in state["comparison"].get("contradictions", [])}
    outline_text = " ".join(
        sec["heading"] + " " + sec["summary"] + " " + " ".join(sec["key_points"])
        for sec in outline.get("sections", [])).lower()
    for topic in contra_topics:
        if topic.split()[0] not in outline_text:
            issues.append(f"Contradiction on '{topic}' is not covered in the outline.")
    critique = ("No issues found — outline is cited, structured, and covers the contradictions."
                if not issues else "Issues: " + "; ".join(issues))
    return {"critique": critique, "critique_issues": issues, **_log("critique", critique)}


def finalize_node(state: dict) -> dict:
    final = {
        "topic": state["topic"],
        "summaries": state["summaries"],
        "comparison": state["comparison"],
        "outline": state["outline"],
        "mode": llm.detect_mode(),
    }
    checked = ReviewOutput(**final).model_dump()
    if state.get("revisions", 0) > 0:
        checked["outline"]["conclusion"] += (
            f" (Revised {state['revisions']} time(s) after self-critique.)")
    return {"final": checked, **_log(
        "finalize",
        f"Literature review ready — {len(checked['summaries'])} summaries, "
        f"{len(checked['comparison']['contradictions'])} contradictions, "
        f"{len(checked['outline']['sections'])} outline sections.")}


def _should_revise(state: dict) -> str:
    if state.get("critique_issues") and state.get("revisions", 0) < MAX_REVISIONS:
        return "revise"
    return "finalize"


def _revise_node(state: dict) -> dict:
    return {"revisions": state.get("revisions", 0) + 1, **_log(
        "revise",
        f"Revision {state.get('revisions', 0) + 1}/{MAX_REVISIONS} — "
        f"re-drafting to fix: {'; '.join(state.get('critique_issues', []))}")}


def build_graph():
    from langgraph.graph import END, START, StateGraph
    from typing import TypedDict

    class _State(TypedDict, total=False):
        topic: str
        papers: list[str]
        paper_objs: list
        plan: list[str]
        summaries: list[dict]
        comparison: dict
        outline: dict
        critique: str
        critique_issues: list[str]
        revisions: int
        final: dict
        log: Annotated[list[dict], add]

    graph = StateGraph(_State)
    graph.add_node("plan", plan_node)
    graph.add_node("read_papers", read_node)
    graph.add_node("compare", compare_node)
    graph.add_node("draft_outline", draft_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", _revise_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "read_papers")
    graph.add_edge("read_papers", "compare")
    graph.add_edge("compare", "draft_outline")
    graph.add_edge("draft_outline", "critique")
    graph.add_conditional_edges("critique", _should_revise,
                                {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "draft_outline")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_review(topic: str, paper_objs: Optional[list] = None) -> dict:
    """Run the full literature-review pipeline over the paper collection."""
    objs = paper_objs if paper_objs is not None else papers.load_all_papers()
    graph = build_graph()
    initial: dict = {
        "topic": topic,
        "papers": [p.source for p in objs],
        "paper_objs": objs,
        "plan": [], "summaries": [], "comparison": {}, "outline": {},
        "critique": "", "critique_issues": [], "revisions": 0,
        "final": {}, "log": [],
    }
    result = graph.invoke(initial)
    sample_sources = [p.source for p in objs if papers.is_sample_source(p.source)]
    notice = ""
    if sample_sources:
        if len(sample_sources) == len(objs):
            notice = ("All papers in this review are invented sample notes. "
                      + papers.SAMPLE_NOTICE)
        else:
            notice = ("These papers in this review are invented sample notes: "
                      + ", ".join(sample_sources) + ". " + papers.SAMPLE_NOTICE)
    return {
        "topic": topic,
        "mode": llm.detect_mode(),
        "notice": notice,
        "plan": result.get("plan", []),
        "log": result.get("log", []),
        "revisions": result.get("revisions", 0),
        "critique": result.get("critique", ""),
        "final": result.get("final", {}),
    }


# ---------------------------------------------------------------------------
# Q&A over the paper collection — Product 1's RAG answer, reused.
# ---------------------------------------------------------------------------
def ask_question(question: str, paper_objs: Optional[list] = None) -> dict:
    objs = paper_objs if paper_objs is not None else papers.load_all_papers()
    chunks = papers.retrieve(question, objs, k=5)
    sources = [{"source": c.source, "text": c.text, "section": c.section}
               for c in chunks]
    mode = llm.detect_mode()
    if mode != "mock":
        try:
            context = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
            answer = llm.llm_text(
                "Answer the question using ONLY the context below. "
                "If the papers disagree, say so and show both sides. "
                "Cite the source file for each fact, e.g. [01-rag-retrieval-note.md].\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            return {"question": question, "answer": answer, "sources": sources, "mode": mode}
        except Exception:
            pass
    # Mock: extractive answer — the most relevant sentences, with citations.
    stop = {"what", "where", "when", "who", "how", "does", "did", "do",
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "of", "for", "to", "and"}
    words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stop}
    sentences: list[tuple[int, str, str]] = []
    for c in chunks:
        for sent in re.split(r"(?<=[.!])\s+", c.text):
            sent = sent.strip().lstrip("- ").strip()
            if len(sent) < 15 or sent.startswith("#") or sent.startswith(">"):
                continue  # skip titles and the sample-note disclaimer block
            sentences.append((sum(1 for w in words if w in sent.lower()), sent, c.source))
    sentences.sort(key=lambda s: s[0], reverse=True)
    top = [s for s in sentences if s[0] > 0][:3] or sentences[:2]
    # Surface disagreement: if the retrieved chunks come from papers with
    # opposing claims, say so explicitly instead of picking one side.
    answer = "Based on the papers:\n" + "\n".join(
        f"- {sent} [{src}]" for _, sent, src in top)
    srcs = {s for _, _, s in top}
    if len(srcs) >= 2:
        answer += ("\n\nNote: the answer draws on more than one paper — "
                   "check the comparison view, because these papers do not "
                   "always agree with each other.")
    seen, uniq = set(), []
    for s in sources:
        if s["source"] not in seen:
            seen.add(s["source"])
            uniq.append(s)
    return {"question": question, "answer": answer, "sources": uniq, "mode": "mock"}
