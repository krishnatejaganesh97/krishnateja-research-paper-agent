"""Eval script for the Teja Research Paper Agent.

What this teaches: evals for a research agent — test the whole pipeline,
not one step. Five checks:
  1. Summarize — all 4 sample papers summarized, each with problem /
     method / findings / limitations and a citation.
  2. Compare — both designed contradictions are found (chunk size AND
     self-critique), and at least one gap is reported.
  3. Outline — at least 3 sections, every citation is a real paper in
     the collection, and the contradictions are covered in the outline.
  4. Q&A (chunk size) — the answer shows both sides of the disagreement
     and cites at least 2 papers.
  5. Q&A (agents) — the answer is on-task (mentions agents) and cited.

Run from the project root:
  python backend/eval.py
Exit code 0 = all 5 passed, 1 = at least one failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import agent, papers  # noqa: E402

REQUIRED_NODES = {"plan", "read_papers", "compare", "draft_outline", "critique", "finalize"}


def main() -> int:
    topic = "RAG retrieval, evaluation, and agentic workflows"
    print(f"Running literature review for: {topic}\n")
    out = agent.run_review(topic)
    final = out["final"]
    results: list[dict] = []
    passed = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        passed += ok
        results.append({"test": name, "pass": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}\n")

    # 1. Summarize
    summaries = final.get("summaries", [])
    known = {p.source for p in papers.load_all_papers()}
    fields_ok = all(
        len(s.get(f, "")) >= 10 for s in summaries
        for f in ("problem", "method", "findings", "limitations")
    ) and all(s.get("citations") for s in summaries)
    check("1. Summarize all papers",
          len(summaries) == 4 and fields_ok,
          f"{len(summaries)} summaries (expected 4); all fields + citations present: {fields_ok}; "
          f"sources: {[s['source'] for s in summaries]}")

    # 2. Compare — both designed contradictions + a gap
    comparison = final.get("comparison", {})
    contra_topics = [c["topic"] for c in comparison.get("contradictions", [])]
    check("2. Compare: contradictions + gaps found",
          "chunk size for RAG" in contra_topics
          and "value of self-critique in agents" in contra_topics
          and len(comparison.get("gaps", [])) >= 1,
          f"contradictions: {contra_topics}; "
          f"gaps: {[g['topic'] for g in comparison.get('gaps', [])]}; "
          f"agreements: {len(comparison.get('agreements', []))}")

    # 3. Outline — structure, real citations, contradictions covered
    outline = final.get("outline", {})
    outline_text = " ".join(
        sec["heading"] + " " + sec["summary"] + " " + " ".join(sec["key_points"])
        for sec in outline.get("sections", [])).lower()
    cites_real = all(c in known for c in outline.get("citations", [])) and bool(outline.get("citations"))
    contra_covered = "chunk" in outline_text and "self-critique" in outline_text
    check("3. Outline: structured, cited, contradictions covered",
          len(outline.get("sections", [])) >= 3 and cites_real and contra_covered,
          f"{len(outline.get('sections', []))} sections; citations all real: {cites_real} "
          f"{outline.get('citations')}; contradictions covered: {contra_covered}")

    # 4. Q&A — chunk size: both sides + 2 citations
    qa1 = agent.ask_question("What chunk size works best for RAG?")
    qa1_sources = {s["source"] for s in qa1["sources"]}
    check("4. Q&A chunk size: both sides cited",
          "chunk" in qa1["answer"].lower()
          and "01-rag-retrieval-note.md" in qa1_sources
          and "02-rag-evaluation-note.md" in qa1_sources,
          f"sources: {sorted(qa1_sources)}; answer: {qa1['answer'][:200]}...")

    # 5. Q&A — agents: on-task and cited
    qa2 = agent.ask_question("Do agentic workflows help on multi-step tasks?")
    check("5. Q&A agents: on-task and cited",
          "agent" in qa2["answer"].lower() and bool(qa2["sources"]),
          f"sources: {[s['source'] for s in qa2['sources']]}; "
          f"answer: {qa2['answer'][:200]}...")

    # Graph shape (not counted in the 5, but must hold)
    nodes_seen = {e["node"] for e in out["log"]}
    graph_ok = REQUIRED_NODES <= nodes_seen and out["revisions"] <= 2
    print(f"Graph check: nodes {sorted(nodes_seen)} | revisions {out['revisions']} "
          f"| graph OK: {graph_ok}")
    if not graph_ok:
        print("WARNING: graph shape check failed (not counted in the 5 tests).")

    print(f"\nResult: {passed}/{len(results)} passed (mode: {out['mode']})")
    out_path = Path(__file__).resolve().parent.parent / "eval_results.json"
    out_path.write_text(json.dumps(
        {"mode": out["mode"], "passed": passed, "total": len(results),
         "graph_ok": graph_ok, "results": results}, indent=2))
    print(f"Wrote {out_path}")
    return 0 if passed == len(results) and graph_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
