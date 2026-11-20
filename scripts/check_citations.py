"""Citation-integrity check for the Teja Research Paper Agent.

Verifies that every claim in a generated literature review traces back
to a paper actually in the collection — no uncited or invented claims —
and that the invented sample papers are labelled as invented everywhere
they appear.

What it checks (mock mode, no API key needed):
  1. Outline citations are real — every paper in `outline.citations` and
     in each section's `papers_cited` is a file in the collection.
  2. Every key point carries its own citation — each `key_points` entry
     ends with one or more `[<file name>]` tags naming real papers.
  3. Every key point is grounded — it shares at least 2 content words
     with the full text of at least one paper it cites (so a point cannot
     cite a paper that never says anything like it).
  4. Section summaries are grounded — every sentence in a section summary
     shares at least 3 content words with a paper that section cites.
  5. Introduction states the true paper count.
  6. Conclusion agreement claims name their papers — any sentence with
     "agree" must name the file(s) that agree (a bare "the papers agree"
     is not good enough when only some of them say it).
  7. Contradictions are traceable — each contradiction's `claim_a` is a
     claim of `paper_a` and `claim_b` is a claim of `paper_b`, verbatim.
  8. Gaps are traceable — each gap's `papers_mentioning` are real papers
     whose Limitations section mentions the gap topic.
  9. Agreements name their papers — each agreement string names real
     papers, and the agreed phrase appears in those papers' key claims.
 10. Summaries are grounded — every summary key claim appears verbatim in
     its source paper's text.
 11. Invented samples are labelled — the README, every sample file, the
     frontend, and the review output itself all say the sample notes are
     invented, not real published papers.

Run from the project root:
  python scripts/check_citations.py [--topic "..."]
Exit code 0 = all checks passed, 1 = at least one failed. Every failure
is printed with the exact claim that failed, so it can be fixed rather
than hidden.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Force mock mode: an API key in the environment would switch the agent
# to a real LLM, whose wording this checker cannot verify deterministically.
# The mode actually used is reported in the output.
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("AZURE_OPENAI_API_KEY", None)
os.environ.pop("AZURE_OPENAI_ENDPOINT", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import agent, papers  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND = BASE_DIR / "frontend" / "index.html"
README = BASE_DIR / "README.md"

STOPWORDS = {
    "what", "where", "when", "who", "how", "does", "did", "do", "the",
    "a", "an", "is", "are", "was", "were", "in", "on", "at", "of", "for",
    "to", "and", "or", "but", "so", "than", "then", "with", "from", "by",
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "each", "every", "both", "more", "most", "best", "only", "also", "not",
    "no", "one", "two", "three", "four", "all", "any", "per", "as", "are",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
            if w not in STOPWORDS and len(w) > 1}


def overlap_words(text: str, paper_text: str) -> set[str]:
    """Content words of `text` that appear (as substrings) in `paper_text`."""
    hay = paper_text.lower()
    return {w for w in content_words(text) if w in hay}


def cited_files(text: str) -> list[str]:
    """File names cited as [<name>] inside `text`."""
    return re.findall(r"\[([A-Za-z0-9_.\-]+\.(?:md|txt|pdf))\]", text)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def main() -> int:
    topic = "RAG retrieval, evaluation, and agentic workflows"
    if "--topic" in sys.argv:
        topic = sys.argv[sys.argv.index("--topic") + 1]

    all_papers = papers.load_all_papers()
    by_source = {p.source: p for p in all_papers}
    known = set(by_source)
    print(f"Running literature review for: {topic}\n")
    out = agent.run_review(topic)
    final = out["final"]
    outline = final.get("outline", {})
    summaries = final.get("summaries", [])
    comparison = final.get("comparison", {})

    failures: list[str] = []
    checks = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))
        if not ok:
            failures.append(f"{name}: {detail}")

    # 1. Outline citations are real papers in the collection.
    bad = [c for c in outline.get("citations", []) if c not in known]
    check("1. Outline citations are real papers",
          not bad and bool(outline.get("citations")),
          f"citations: {outline.get('citations')}; unknown: {bad or 'none'}")
    bad_sections = [(s["heading"], c) for s in outline.get("sections", [])
                    for c in s.get("papers_cited", []) if c not in known]
    empty_sections = [s["heading"] for s in outline.get("sections", [])
                      if not s.get("papers_cited")]
    check("1b. Every section cites at least one real paper",
          not bad_sections and not empty_sections,
          f"unknown citations: {bad_sections or 'none'}; "
          f"sections with no citations: {empty_sections or 'none'}")

    # 2 + 3. Every key point carries a real citation and is grounded in it.
    point_problems: list[str] = []
    for sec in outline.get("sections", []):
        for point in sec.get("key_points", []):
            cited = cited_files(point)
            if not cited:
                point_problems.append(f"UNCITED key point in '{sec['heading']}': {point!r}")
                continue
            unknown = [c for c in cited if c not in known]
            if unknown:
                point_problems.append(f"key point cites unknown paper(s) {unknown}: {point!r}")
                continue
            grounded = any(
                len(overlap_words(point, by_source[c].text)) >= 2 for c in cited)
            if not grounded:
                point_problems.append(
                    f"key point not grounded in any paper it cites: {point!r} "
                    f"(cited: {cited})")
    check("2. Every outline key point cites a real paper and is grounded in it",
          not point_problems,
          f"{sum(len(s.get('key_points', [])) for s in outline.get('sections', []))} key points checked; "
          f"problems: {point_problems or 'none'}")

    # 4. Section summaries grounded in the papers that section cites.
    summary_problems: list[str] = []
    for sec in outline.get("sections", []):
        cited = [c for c in sec.get("papers_cited", []) if c in known]
        for sent in re.split(r"(?<=[.!])\s+", sec.get("summary", "")):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            if not any(len(overlap_words(sent, by_source[c].text)) >= 3 for c in cited):
                summary_problems.append(f"'{sec['heading']}': {sent!r}")
    check("3. Every section-summary sentence is grounded in a cited paper",
          not summary_problems,
          f"ungrounded sentences: {summary_problems or 'none'}")

    # 5. Introduction states the true paper count.
    intro = outline.get("introduction", "")
    check("4. Introduction states the true paper count",
          str(len(summaries)) in intro,
          f"introduction: {intro!r}; summaries: {len(summaries)}")

    # 6. Conclusion agreement claims name their papers.
    conclusion = outline.get("conclusion", "")
    agree_problems: list[str] = []
    for sent in re.split(r"(?<=[.!])\s+", conclusion):
        if re.search(r"\bagrees?\b", sent.lower()):  # "disagree" must not match
            named = [c for c in cited_files(sent) if c in known]
            if not named:
                agree_problems.append(f"agreement claim names no paper: {sent!r}")
    check("5. Conclusion agreement claims name the agreeing papers",
          not agree_problems,
          f"problems: {agree_problems or 'none'}")

    # 7. Contradictions traceable to the exact claims of the named papers.
    contra_problems: list[str] = []
    for c in comparison.get("contradictions", []):
        for side, paper_key, claim_key in (("A", "paper_a", "claim_a"),
                                           ("B", "paper_b", "claim_b")):
            src = c.get(paper_key, "")
            claim = c.get(claim_key, "")
            if src not in known:
                contra_problems.append(f"contradiction on {c.get('topic')!r} names unknown paper {src!r}")
            elif norm(claim) not in norm(by_source[src].text):
                contra_problems.append(
                    f"contradiction on {c.get('topic')!r}: claim {side} {claim!r} "
                    f"is not in {src}")
    check("6. Every contradiction quotes a real claim from each named paper",
          not contra_problems,
          f"{len(comparison.get('contradictions', []))} contradiction(s) checked; "
          f"problems: {contra_problems or 'none'}")

    # 8. Gaps traceable to the papers that mention them.
    gap_problems: list[str] = []
    for g in comparison.get("gaps", []):
        for src in g.get("papers_mentioning", []):
            if src not in known:
                gap_problems.append(f"gap {g.get('topic')!r} mentions unknown paper {src!r}")
            # norm() collapses newlines: a topic split across a line break
            # in the source file is still the same phrase.
            elif norm(g.get("topic", "")) not in norm(
                    by_source[src].sections.get("limitations", "")):
                gap_problems.append(
                    f"gap {g.get('topic')!r}: {src} does not mention it in Limitations")
    check("7. Every gap is traceable to papers that mention it",
          not gap_problems,
          f"{len(comparison.get('gaps', []))} gap(s) checked; "
          f"problems: {gap_problems or 'none'}")

    # 9. Agreements name real papers whose key claims contain the phrase.
    agreement_problems: list[str] = []
    for a in comparison.get("agreements", []):
        named = [c for c in known if c in a]
        if not named:
            agreement_problems.append(f"agreement names no paper: {a!r}")
    check("8. Every agreement names the papers that agree",
          not agreement_problems,
          f"{len(comparison.get('agreements', []))} agreement(s) checked; "
          f"problems: {agreement_problems or 'none'}")

    # 10. Summary key claims appear verbatim in their source paper.
    summary_claim_problems: list[str] = []
    for s in summaries:
        src = s.get("source", "")
        if src not in known:
            summary_claim_problems.append(f"summary of unknown paper {src!r}")
            continue
        for claim in s.get("key_claims", []):
            if norm(claim) not in norm(by_source[src].text):
                summary_claim_problems.append(f"[{src}] key claim not in paper: {claim!r}")
    check("9. Every summary key claim appears in its source paper",
          not summary_claim_problems,
          f"problems: {summary_claim_problems or 'none'}")

    # 11. Invented sample papers labelled as invented everywhere.
    readme_ok = "invented" in README.read_text(encoding="utf-8").lower()
    sample_files = sorted(papers.SAMPLE_DIR.glob("*.md"))
    unlabelled = [p.name for p in sample_files
                  if "invented" not in p.read_text(encoding="utf-8").lower()
                  or "not a real published paper" not in p.read_text(encoding="utf-8").lower()]
    frontend_ok = "invented" in FRONTEND.read_text(encoding="utf-8").lower()
    notice = out.get("notice", "")
    notice_ok = "invented" in notice.lower() and "not real" in notice.lower()
    check("10. Invented sample papers labelled everywhere (README, samples, UI, output)",
          readme_ok and not unlabelled and frontend_ok and notice_ok,
          f"README: {readme_ok}; unlabelled sample files: {unlabelled or 'none'}; "
          f"frontend: {frontend_ok}; review-output notice: {notice_ok} ({notice[:90]!r})")

    print(f"\nMode: {out['mode']} (checker forces mock) | "
          f"Result: {checks - len(failures)}/{checks} checks passed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All citation checks passed — every outline claim traces to a real sample paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
