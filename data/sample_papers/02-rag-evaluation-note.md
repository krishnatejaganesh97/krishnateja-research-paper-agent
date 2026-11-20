# SAMPLE RESEARCH NOTE — Evaluating RAG systems

> This is a short, invented sample note written for the Krishnateja Research Paper
> Agent demo. It is NOT a real published paper: it has no authors, no venue,
> and no real experimental results. It exists only so the agent has papers
> to summarize, compare, and build a literature review from.

## Problem

Teams building RAG systems need a way to tell whether a change helped or
hurt. Judging by reading a few answers by hand does not scale, and a
single score hides what went wrong. The question is which automatic
evaluation measures are worth tracking.

## Method

This note reviews evaluation measures for RAG: faithfulness (is every
claim in the answer supported by retrieved passages?), citation precision
(does each citation actually support the claim next to it?), and answer
relevance (does the answer address the question?). It also compares chunk
sizes, because chunk size changes what the retriever can return.

## Findings

Smaller chunks improve citation precision because each retrieved passage
is tightly focused, so a citation points at exactly the claim it supports.
Faithfulness is the most important measure to track first: an unfaithful
answer is wrong even when it sounds fluent. RAG answers should cite their
sources, and every citation should be checked against the passage it
points to.

## Limitations

The measures were compared on a small, hand-built test set, not on a
standard benchmark. No human evaluation was done to check whether the
automatic scores agree with human judges. Multilingual documents were
not tested.

## Key claims

- Smaller chunks improve citation precision.
- Faithfulness is the most important RAG evaluation measure.
- RAG answers should cite their sources.
- Every citation should be checked against its passage.
