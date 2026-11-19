# SAMPLE RESEARCH NOTE — Agentic workflows for multi-step tasks

> This is a short, invented sample note written for the Teja Research Paper
> Agent demo. It is NOT a real published paper: it has no authors, no venue,
> and no real experimental results. It exists only so the agent has papers
> to summarize, compare, and build a literature review from.

## Problem

Some tasks need several steps: plan, search, draft, check, revise. A
single language-model call cannot do all of that well, because it cannot
go back and fix its own mistakes. The question is whether splitting the
work into an agent with explicit steps actually helps, and what it costs.

## Method

This note compares a single-shot answer with a multi-step agent on
multi-hop questions that need two or more documents to answer. The agent
plans its steps, calls a retrieval tool, drafts an answer, critiques the
draft, and revises it. Both approaches are judged on answer correctness
and on whether the answer cites its sources.

## Findings

Multi-step agents outperform single-shot answers on multi-hop questions,
because the agent can retrieve, check, and revise instead of guessing
everything at once. Self-critique reduces errors: the critique step
catches missing citations and unsupported claims before the answer is
shown. The trade-off is cost: agents make more language-model calls, so
they are slower and more expensive per question.

## Limitations

Only multi-hop questions were tested; on simple single-step questions
the agent offers little benefit. The critique step was tested with one
model only, and no human evaluation of the revised answers was done.

## Key claims

- Multi-step agents outperform single-shot answers on multi-hop questions.
- Self-critique reduces errors.
- Agents cost more per question than single-shot answers.
- RAG answers should cite their sources.
