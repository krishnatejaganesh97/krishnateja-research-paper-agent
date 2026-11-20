# SAMPLE RESEARCH NOTE — Limits of self-critique in LLM agents

> This is a short, invented sample note written for the Krishnateja Research Paper
> Agent demo. It is NOT a real published paper: it has no authors, no venue,
> and no real experimental results. It exists only so the agent has papers
> to summarize, compare, and build a literature review from.

## Problem

Self-critique is widely used in agents: the model reviews its own draft
and revises it. But a model reviewing its own work may miss the same
mistakes it just made, or introduce new ones while fixing old ones. The
question is how much self-critique actually helps, and when it hurts.

## Method

This note tests an agent with and without a self-critique step on the
same set of research-summary tasks. Drafts are scored for correctness,
citation precision, and faithfulness to the source documents. The
critique step is also run with a different, smaller model to see whether
the critic needs to be as strong as the drafter.

## Findings

Self-critique provides limited benefit on tasks the model already answers
well: it fixes a few missing citations but rarely changes a wrong answer
into a right one. In some cases the revision step introduces new errors,
rewriting a correct sentence into an incorrect one. Using a smaller model
as the critic makes the benefit smaller still. Self-critique is most
useful as a formatting and citation check, not as a correctness check.

## Limitations

Only research-summary tasks were tested, so the findings may not apply
to coding or math tasks. The study used one model family, and no human
evaluation was done to confirm the automatic scores. Multilingual
documents were not tested.

## Key claims

- Self-critique provides limited benefit on tasks the model already answers well.
- Revision can introduce new errors.
- Self-critique is most useful as a citation check.
- Smaller critic models help less.
