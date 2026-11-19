# SAMPLE RESEARCH NOTE — Retrieval quality in RAG systems

> This is a short, invented sample note written for the Teja Research Paper
> Agent demo. It is NOT a real published paper: it has no authors, no venue,
> and no real experimental results. It exists only so the agent has papers
> to summarize, compare, and build a literature review from.

## Problem

Retrieval-Augmented Generation (RAG) answers depend on the passages the
retriever returns. If retrieval returns the wrong passages, even a strong
language model gives a wrong or unsupported answer. The open question is
which retrieval choices matter most: chunk size, embedding model, or
re-ranking.

## Method

This note compares three retrieval settings on a small question-answering
collection built from public documentation: dense retrieval with a
small embedding model, keyword retrieval, and dense retrieval followed by
a re-ranker. Answer quality is judged by whether the answer is supported
by the retrieved passages, and whether it cites its sources.

## Findings

Larger chunks improve answer quality because each chunk carries more
context, so the language model has to guess less. Dense retrieval beats
keyword retrieval on paraphrased questions, while keyword retrieval is
strong on questions that repeat exact terms from the documents. Adding a
re-ranker on top of dense retrieval gives the best supported answers.
RAG answers should cite their sources so a reader can check them.

## Limitations

The collection is small and made of documentation, not research papers.
Only one embedding model was tested, and no human evaluation was done.
Latency and cost of the re-ranker were not measured.

## Key claims

- Larger chunks improve answer quality.
- Dense retrieval beats keyword retrieval on paraphrased questions.
- RAG answers should cite their sources.
- Re-ranking improves supported answers.
