# Shared LLM helpers — same provider rule as Products 1 and 2.
#
#   mock   — no key set. Deterministic, labelled mock: the agent uses
#            extractive / rule-based logic instead of an LLM. Runs offline.
#   openai — OPENAI_API_KEY set: real LLM.
#   azure  — AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT set: Azure OpenAI.
# Never hard-code a key. See .env.example.

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def detect_mode() -> str:
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def _llm_client() -> tuple[Any, str]:
    mode = detect_mode()
    if mode == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        return client, os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    from openai import OpenAI

    return OpenAI(), os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def llm_text(prompt: str, temperature: float = 0.2) -> str:
    """One chat completion; raises in mock mode (callers fall back)."""
    client, model = _llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def llm_json(prompt: str) -> dict:
    """Ask the LLM for JSON only and parse it; raises on bad JSON."""
    text = llm_text(prompt + "\nReply with JSON only, no markdown fences.").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)
