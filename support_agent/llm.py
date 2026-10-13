"""The model behind the agent, kept behind a one-method interface.

The graph only ever calls `complete_json(system, user) -> dict`. That makes the
model swappable (NVIDIA, OpenAI, Bedrock, Gemini) and lets the tests use a
scripted fake with no network calls.
"""

import json
import os
import re
from typing import Protocol

from openai import OpenAI


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


def _first_json_object(text: str) -> dict:
    """Pull the first {...} block out of a model reply and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"model returned no JSON: {text[:200]!r}")
    return json.loads(match.group(0))


class NvidiaLLM:
    """OpenAI-compatible client pointed at the NVIDIA API catalog."""

    def __init__(self, model: str = "nvidia/nemotron-3-super-120b-a12b"):
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("Set NVIDIA_API_KEY (it is in ~/.zprofile)")
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        self.model = model

    def complete_json(self, system: str, user: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1500,  # room for the model's reasoning plus the answer
        )
        msg = resp.choices[0].message
        # Reasoning models may leave `content` empty and put text elsewhere.
        text = msg.content or getattr(msg, "reasoning_content", "") or ""
        return _first_json_object(text)
