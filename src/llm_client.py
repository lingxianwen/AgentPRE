"""OpenAI-compatible client used by configured field-mapping/semantic endpoints.

Matches the FastAPI endpoint shape of Qwen3/api_server_qwen3.py:
  POST /v1/chat/completions
  body: {messages, max_tokens, stream, enable_thinking}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable

import requests
import yaml


@dataclass
class LLMConfig:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: int
    enable_thinking: bool

    @classmethod
    def from_yaml(cls, path: str) -> "LLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["llm"]
        return cls(**cfg)


class Qwen3Client:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"

    def chat(self, messages: list[dict], **overrides) -> str:
        payload = {
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", self.cfg.max_tokens),
            "stream": False,
            "enable_thinking": overrides.get("enable_thinking", self.cfg.enable_thinking),
        }
        r = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict], retries: int = 2, **overrides) -> dict:
        """Call chat and parse a JSON object out of the reply (tolerates code fences)."""
        last_err = None
        for _ in range(retries + 1):
            text = self.chat(messages, **overrides)
            try:
                return _extract_json(text)
            except ValueError as e:
                last_err = e
                time.sleep(0.5)
        raise RuntimeError(f"LLM did not return valid JSON: {last_err}")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # strip code fence
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    # find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/config.yaml"
    cfg = LLMConfig.from_yaml(cfg_path)
    c = Qwen3Client(cfg)
    out = c.chat([{"role": "user", "content": "回复一句话：模型已连接。"}], max_tokens=64)
    print(out)
