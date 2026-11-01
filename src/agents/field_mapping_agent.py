"""Agent 3: Field-Mapping Agent.

Wraps the configured fine-tuned field-mapping endpoint (the paper's flagship is
DeepSeek-Coder-33B on the expanded v7 corpus). Given a raw application
-layer byte stream + 5-tuple context, returns a field schema:
    [{off, len, type, role, name, value}, ...]

Applies the same gap-fill post-processing used in evaluation so the returned
fields contiguously tile the frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMConfig, Qwen3Client, _extract_json  # noqa: E402
from src.eval.postprocess import tile_fields  # noqa: E402


INSTRUCTION = (
    "You are a protocol field-mapping agent. Given the application-layer byte "
    "stream of one protocol frame (in hex) plus the transport context, output a "
    "STRICT JSON object {\"fields\":[...]} where each field has keys: off (byte "
    "offset, int), len (byte length, int), type (one of uint8|uint16_be|uint16_le|"
    "uint32_be|bytes), role (one of constant|length|sequence|opcode|address|count|"
    "payload|checksum), name (snake_case short field name), value (decoded int "
    "for scalar types; hex string for bytes). Fields MUST be contiguous and "
    "cover the entire frame. Do not include explanations."
)


def _format_hex(payload: bytes) -> str:
    lines = []
    for i in range(0, len(payload), 16):
        lines.append(" ".join(f"{b:02x}" for b in payload[i : i + 16]))
    return "\n".join(lines)


class FieldMappingAgent:
    def __init__(self, cfg: LLMConfig, tile: bool = True):
        self.client = Qwen3Client(cfg)
        self.cfg = cfg
        self.tile = tile

    def _build_input(self, payload: bytes, proto, sport, dport, direction) -> str:
        return (
            f"Protocol context: transport={proto}, src_port={sport}, "
            f"dst_port={dport}, direction={direction}\n"
            f"Frame length: {len(payload)} bytes\n"
            f"Bytes (hex):\n{_format_hex(payload)}"
        )

    def map_fields(self, payload: bytes, proto="tcp", sport=0, dport=0,
                   direction="c2s") -> list[dict]:
        """Return field schema for one frame. Empty list on failure."""
        msgs = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": self._build_input(payload, proto, sport, dport, direction)},
        ]
        try:
            reply = self.client.chat(msgs, max_tokens=self.cfg.max_tokens)
            obj = _extract_json(reply)
            fields = obj.get("fields", [])
        except Exception:
            fields = []
        if self.tile and fields:
            fields = tile_fields(fields, len(payload))
        return fields

    @staticmethod
    def opcode_of(fields: list[dict]):
        """Extract the opcode value from a field list (first 'opcode' role)."""
        for f in fields:
            if f.get("role") == "opcode":
                v = f.get("value")
                if isinstance(v, int):
                    return v
                if isinstance(v, str):
                    try:
                        return int(v, 0)
                    except ValueError:
                        return None
        return None
