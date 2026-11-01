"""Multi-hypothesis validation for length / checksum fields.

The original Correlation Agent assumed one rule: a length field's value equals
the number of bytes that follow it (the Modbus/MBAP case). That fails on:

  * TLV-nested length  — CIP / EtherNet/IP: a length governs only its enclosing
    TLV value (up to the next sibling length / a fixed encapsulation header),
    not the whole remaining frame.
  * Segmented CRC/length — DNP3 data-link: the "Length" counts header + user
    data but NOT the per-block CRCs, and the transmitted user data is chunked
    into 16-byte blocks each followed by a 2-byte CRC, so declared != bytes_after.

Instead of one hard-coded rule we enumerate a bank of plausible length
interpretations (generic + a few protocol-informed) and accept the field if ANY
holds, recording which one. We also:
  * check TLV nesting consistency (each inner length region must fit inside its
    enclosing length region), and
  * try to VERIFY checksum-role fields against a small bank of CRC-16 models over
    several candidate spans (handles segmented / trailing CRC).

Pure stdlib. Importable by both the Correlation Agent and the eval harness.
"""
from __future__ import annotations

from math import ceil


def _as_int(v):
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v, 0)
        except ValueError:
            return None
    return None


def field_int_from_raw(field: dict, raw: bytes):
    """Decode an integer operand from captured bytes, never from LLM text.

    The mapper is allowed to locate and type a field, but its generated
    ``value`` must not become an authority for a deterministic relation.  If
    the recovered span/type cannot be decoded, callers may reject the relation
    or explicitly fall back for legacy analyses.
    """
    off, ln = field.get("off"), field.get("len")
    if not isinstance(off, int) or not isinstance(ln, int) or ln <= 0:
        return None
    if off < 0 or off + ln > len(raw):
        return None
    ty = str(field.get("type", "")).lower()
    if not (ty.startswith("uint") or ty.startswith("int")):
        return None
    endian = "little" if ty.endswith("_le") else "big"
    return int.from_bytes(raw[off:off + ln], endian, signed=ty.startswith("int"))


def field_uint_candidates_from_raw(field: dict, raw: bytes) -> list[int]:
    """Return the finite byte-order candidates for an unsigned operand."""
    off, ln = field.get("off"), field.get("len")
    if not isinstance(off, int) or not isinstance(ln, int) or ln <= 0:
        return []
    if off < 0 or off + ln > len(raw):
        return []
    chunk = raw[off:off + ln]
    values = [int.from_bytes(chunk, "big")]
    if ln > 1:
        values.append(int.from_bytes(chunk, "little"))
    return list(dict.fromkeys(values))


# --------------------------------------------------------------------------- #
# length interpretations
# --------------------------------------------------------------------------- #
def _expected_transmitted(data_len: int, block: int = 16, crc: int = 2) -> int:
    """Bytes on the wire to carry `data_len` data bytes with a `crc`-byte CRC
    appended after every `block` data bytes (DNP3-style segmentation)."""
    if data_len <= 0:
        return max(0, data_len)
    return data_len + crc * ceil(data_len / block)


def length_hypotheses(lf: dict, fields: list[dict], frame_len: int,
                      raw: bytes | None = None) -> dict:
    """Return {hypothesis_name: bool} for one length field `lf`."""
    declared_values = (field_uint_candidates_from_raw(lf, raw)
                       if raw is not None else [_as_int(lf.get("value"))])
    declared_values = [value for value in declared_values if value is not None]
    off, ln = lf.get("off", 0), lf.get("len", 0)
    if not declared_values:
        return {}
    after = frame_len - (off + ln)          # bytes following the length field
    from_field = frame_len - off            # bytes from the field's own start

    # next length field after this one (for TLV value scoping)
    next_len_off = None
    for f in fields:
        if f.get("role") == "length" and f.get("off", 0) > off:
            o = f["off"]
            next_len_off = o if next_len_off is None else min(next_len_off, o)

    H = {
        "bytes_after":   any(declared == after for declared in declared_values),
        "incl_self":     any(declared == after + ln for declared in declared_values),
        "from_field":    any(declared == from_field for declared in declared_values),
        "whole_frame":   any(declared == frame_len for declared in declared_values),
        "words_after":   any(declared * 2 == after for declared in declared_values),
        "minus_crc2":    any(declared == after - 2 for declared in declared_values),
        "minus_crc4":    any(declared == after - 4 for declared in declared_values),
        # DNP3-style: declared = data bytes (no CRCs); wire bytes have CRC/16B
        "crc_segmented": any(after == _expected_transmitted(declared)
                             for declared in declared_values),
        # CIP/EtherNet-IP encapsulation: data length after the 24-byte ENIP header
        "enip_encap":    any(declared == frame_len - 24 for declared in declared_values),
    }
    # DNP3 data-link: Length L = 5 (ctrl+dest+src) + app_data; after the 1-byte
    # length field come those 5 bytes + their 2-byte CRC + segmented app blocks.
    H["dnp3_link"] = any(
        declared >= 5 and after == 7 + _expected_transmitted(declared - 5)
        for declared in declared_values
    )
    # TLV value: length governs the bytes up to the next sibling length field.
    if next_len_off is not None:
        H["tlv_to_next"] = any(
            declared == next_len_off - (off + ln) for declared in declared_values
        )
    return H


def validate_length_field(lf: dict, fields: list[dict], frame_len: int,
                          raw: bytes | None = None):
    """Return (consistent: bool, hypothesis: str|None) — consistent if ANY
    interpretation in the bank holds."""
    H = length_hypotheses(lf, fields, frame_len, raw)
    for name, ok in H.items():
        if ok:
            return True, name
    return False, None


def check_tlv_nesting(fields: list[dict], frame_len: int,
                      raw: bytes | None = None) -> dict:
    """TLV nesting consistency: resolve each length field's governed region and
    verify regions are properly nested (no partial overlap). Returns
    {n_length, n_nested_ok, consistent}."""
    lens = sorted([f for f in fields if f.get("role") == "length"],
                  key=lambda f: f.get("off", 0))
    regions = []
    for lf in lens:
        ok, hyp = validate_length_field(lf, fields, frame_len, raw)
        candidates = (field_uint_candidates_from_raw(lf, raw)
                      if raw is not None else [_as_int(lf.get("value"))])
        declared = next((value for value in candidates
                         if length_hypotheses({**lf, "value": value}, fields,
                                              frame_len).get(hyp, False)), None)
        off, ln = lf.get("off", 0), lf.get("len", 0)
        if declared is None:
            continue
        # governed region [start, end)
        start = off + ln
        if hyp in ("bytes_after", "incl_self", "crc_segmented", "dnp3_link", None):
            end = frame_len
        elif hyp == "tlv_to_next":
            end = start + declared
        elif hyp == "from_field":
            start, end = off, frame_len
        elif hyp == "whole_frame":
            start, end = 0, frame_len
        else:
            end = min(frame_len, start + max(declared, 0))
        regions.append((start, end))

    def nested(a, b):
        # a inside b, or disjoint, or b inside a — but not partial overlap
        (a0, a1), (b0, b1) = a, b
        return (a0 >= b0 and a1 <= b1) or (b0 >= a0 and b1 <= a1) or a1 <= b0 or b1 <= a0

    n_ok = 0
    pairs = 0
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            pairs += 1
            if nested(regions[i], regions[j]):
                n_ok += 1
    return {"n_length": len(lens), "n_region_pairs": pairs,
            "n_nested_ok": n_ok, "consistent": (pairs == 0 or n_ok == pairs)}


# --------------------------------------------------------------------------- #
# checksum / CRC verification
# --------------------------------------------------------------------------- #
def _reflect(x: int, width: int) -> int:
    r = 0
    for _ in range(width):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def crc16(data: bytes, poly: int, init: int, refin: bool, refout: bool, xorout: int) -> int:
    crc = init
    for b in data:
        if refin:
            b = _reflect(b, 8)
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    if refout:
        crc = _reflect(crc, 16)
    return crc ^ xorout


CRC16_MODELS = {
    "modbus":     dict(poly=0x8005, init=0xFFFF, refin=True,  refout=True,  xorout=0x0000),
    "dnp3":       dict(poly=0x3D65, init=0x0000, refin=True,  refout=True,  xorout=0xFFFF),
    "ccitt_false":dict(poly=0x1021, init=0xFFFF, refin=False, refout=False, xorout=0x0000),
    "kermit":     dict(poly=0x1021, init=0x0000, refin=True,  refout=True,  xorout=0x0000),
    "x25":        dict(poly=0x1021, init=0xFFFF, refin=True,  refout=True,  xorout=0xFFFF),
    "arc":        dict(poly=0x8005, init=0x0000, refin=True,  refout=True,  xorout=0x0000),
}


def verify_checksum_field(cf: dict, raw: bytes, fields: list[dict]):
    """Try to validate a checksum-role field against the CRC-16 bank over a few
    candidate spans. Return (verified: bool, model: str|None, span: str|None)."""
    off, ln = cf.get("off", 0), cf.get("len", 0)
    if ln != 2 or off + ln > len(raw):
        return False, None, None
    declared_be = int.from_bytes(raw[off:off + ln], "big")
    declared_le = int.from_bytes(raw[off:off + ln], "little")
    # candidate spans the CRC could cover (start offsets), all ending at `off`
    starts = {0}
    for f in fields:
        if f.get("off", 0) < off:
            starts.add(f.get("off", 0))
    for start in sorted(starts):
        span = raw[start:off]
        if not span:
            continue
        for name, m in CRC16_MODELS.items():
            c = crc16(span, **m)
            if c in (declared_be, declared_le):
                return True, name, f"[{start}:{off}]"
    return False, None, None
