"""SQLite-backed Message / Field / Session three-layer graph store.

This is the structured intermediate medium that the four agents exchange
information through (replacing LIVA's Neo4j; SQLite is enough for PRE-scale
graphs and needs no external service).

Schema:
    sessions(session_id, src_ip, dst_ip, sport, dport, proto, protocol)
    messages(msg_id, session_id, pkt_idx, direction, ts, payload_hex,
             frame_len, opcode)
    fields(field_id, msg_id, off, len, type, role, name, value)
    correlations(corr_id, kind, msg_a, msg_b, detail)   -- filled by Agent 4
    states(state_id, session_id, label, detail)          -- filled by Agent 5
    transitions(trans_id, session_id, src_state, dst_state, condition)

A "session" is one 5-tuple flow direction-agnostic (client<->server pair).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field as dc_field
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   INTEGER PRIMARY KEY,
    src_ip       TEXT, dst_ip TEXT,
    sport        INTEGER, dport INTEGER,
    proto        TEXT,
    protocol     TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    msg_id       INTEGER PRIMARY KEY,
    session_id   INTEGER,
    pkt_idx      INTEGER,
    direction    TEXT,
    ts           REAL,
    payload_hex  TEXT,
    frame_len    INTEGER,
    opcode       INTEGER,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE TABLE IF NOT EXISTS fields (
    field_id     INTEGER PRIMARY KEY,
    msg_id       INTEGER,
    off          INTEGER, len INTEGER,
    type         TEXT, role TEXT, name TEXT, value TEXT,
    FOREIGN KEY(msg_id) REFERENCES messages(msg_id)
);
CREATE TABLE IF NOT EXISTS correlations (
    corr_id      INTEGER PRIMARY KEY,
    kind         TEXT,         -- 'req_resp' | 'length_coupling' | 'seq_increment'
    msg_a        INTEGER, msg_b INTEGER,
    detail       TEXT
);
CREATE TABLE IF NOT EXISTS states (
    state_id     INTEGER PRIMARY KEY,
    session_id   INTEGER,
    label        TEXT,
    detail       TEXT
);
CREATE TABLE IF NOT EXISTS transitions (
    trans_id     INTEGER PRIMARY KEY,
    session_id   INTEGER,
    src_state    TEXT, dst_state TEXT,
    condition    TEXT
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_field_msg ON fields(msg_id);
"""


@dataclass
class Message:
    msg_id: int
    session_id: int
    pkt_idx: int
    direction: str
    ts: float
    payload_hex: str
    frame_len: int
    opcode: Optional[int]
    fields: list[dict] = dc_field(default_factory=list)


class GraphStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---------------- write side ----------------
    def get_or_create_session(self, src_ip, dst_ip, sport, dport, proto, protocol) -> int:
        # Normalize so client<->server is one session regardless of direction:
        # order endpoints by (ip, port) so both directions map to same key.
        a = (src_ip, sport)
        b = (dst_ip, dport)
        lo, hi = sorted([a, b])
        cur = self.conn.execute(
            "SELECT session_id FROM sessions WHERE src_ip=? AND sport=? AND dst_ip=? AND dport=? AND proto=?",
            (lo[0], lo[1], hi[0], hi[1], proto),
        )
        row = cur.fetchone()
        if row:
            return row["session_id"]
        cur = self.conn.execute(
            "INSERT INTO sessions(src_ip,dst_ip,sport,dport,proto,protocol) VALUES (?,?,?,?,?,?)",
            (lo[0], hi[0], lo[1], hi[1], proto, protocol),
        )
        return cur.lastrowid

    def add_message(self, session_id, pkt_idx, direction, ts, payload_hex,
                    frame_len, opcode, fields: list[dict]) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(session_id,pkt_idx,direction,ts,payload_hex,frame_len,opcode) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_id, pkt_idx, direction, ts, payload_hex, frame_len, opcode),
        )
        msg_id = cur.lastrowid
        for f in fields:
            self.conn.execute(
                "INSERT INTO fields(msg_id,off,len,type,role,name,value) VALUES (?,?,?,?,?,?,?)",
                (msg_id, f.get("off"), f.get("len"), f.get("type"),
                 f.get("role"), f.get("name"), json.dumps(f.get("value"), ensure_ascii=False)),
            )
        return msg_id

    def add_correlation(self, kind, msg_a, msg_b, detail: dict):
        self.conn.execute(
            "INSERT INTO correlations(kind,msg_a,msg_b,detail) VALUES (?,?,?,?)",
            (kind, msg_a, msg_b, json.dumps(detail, ensure_ascii=False)),
        )

    def add_state(self, session_id, label, detail: dict):
        self.conn.execute(
            "INSERT INTO states(session_id,label,detail) VALUES (?,?,?)",
            (session_id, label, json.dumps(detail, ensure_ascii=False)),
        )

    def add_transition(self, session_id, src_state, dst_state, condition):
        self.conn.execute(
            "INSERT INTO transitions(session_id,src_state,dst_state,condition) VALUES (?,?,?,?)",
            (session_id, src_state, dst_state, condition),
        )

    def commit(self):
        self.conn.commit()

    # ---------------- read side ----------------
    def sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM sessions").fetchall()

    def messages_of(self, session_id, order_by="pkt_idx") -> list[Message]:
        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE session_id=? ORDER BY {order_by}", (session_id,)
        ).fetchall()
        out = []
        for r in rows:
            fields = self.fields_of(r["msg_id"])
            out.append(Message(
                msg_id=r["msg_id"], session_id=r["session_id"], pkt_idx=r["pkt_idx"],
                direction=r["direction"], ts=r["ts"], payload_hex=r["payload_hex"],
                frame_len=r["frame_len"], opcode=r["opcode"], fields=fields,
            ))
        return out

    def fields_of(self, msg_id) -> list[dict]:
        rows = self.conn.execute(
            "SELECT off,len,type,role,name,value FROM fields WHERE msg_id=? ORDER BY off", (msg_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["value"] = json.loads(d["value"]) if d["value"] is not None else None
            except (json.JSONDecodeError, TypeError):
                pass
            out.append(d)
        return out

    def field_value(self, msg_id, role) -> Optional[str]:
        """Return the value of the first field with the given role in a message."""
        for f in self.fields_of(msg_id):
            if f["role"] == role:
                return f["value"]
        return None

    def correlations(self, kind=None) -> list[sqlite3.Row]:
        if kind:
            return self.conn.execute("SELECT * FROM correlations WHERE kind=?", (kind,)).fetchall()
        return self.conn.execute("SELECT * FROM correlations").fetchall()

    def stats(self) -> dict:
        def n(t): return self.conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        return {t: n(t) for t in ("sessions", "messages", "fields",
                                  "correlations", "states", "transitions")}

    def close(self):
        self.conn.close()
