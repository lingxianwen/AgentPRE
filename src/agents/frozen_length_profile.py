"""Training-only length-convention calibration with explicit abstention.

Unlike per-frame existential matching, the rule and byte order are frozen before
evaluation. This optional policy does not change the historical pipeline default.
"""
from __future__ import annotations
from collections import defaultdict
from .length_validation import length_hypotheses


def slot(domain, field):
    return f"{domain}:{field.get('off')}:{field.get('len')}"


def candidates(field, fields, raw):
    off, size = field.get('off'), field.get('len')
    if not isinstance(off, int) or not isinstance(size, int) or off < 0 or size < 1 or off + size > len(raw):
        return set()
    matches = set()
    for endian in ('big', 'little') if size > 1 else ('big',):
        value = int.from_bytes(raw[off:off+size], endian)
        for name, yes in length_hypotheses({**field, 'value': value}, fields, len(raw)).items():
            # Same equation, not an independently identifiable convention.
            if yes and name != 'from_field':
                matches.add((name, endian))
    return matches


def calibrate(examples, min_messages=3, min_frame_lengths=2):
    """Examples are (domain, raw bytes, training fields), never test outcomes."""
    groups = defaultdict(dict)
    for domain, raw, fields in examples:
        for field in fields:
            if field.get('role') == 'length':
                groups[slot(domain, field)][raw] = candidates(field, fields, raw)
    entries = {}
    for key, messages in groups.items():
        shared = set.intersection(*messages.values())
        n_lengths = len({len(raw) for raw in messages})
        eligible = len(messages) >= min_messages and n_lengths >= min_frame_lengths
        identified = eligible and len(shared) == 1
        entries[key] = {
            'n_unique_messages': len(messages), 'n_frame_lengths': n_lengths,
            'candidates': [list(pair) for pair in sorted(shared)],
            'status': 'identified' if identified else 'abstain',
            'reason': 'unique_consistent_rule' if identified else
                      'insufficient_support' if not eligible else 'ambiguous_or_inconsistent',
        }
    return {'version': 1, 'selection': 'intersection on unique training bytes only',
            'min_messages': min_messages, 'min_frame_lengths': min_frame_lengths,
            'entries': entries}


def validate(profile, domain, field, fields, raw):
    entry = profile['entries'].get(slot(domain, field))
    if not entry or entry['status'] != 'identified':
        return False, None, 'abstain'
    rule, endian = entry['candidates'][0]
    accepted = (rule, endian) in candidates(field, fields, raw)
    return accepted, f'{rule}:{endian}', 'accepted' if accepted else 'rejected'
