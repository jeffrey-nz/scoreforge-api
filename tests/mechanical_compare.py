"""Shared logic for the mechanical-transcription regression test.

Compares the FULLY MECHANICAL pipeline (oemer OMR -> omer_import -> fit_to_meter,
no AI, no hand-editing) against the hand-verified bars of the original Fur Elise
job. Each bar is a musical comparison: per voice, the sequence of (pitch-set,
duration) events must match. Grace notes are ignored (non-metrical ornaments the
mechanical OMR doesn't produce).
"""
from __future__ import annotations
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, 'fixtures', 'fur_elise_gold.json')
MXML_DIR = os.path.join(HERE, 'fixtures', 'omr_out')

_TAG = {'w': 64, 'h': 32, 'q': 16, '8': 8, '16': 4, '32': 2, '64': 1}  # ticks @ div16
_STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def _midi(name):
    m = re.match(r'^([A-Ga-g])([#b]?)(-?\d+)$', name)
    if not m:
        return None
    pc = _STEP[m.group(1).upper()] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)
    return (int(m.group(3)) + 1) * 12 + pc


def parse_voice(s):
    """compact voice string -> [(frozenset(midi), ticks)]; rests = empty set;
    grace ('g') tokens skipped."""
    out = []
    for tok in str(s or '').strip().split():
        m = re.match(r'^(.+)\(([whq0-9]+|g)(\.?)\)$', tok)
        if not m:
            continue
        head, tag, dot = m.group(1), m.group(2), m.group(3)
        if tag == 'g':
            continue
        ticks = _TAG.get(tag)
        if ticks is None:
            continue
        if dot == '.':
            ticks = ticks * 3 // 2
        if re.match(r'^[Rr]$', head):
            out.append((frozenset(), ticks))
        else:
            ms = frozenset(x for x in (_midi(p) for p in head.split('+')) if x is not None)
            out.append((ms, ticks))
    return out


def bar_matches(gold, mech):
    """Strict: pitch-set AND duration sequences match in both voices."""
    if mech is None:
        return False
    return (parse_voice(gold.get('melody')) == parse_voice(mech.get('melody')) and
            parse_voice(gold.get('bass')) == parse_voice(mech.get('bass')))


def pitch_seq(s):
    """Sounding pitches in order (drop rests + durations) — an OMR pitch-accuracy
    metric independent of the (unreliable) mechanical rhythm."""
    return [ms for ms, _t in parse_voice(s) if ms]


def bar_pitch_matches(gold, mech):
    if mech is None:
        return False
    return (pitch_seq(gold.get('melody')) == pitch_seq(mech.get('melody')) and
            pitch_seq(gold.get('bass')) == pitch_seq(mech.get('bass')))


def onset_seq(s):
    """[(pitch-set, onset-tick)] for sounding notes — rests advance the cursor but
    aren't events. Ignores held-duration and how trailing rests are split, so it
    measures the audible question: do the notes land on the right beats?"""
    seq, cur = [], 0
    for ms, ticks in parse_voice(s):
        if ms:
            seq.append((ms, cur))
        cur += ticks
    return seq


def bar_onset_matches(gold, mech):
    if mech is None:
        return False
    return (onset_seq(gold.get('melody')) == onset_seq(mech.get('melody')) and
            onset_seq(gold.get('bass')) == onset_seq(mech.get('bass')))


def load_gold():
    with open(GOLD, encoding='utf-8') as f:
        return json.load(f)


def find_musicxml():
    if not os.path.isdir(MXML_DIR):
        return None
    for f in os.listdir(MXML_DIR):
        if f.lower().endswith(('.musicxml', '.xml')):
            return os.path.join(MXML_DIR, f)
    return None


def mechanical_bars(timesig='3/8'):
    """Run the mechanical conversion on the cached oemer MusicXML: returns
    [{melody, bass}] per measure, fit to the meter. None if no MusicXML yet."""
    mxml = find_musicxml()
    if not mxml:
        return None
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from app.core import omer_import
    raw = omer_import.musicxml_to_bars(mxml)
    mq = omer_import.meter_quarters(timesig)
    return [{'melody': omer_import.fit_to_meter(b.get('melody', ''), mq, slack='absorb'),
             'bass':   omer_import.fit_to_meter(b.get('bass', ''), mq, slack='trail')} for b in raw]


def mech_for_bar(mech_bars, n):
    """Gold bar number n -> the mechanical measure at index n-1 (page-1 1:1 map)."""
    if mech_bars is None:
        return None
    i = n - 1
    return mech_bars[i] if 0 <= i < len(mech_bars) else None
