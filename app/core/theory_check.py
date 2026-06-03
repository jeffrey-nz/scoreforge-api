"""
theory_check.py - Mechanical (rule-based, no AI) music-theory validation for
showcase pieces. Reads a piece's MIDI tracks and checks them against basic
theory expectations, producing a pass/warn/fail verdict plus a list of issues.

The result is written to showcase-midi/<id>/theory.json so the dashboard can
show an at-a-glance validity badge on each card.

Checks performed:
  - key conformance : share of pitched notes that fall inside the piece's key
  - bar structure   : does the note content roughly fit the declared bar count
  - pitch range     : notes stranded outside a sane instrument range
  - empty content   : tracks / pieces with no notes

Usage:
    python theory_check.py <piece_id> [--key "A minor"] [--time-sig 3/8] [--bars N]
    python theory_check.py --all       # validate every piece in showcase-midi/
    python theory_check.py <piece_id> --json   # print result, don't write file
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DASHBOARD     = Path(__file__).parent
SHOWCASE_MIDI = Path(os.environ.get('MIDI_OUTPUT_DIR', str(DASHBOARD / 'showcase-midi')))

PITCH_CLASS = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5,
    'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10,
    'B': 11,
}

# Scale formulas: semitone offsets from the tonic, keyed by a normalised
# (lower-cased, spaces stripped) mode name. `minor` includes the raised 7th
# (harmonic-minor leading tone), which appears in almost all minor-key music
# and would otherwise produce false "out of key" hits.
SCALE_DEGREES = {
    'major':            [0, 2, 4, 5, 7, 9, 11],
    'ionian':           [0, 2, 4, 5, 7, 9, 11],
    'minor':            [0, 2, 3, 5, 7, 8, 10, 11],
    'naturalminor':     [0, 2, 3, 5, 7, 8, 10],
    'aeolian':          [0, 2, 3, 5, 7, 8, 10],
    'dorian':           [0, 2, 3, 5, 7, 9, 10],
    'phrygian':         [0, 1, 3, 5, 7, 8, 10],
    'lydian':           [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':       [0, 2, 4, 5, 7, 9, 10],
    'locrian':          [0, 1, 3, 5, 6, 8, 10],
    'harmonicminor':    [0, 2, 3, 5, 7, 8, 11],
    'melodicminor':     [0, 2, 3, 5, 7, 9, 11],
    'hijaz':            [0, 1, 4, 5, 7, 8, 10],   # maqam Hijaz / phrygian dominant
    'phrygiandominant': [0, 1, 4, 5, 7, 8, 10],
    'majorpentatonic':  [0, 2, 4, 7, 9],
    'minorpentatonic':  [0, 3, 5, 7, 10],
    'blues':            [0, 3, 5, 6, 7, 10],
}


_MODE_ALIASES = {'': 'major', 'maj': 'major', 'min': 'minor', 'm': 'minor'}


def parse_key(key_str):
    """Parse 'A minor', 'C Major', 'D Dorian', 'A Hijaz' -> (root_pc, mode).

    `mode` is a normalised key into SCALE_DEGREES, or the raw (unknown) name."""
    if not key_str:
        return None, None
    m = re.match(r'^([A-Ga-g])([#b]?)\s*(.*)$', key_str.strip())
    if not m:
        return None, None
    root = PITCH_CLASS.get((m.group(1) + m.group(2)).upper())
    if root is None:
        return None, None
    rest = re.sub(r'\s+', '', m.group(3).strip().lower())
    mode = _MODE_ALIASES.get(rest, rest)
    return root, mode


def key_pitch_classes(root_pc, mode):
    """Allowed pitch classes for a key, or None if the mode is unknown."""
    degs = SCALE_DEGREES.get(mode)
    if degs is None:
        return None
    return {(root_pc + d) % 12 for d in degs}


_HARDCODED_META = None


def _piece_meta(piece_id, piece_dir):
    """Resolve a piece's key/timeSig/bars. Imported pieces carry catalog.json;
    built-in pieces live in showcase_compositions.CATALOG."""
    cat_path = piece_dir / 'catalog.json'
    if cat_path.exists():
        try:
            return json.loads(cat_path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    global _HARDCODED_META
    if _HARDCODED_META is None:
        try:
            if str(DASHBOARD) not in sys.path:
                sys.path.insert(0, str(DASHBOARD))
            from showcase_compositions import CATALOG
            _HARDCODED_META = CATALOG
        except Exception:
            _HARDCODED_META = {}
    return _HARDCODED_META.get(piece_id, {})


def read_midi_notes(path):
    """Return a list of {midi, start_qL, end_qL} for one MIDI file."""
    import mido
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat or 480
    notes = []
    for tr in mid.tracks:
        abs_t = 0
        ons = {}
        for msg in tr:
            abs_t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                ons[msg.note] = abs_t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                s = ons.pop(msg.note, None)
                if s is None:
                    continue
                notes.append({'midi': msg.note,
                              'start_qL': s / tpb,
                              'end_qL':   abs_t / tpb})
    return notes


def validate_piece(piece_id, key=None, time_sig=None, bars=None):
    """Run mechanical theory checks on a piece. Returns a result dict."""
    piece_dir = SHOWCASE_MIDI / piece_id
    if not piece_dir.is_dir():
        raise FileNotFoundError(f'no piece directory: {piece_dir}')

    # Pull key / timeSig / bars from catalog.json (imported pieces) or the
    # hardcoded CATALOG in showcase_compositions.py (built-in pieces).
    cat = _piece_meta(piece_id, piece_dir)
    key      = key      or cat.get('key')
    time_sig = time_sig or cat.get('timeSig', '4/4')
    bars     = bars     or cat.get('bars')

    issues = []   # each: {severity: 'error'|'warn', check, message}

    # ── Gather pitched-track notes (melody, bass, pad — exclude drums) ───────
    pitched = {}
    for track in ('melody', 'bass', 'pad'):
        fp = piece_dir / f'{track}.mid'
        if fp.exists():
            try:
                pitched[track] = read_midi_notes(fp)
            except Exception as e:
                issues.append({'severity': 'warn', 'check': 'read',
                               'message': f'{track}.mid unreadable: {e}'})

    all_notes = [n for ns in pitched.values() for n in ns]
    total = len(all_notes)

    if total == 0:
        return _finalise(piece_id, key, time_sig, bars, 0, [
            {'severity': 'error', 'check': 'empty',
             'message': 'no notes in any pitched track'}])

    # ── Check 1: key conformance ─────────────────────────────────────────────
    root_pc, mode = parse_key(key)
    in_key_ratio = None
    allowed = key_pitch_classes(root_pc, mode) if root_pc is not None else None
    if allowed is None:
        # Missing or exotic key/mode — can't check conformance, don't penalise.
        pass
    else:
        in_key = sum(1 for n in all_notes if (n['midi'] % 12) in allowed)
        in_key_ratio = in_key / total
        pct = round(in_key_ratio * 100)
        if in_key_ratio < 0.80:
            issues.append({'severity': 'error', 'check': 'key',
                           'message': f'only {pct}% of notes are in {key} '
                                      f'({total - in_key} out-of-key notes)'})
        elif in_key_ratio < 0.92:
            issues.append({'severity': 'warn', 'check': 'key',
                           'message': f'{pct}% of notes in {key} — some '
                                      f'chromatic/out-of-key content'})

    # ── Check 2: bar structure ───────────────────────────────────────────────
    try:
        num, den = map(int, str(time_sig).split('/'))
        qL_per_bar = num * 4 / den
    except Exception:
        qL_per_bar = 4.0
    end_qL = max((n['end_qL'] for n in all_notes), default=0)
    played_bars = end_qL / qL_per_bar if qL_per_bar else 0
    if bars and bars > 0:
        ratio = played_bars / bars
        if ratio > 1.15:
            issues.append({'severity': 'warn', 'check': 'bars',
                           'message': f'content runs ~{played_bars:.0f} bars but '
                                      f'catalog says {bars} — likely OMR over-read'})
        elif ratio < 0.6:
            issues.append({'severity': 'warn', 'check': 'bars',
                           'message': f'content is only ~{played_bars:.0f} bars '
                                      f'of the declared {bars}'})

    # ── Check 3: pitch range sanity ──────────────────────────────────────────
    stray = [n for n in all_notes if n['midi'] < 21 or n['midi'] > 108]
    if stray:
        issues.append({'severity': 'warn', 'check': 'range',
                       'message': f'{len(stray)} note(s) outside the playable '
                                  f'piano range (A0–C8)'})

    # ── Check 4: track presence ──────────────────────────────────────────────
    if 'melody' in pitched and len(pitched['melody']) == 0:
        issues.append({'severity': 'warn', 'check': 'empty',
                       'message': 'melody track has no notes'})

    # ── Score ────────────────────────────────────────────────────────────────
    score = 100
    for it in issues:
        score -= 22 if it['severity'] == 'error' else 9
    score = max(0, score)

    return _finalise(piece_id, key, time_sig, bars, total, issues, score,
                     in_key_ratio)


def _finalise(piece_id, key, time_sig, bars, note_count, issues, score=None,
              in_key_ratio=None):
    if score is None:
        score = 0 if any(i['severity'] == 'error' for i in issues) else 100
    has_error = any(i['severity'] == 'error' for i in issues)
    if has_error or score < 60:
        status = 'invalid'
    elif issues:
        status = 'warn'
    else:
        status = 'valid'
    return {
        'id': piece_id,
        'status': status,
        'score': score,
        'noteCount': note_count,
        'key': key,
        'timeSig': time_sig,
        'inKeyPct': round(in_key_ratio * 100) if in_key_ratio is not None else None,
        'issues': issues,
    }


def write_result(piece_id, result):
    out = SHOWCASE_MIDI / piece_id / 'theory.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return out


def main():
    p = argparse.ArgumentParser(description='Mechanical music-theory validation')
    p.add_argument('piece_id', nargs='?', help='piece id in showcase-midi/')
    p.add_argument('--key', help='key override, e.g. "A minor"')
    p.add_argument('--time-sig', dest='time_sig', help='time signature, e.g. 3/8')
    p.add_argument('--bars', type=int, help='declared bar count')
    p.add_argument('--all', action='store_true', help='validate every piece')
    p.add_argument('--json', action='store_true',
                   help='print result JSON, do not write theory.json')
    args = p.parse_args()

    if args.all:
        ids = sorted(d.name for d in SHOWCASE_MIDI.iterdir() if d.is_dir())
        ok = warn = bad = 0
        for pid in ids:
            try:
                result = validate_piece(pid)
            except Exception as e:
                print(f'  {pid:28s} ERROR  {e}', file=sys.stderr)
                continue
            write_result(pid, result)
            status = result['status']
            ok   += status == 'valid'
            warn += status == 'warn'
            bad  += status == 'invalid'
            mark = {'valid': 'OK  ', 'warn': 'WARN', 'invalid': 'FAIL'}[status]
            print(f'  {pid:28s} {mark} score={result["score"]:3d}  '
                  f'{len(result["issues"])} issue(s)')
        print(f'\n[theory] {len(ids)} pieces: {ok} valid, {warn} warn, {bad} invalid')
        return 0

    if not args.piece_id:
        print('error: provide a piece_id or --all', file=sys.stderr)
        return 2

    try:
        result = validate_piece(args.piece_id, key=args.key,
                                time_sig=args.time_sig, bars=args.bars)
    except Exception as e:
        print(f'[theory] {e}', file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        out = write_result(args.piece_id, result)
        print(f'[theory] {args.piece_id}: {result["status"]} '
              f'(score {result["score"]}, {len(result["issues"])} issues) -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
