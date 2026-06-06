"""Convert an OMR MusicXML (from oemer) into our compact bar note-strings.

oemer reads a score image and emits MusicXML; music21 parses it; we flatten each
measure's treble/bass voices into the `Pitch(dur)` tokens the player understands.
This is the mechanical drafting half — a human/Claude then verifies against the
source crop and fixes whatever the recogniser got wrong.
"""
from __future__ import annotations
from typing import List, Dict

# music21 duration.type -> our compact tag. We key off the *type* (note shape),
# not quarterLength: oemer emits no time signature, so music21 assumes 4/4 and
# mangles quarterLengths, but the engraved note shapes (16th/eighth/...) survive.
_TYPE_TAG = {'whole': 'w', 'half': 'h', 'quarter': 'q', 'eighth': '8',
             '16th': '16', '32nd': '32', '64th': '64'}
# Rests this long in a sub-quarter meter are oemer's 4/4 padding, not real rests.
_PAD_REST_QL = 1.0


def _dur_tag(el) -> str:
    """Compact duration tag from a music21 element's note *type* + dots."""
    tag = _TYPE_TAG.get(el.duration.type, '16')
    return tag + ('.' if getattr(el.duration, 'dots', 0) else '')


def _name(p) -> str:
    """music21 pitch -> our 'E5' / 'Eb5' / 'E#5'."""
    return p.nameWithOctave.replace('-', 'b')


_QL_SNAP = [(4.0, 'w'), (3.0, 'h.'), (2.0, 'h'), (1.5, 'q.'), (1.0, 'q'),
            (0.75, '8.'), (0.5, '8'), (0.375, '16.'), (0.25, '16'), (0.125, '32')]


def _ql_to_tag(ql: float) -> str:
    return min(_QL_SNAP, key=lambda bt: abs(bt[0] - ql))[1]


def meter_quarters(timesig: str) -> float:
    try:
        n, d = (int(x) for x in str(timesig).split('/'))
        return n * 4 / d
    except Exception:
        return 4.0


def fit_to_meter(token_str: str, meter_q: float) -> str:
    """Re-quantise a bar to its meter using oemer's (reliable) PITCH sequence but
    NOT its (unreliable) durations: drop rests, lay the notes as 16ths, and if
    they under-fill, lengthen the first (downbeat) note to absorb the slack — so
    runs come out as exact 16ths and held-note bars get a long downbeat + 16ths.
    A pragmatic playable default; exact inner rhythm is then a review fix."""
    pitches = [t.split('(')[0] for t in token_str.split() if not t[:2].upper().startswith('R')]
    n = len(pitches)
    if n == 0:
        return ''
    base = 0.25
    if n * base <= meter_q + 1e-6:
        durs = [base] * n
        durs[0] += meter_q - n * base          # held downbeat absorbs the slack
    else:
        durs = [meter_q / n] * n               # denser than 16ths: split evenly
    return ' '.join(f'{p}({_ql_to_tag(d)})' for p, d in zip(pitches, durs))


def musicxml_to_bars(path: str, drop_pad_rests: bool = True) -> List[Dict]:
    """Return [{'melody','bass'}] per measure (treble part -> melody, bass part
    -> bass). Chords are kept in full as a '+'-joined token (low to high), e.g.
    A4+C5+E5(q). Padding rests (a quarter or longer, from oemer's 4/4 assumption)
    are dropped so the recovered bars hold just the engraved notes for review
    against meter.
    """
    import music21 as m21
    score = m21.converter.parse(path)
    parts = list(score.parts)
    if not parts:
        return []
    treble = parts[0]
    bass = parts[1] if len(parts) > 1 else None

    def voice_tokens(measure, pick_top: bool) -> str:
        toks = []
        for el in measure.notesAndRests:
            if el.isRest:
                if drop_pad_rests and float(el.quarterLength) >= _PAD_REST_QL:
                    continue                       # skip 4/4-padding rest
                toks.append(f'R({_dur_tag(el)})')
            elif el.isChord:
                ps = sorted(el.pitches, key=lambda p: p.midi)   # low → high
                toks.append('+'.join(_name(p) for p in ps) + f'({_dur_tag(el)})')
            else:
                toks.append(f'{_name(el.pitch)}({_dur_tag(el)})')
        return ' '.join(toks)

    t_meas = list(treble.getElementsByClass('Measure'))
    b_meas = list(bass.getElementsByClass('Measure')) if bass else []
    out = []
    for i, tm in enumerate(t_meas):
        mel = voice_tokens(tm, pick_top=True)
        bas = voice_tokens(b_meas[i], pick_top=False) if i < len(b_meas) else ''
        out.append({'melody': mel, 'bass': bas})
    return out


if __name__ == '__main__':
    import sys, json
    print(json.dumps(musicxml_to_bars(sys.argv[1]), indent=1))
