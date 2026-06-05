"""Convert an OMR MusicXML (from oemer) into our compact bar note-strings.

oemer reads a score image and emits MusicXML; music21 parses it; we flatten each
measure's treble/bass voices into the `Pitch(dur)` tokens the player understands.
This is the mechanical drafting half — a human/Claude then verifies against the
source crop and fixes whatever the recogniser got wrong.
"""
from __future__ import annotations
from typing import List, Dict

# quarterLength -> our duration tag (no dot); dotted handled separately
_BASE = [(4.0, 'w'), (2.0, 'h'), (1.0, 'q'), (0.5, '8'), (0.25, '16'),
         (0.125, '32'), (0.0625, '64')]


def _dur_tag(ql: float) -> str:
    """Nearest compact duration tag for a quarter-length, with a dot if it is
    ~1.5x a base value."""
    best = None
    for base, tag in _BASE:
        for dotted, suffix in ((1.0, ''), (1.5, '.')):
            val = base * dotted
            err = abs(ql - val) / val
            if best is None or err < best[0]:
                best = (err, tag + suffix)
    return best[1]


def _name(p) -> str:
    """music21 pitch -> our 'E5' / 'Eb5' / 'E#5'."""
    return p.nameWithOctave.replace('-', 'b')


def musicxml_to_bars(path: str) -> List[Dict]:
    """Return [{'melody','bass'}] per measure (treble part -> melody, bass part
    -> bass). Chords reduce to their top note in the treble, bottom in the bass.
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
            tag = _dur_tag(float(el.quarterLength) or 0.25)
            if el.isRest:
                toks.append(f'R({tag})')
            elif el.isChord:
                ps = sorted(el.pitches, key=lambda p: p.midi)
                p = ps[-1] if pick_top else ps[0]
                toks.append(f'{_name(p)}({tag})')
            else:
                toks.append(f'{_name(el.pitch)}({tag})')
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
