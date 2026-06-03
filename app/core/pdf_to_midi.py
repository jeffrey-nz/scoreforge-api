#!/usr/bin/env python3
"""
pdf_to_midi.py — Convert sheet music (PDF / MusicXML / MXL / MIDI) into the
project's 4-track MIDI structure (bass.mid, pad.mid, melody.mid, drums.mid).

Usage:
  python pdf_to_midi.py <input_file> --id <piece_id> [options]

Options:
  --id      Piece ID (slug, e.g. "bach_bwv772"). Defaults to input filename stem.
  --title   Human-readable title. Defaults to piece ID.
  --out     Output base directory (default: <this_file_dir>/showcase-midi)
  --bpm     Override BPM detected from score (integer)
  --no-drums  Suppress auto-generated drum pattern (default: drums silent for classical)

Supported input formats:
  .pdf       AI-vision transcription — every page is read by an LLM
             (ai_transcribe.py). No mechanical OMR.
  .xml .musicxml  MusicXML
  .mxl       Compressed MusicXML
  .mid .midi MIDI (re-maps tracks -> 4-track structure)

Output (written to <out>/<id>/):
  bass.mid, pad.mid, melody.mid, drums.mid
  catalog.json   — catalog entry JSON (print to stdout as well)
  preview_p1.png — first-page render (PDF input only)
  _pages/        — per-page PNGs + manifest used by the AI verify pass

Exit codes:  0 = success, 1 = error, 2 = no note data found
"""
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUT = SCRIPT_DIR / 'showcase-midi'

# ── MIDI helpers ──────────────────────────────────────────────────────────────

def _sec_to_ticks(sec, bpm, tpb=480):
    return int(round(sec * tpb * bpm / 60.0))


def write_midi_track(note_events, bpm, time_sig_str, output_path,
                     channel=0, tpb=480):
    """Write a single MIDI file (type 1, 1 track) from a list of note dicts.

    Each dict: {note: int, start: float (s), end: float (s), vel: int}
    An empty note_events list produces a header-only MIDI file (silence).
    """
    import mido

    num_str, den_str = (time_sig_str + '/4').split('/')[:2]
    numerator   = int(num_str)
    denominator = int(den_str)
    us_per_beat = mido.bpm2tempo(bpm)

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)

    trk.append(mido.MetaMessage('set_tempo',    tempo=us_per_beat, time=0))
    trk.append(mido.MetaMessage('time_signature',
                                numerator=numerator, denominator=denominator,
                                clocks_per_click=24, notated_32nd_notes_per_beat=8,
                                time=0))

    if note_events:
        # Build absolute-tick event list
        events = []
        for ne in note_events:
            on_t  = _sec_to_ticks(ne['start'], bpm, tpb)
            off_t = _sec_to_ticks(ne['end'],   bpm, tpb)
            # Ensure at least 1 tick of sound
            if off_t <= on_t:
                off_t = on_t + 1
            vel = max(1, min(127, int(ne.get('vel', 64))))
            note = int(ne['note'])
            events.append((on_t,  0x90 | channel, note, vel))
            events.append((off_t, 0x80 | channel, note, 0))

        # Sort: ties broken note-off before note-on
        events.sort(key=lambda e: (e[0], 0 if (e[1] & 0xF0) == 0x80 else 1))

        last = 0
        for tick, status, d1, d2 in events:
            delta = tick - last
            trk.append(mido.Message.from_bytes([status, d1, d2], time=delta))
            last = tick

    trk.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(str(output_path))


# ── PDF handling ──────────────────────────────────────────────────────────────

def render_pdf_thumbnail(pdf_path, out_dir, max_width=900):
    """Render first page of PDF to preview_p1.png. Returns path or None."""
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if doc.page_count == 0:
            return None
        page  = doc[0]
        scale = max_width / page.rect.width
        mat   = fitz.Matrix(scale, scale)
        pix   = page.get_pixmap(matrix=mat, alpha=False)
        out   = Path(out_dir) / 'preview_p1.png'
        pix.save(str(out))
        print(f'[pdf] Thumbnail: {out.name}')
        return out
    except Exception as e:
        print(f'[pdf] Thumbnail error: {e}', file=sys.stderr)
        return None


# ── Score parsing (music21) — used for MusicXML / MIDI inputs ─────────────────

def _load_music21():
    try:
        import music21
        return music21
    except ImportError:
        print('[parse] music21 not installed — run: pip install music21', file=sys.stderr)
        sys.exit(1)


def parse_score_file(file_path):
    """Parse a MusicXML / MXL / MIDI file and return a music21 Score."""
    m21 = _load_music21()
    try:
        score = m21.converter.parse(str(file_path))
        if isinstance(score, m21.stream.Opus):
            score = score.mergeScores()
        if not isinstance(score, m21.stream.Score):
            score = m21.stream.Score([score])
        return score
    except Exception as e:
        print(f'[parse] music21 error: {e}', file=sys.stderr)
        return None


def analyze_score(score):
    """Extract metadata from a music21 Score. Returns dict."""
    m21 = _load_music21()

    flat = score.flatten()

    bpm = 120
    for el in flat.getElementsByClass(m21.tempo.MetronomeMark):
        if el.number:
            bpm = int(round(el.number))
            break

    time_sig = '4/4'
    for el in flat.getElementsByClass(m21.meter.TimeSignature):
        time_sig = f'{el.numerator}/{el.denominator}'
        break

    key_label = 'C'
    for el in flat.getElementsByClass(m21.key.Key):
        try:
            name = el.tonic.name.replace('-', 'b')
            key_label = name + (' minor' if el.mode == 'minor' else '')
        except Exception:
            pass
        break
    if key_label == 'C':
        for el in flat.getElementsByClass(m21.key.KeySignature):
            try:
                k = el.asKey()
                name = k.tonic.name.replace('-', 'b')
                key_label = name + (' minor' if k.mode == 'minor' else '')
            except Exception:
                pass
            break

    bars = 0
    for part in score.parts:
        n = len(list(part.getElementsByClass(m21.stream.Measure)))
        if n > bars:
            bars = n

    title = None
    if score.metadata and score.metadata.title:
        title = score.metadata.title

    return {
        'bpm':     bpm,
        'timeSig': time_sig,
        'key':     key_label,
        'bars':    bars,
        'title':   title,
    }


# ── Part -> track assignment (music21 path) ────────────────────────────────────

_BASS_KEYWORDS    = ('bass', 'tuba', 'cello', 'contrabass', 'double bass',
                     'baritone', 'euphonium')
_DRUMS_KEYWORDS   = ('drum', 'percussion', 'perc', 'timpani', 'kit', 'snare',
                     'cymbal', 'marimba', 'vibraphone', 'xylophone')
_MELODY_KEYWORDS  = ('violin', 'flute', 'oboe', 'clarinet', 'trumpet',
                     'soprano', 'piccolo', 'cornet', 'voice', 'vocal')
_PAD_KEYWORDS     = ('viola', 'trombone', 'horn', 'chor', 'choir', 'harmony',
                     'accompan', 'piano', 'organ', 'harpsichord', 'keyboard',
                     'guitar', 'harp', 'accordion')


def _part_name_role(name):
    """Look up role from instrument-name keywords; return None if no match."""
    for kw in _DRUMS_KEYWORDS:
        if kw in name: return 'drums'
    for kw in _BASS_KEYWORDS:
        if kw in name: return 'bass'
    for kw in _MELODY_KEYWORDS:
        if kw in name: return 'melody'
    for kw in _PAD_KEYWORDS:
        if kw in name: return 'pad'
    return None


def _avg_part_pitch(part):
    """Weighted-by-duration average MIDI pitch of a music21 Part. Returns None if empty."""
    total_pitch = 0.0
    total_dur   = 0.0
    try:
        for el in part.flatten().notes:
            try:
                dur = float(el.duration.quarterLength) or 0.25
                pitches = list(getattr(el, 'pitches', None) or [getattr(el, 'pitch', None)])
                for p in pitches:
                    if p is None: continue
                    total_pitch += p.midi * dur
                    total_dur   += dur
            except Exception:
                continue
    except Exception:
        return None
    return (total_pitch / total_dur) if total_dur > 0 else None


def _classify_part(part, part_index, total_parts, m21):
    """Return one of: 'melody', 'bass', 'pad', 'drums'.

    Used when classifying parts one at a time without cross-part comparison.
    Prefer _classify_parts() (collection-level, pitch-aware) when you have
    access to all parts at once.
    """
    instr = None
    try:
        instr = part.getInstrument(returnDefault=False)
    except Exception:
        pass

    name = ''
    if instr:
        name = (instr.instrumentName or '').lower()
        if hasattr(instr, 'midiChannel') and instr.midiChannel == 10:
            return 'drums'

    by_name = _part_name_role(name)
    if by_name:
        return by_name

    if total_parts == 1:
        return 'pad'
    if total_parts == 2:
        return 'melody' if part_index == 0 else 'bass'
    if part_index == 0:
        return 'melody'
    if part_index == total_parts - 1:
        return 'bass'
    return 'pad'


def _classify_parts(parts, m21):
    """Pitch-aware classification for a whole score. Returns list of roles.

    Strategy:
      1. Drums channel / drum keyword wins outright.
      2. Explicit melody/bass/pad keyword wins for that part.
      3. For remaining unassigned parts, rank by average pitch:
         highest -> melody, lowest -> bass, middle -> pad.
    """
    n = len(parts)
    roles = [None] * n

    for i, part in enumerate(parts):
        name = ''
        try:
            instr = part.getInstrument(returnDefault=False)
            if instr:
                name = (instr.instrumentName or '').lower()
                if hasattr(instr, 'midiChannel') and instr.midiChannel == 10:
                    roles[i] = 'drums'; continue
        except Exception:
            pass
        kw_role = _part_name_role(name)
        if kw_role:
            roles[i] = kw_role

    # Pitch-rank the unassigned parts.
    unassigned = [i for i, r in enumerate(roles) if r is None]
    if unassigned:
        pitched = [(i, _avg_part_pitch(parts[i])) for i in unassigned]
        # Parts that are entirely empty fall back to 'pad'.
        empty   = [i for i, p in pitched if p is None]
        ranked  = sorted([(i, p) for i, p in pitched if p is not None],
                         key=lambda x: -x[1])  # highest pitch first
        if len(ranked) == 1:
            # Single unassigned part: default to 'pad' to match prior behaviour.
            roles[ranked[0][0]] = 'pad'
        elif len(ranked) >= 2:
            # If 'melody' already taken by name keyword elsewhere, skip it.
            taken = {r for r in roles if r}
            top_role    = 'melody' if 'melody' not in taken else 'pad'
            bottom_role = 'bass'   if 'bass'   not in taken else 'pad'
            roles[ranked[0][0]]  = top_role
            roles[ranked[-1][0]] = bottom_role
            for idx, _p in ranked[1:-1]:
                roles[idx] = 'pad'
        for i in empty:
            roles[i] = 'pad'

    return roles


def score_to_tracks(score, bpm_override=None):
    """Convert music21 Score -> dict of 4 track note-event lists."""
    m21   = _load_music21()
    parts = list(score.parts)
    n     = len(parts)

    tracks = {'melody': [], 'bass': [], 'pad': [], 'drums': []}

    bpm = bpm_override
    if not bpm:
        for el in score.flatten().getElementsByClass(m21.tempo.MetronomeMark):
            if el.number:
                bpm = float(el.number)
                break
    if not bpm:
        bpm = 120.0

    roles_per_part = _classify_parts(parts, m21)

    for idx, part in enumerate(parts):
        role = roles_per_part[idx]

        for el in part.flatten().notesAndRests:
            if el.isRest:
                continue

            start_sec = float(el.offset) * 60.0 / bpm
            dur_sec   = float(el.duration.quarterLength) * 60.0 / bpm
            end_sec   = start_sec + max(dur_sec, 1 / bpm)

            vel_obj = getattr(el, 'volume', None)
            vel     = int(vel_obj.velocity) if vel_obj and vel_obj.velocity else 64

            pitches = []
            if hasattr(el, 'pitches'):
                pitches = [p.midi for p in el.pitches]
            else:
                try:
                    pitches = [el.pitch.midi]
                except Exception:
                    pass

            for midi_note in pitches:
                tracks[role].append({
                    'note':  midi_note,
                    'start': start_sec,
                    'end':   end_sec,
                    'vel':   vel,
                })

    return tracks, bpm


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(input_path, piece_id, title=None, composer=None, out_base=None,
            bpm_override=None, time_sig_override=None, provider='gemini'):
    """Full pipeline: input -> 4 MIDI files + catalog.json.

    PDFs are transcribed by AI vision (ai_transcribe.py); MusicXML/MIDI inputs
    are converted directly with music21. Returns the catalog dict.
    """
    input_path = Path(input_path)
    out_base   = Path(out_base) if out_base else DEFAULT_OUT
    out_dir    = out_base / piece_id
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()

    if suffix == '.pdf':
        # PDF -> AI-vision transcription. Every page is sent to the LLM, which
        # reads the printed music; there is no mechanical OMR step.
        render_pdf_thumbnail(input_path, out_dir)
        _sd = str(Path(__file__).parent.resolve())
        if _sd not in sys.path:
            sys.path.insert(0, _sd)
        import ai_transcribe
        try:
            tracks, bpm, meta = ai_transcribe.transcribe_pdf(
                input_path, out_dir, piece_id,
                title or piece_id.replace('_', ' ').title(),
                composer or '(unknown composer)',
                bpm_override=bpm_override, provider=provider)
        except Exception as e:
            print(f'[convert] AI transcription failed: {e}', file=sys.stderr)
            sys.exit(2)

    else:
        # MusicXML / MIDI -> direct music21 conversion.
        print(f'[convert] Parsing {input_path.name}…')
        score = parse_score_file(input_path)
        if score is None:
            sys.exit(1)
        meta = analyze_score(score)
        bpm  = bpm_override or meta['bpm']
        print(f'[convert] BPM={bpm}  key={meta["key"]}  '
              f'timeSig={meta["timeSig"]}  bars={meta["bars"]}')
        tracks, bpm = score_to_tracks(score, bpm_override=bpm)

    total_notes = sum(len(v) for v in tracks.values())
    print(f'[convert] Notes — melody:{len(tracks["melody"])}  '
          f'bass:{len(tracks["bass"])}  pad:{len(tracks["pad"])}  '
          f'drums:{len(tracks["drums"])}  total:{total_notes}')

    if total_notes == 0:
        print('[convert] No note data found — nothing to write.', file=sys.stderr)
        sys.exit(2)

    # ── Quality report ────────────────────────────────────────────────────────
    # Density per bar gives a rough indication of how complete the import is.
    # Solo piano usually averages 6-15 notes/bar; orchestral 10-25.  Very low
    # densities (< 3) often indicate OMR missed notes; very high may indicate
    # bar merging / scaling issues.
    bars = meta.get('bars') or 1
    if bars > 0:
        density = total_notes / max(1, bars)
        if density < 3:
            quality = 'SPARSE - likely missing notes'
        elif density > 30:
            quality = 'DENSE - check for bar merging'
        else:
            quality = 'normal'
        per_track = '  '.join(f'{r}:{len(tracks[r])/bars:.1f}'
                              for r in ('melody', 'bass', 'pad', 'drums')
                              if tracks[r])
        print(f'[convert] Quality: {density:.1f} notes/bar ({quality})  '
              f'per-track [{per_track}]')

    # ── Write MIDI files ──────────────────────────────────────────────────────
    ts = meta['timeSig']
    for track_name in ('melody', 'bass', 'pad', 'drums'):
        out_path = out_dir / f'{track_name}.mid'
        write_midi_track(tracks[track_name], bpm, ts, out_path,
                         channel=0 if track_name != 'drums' else 9)
        print(f'[convert] Wrote {out_path.name}  ({len(tracks[track_name])} notes)')

    # ── Catalog entry ─────────────────────────────────────────────────────────
    resolved_title = title or meta.get('title') or piece_id.replace('_', ' ').title()
    catalog = {
        'id':          piece_id,
        'title':       resolved_title,
        'composer':    composer or '',
        'genre':       'Classical',
        'mood':        'Expressive',
        'bpm':         int(round(bpm)),
        'key':         meta['key'],
        'bars':        meta['bars'],
        'timeSig':     meta['timeSig'],
        'source':      'human',
        'importedFrom': str(input_path.name),
        'description': (
            'Imported from PDF via AI-vision transcription (ai_transcribe.py)'
            if suffix == '.pdf'
            else f'Imported from {input_path.suffix.lstrip(".").upper()} by pdf_to_midi.py'
        ),
    }

    cat_path = out_dir / 'catalog.json'
    cat_path.write_text(json.dumps(catalog, indent=2))
    print(f'[convert] Catalog -> {cat_path}')

    return catalog


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert sheet music PDF/MusicXML/MIDI to 4-track showcase MIDI')
    parser.add_argument('input',   help='Input file (.pdf / .xml / .mxl / .mid)')
    parser.add_argument('--id',    dest='piece_id',
                        help='Piece ID slug (default: input filename stem)')
    parser.add_argument('--title', help='Human-readable title')
    parser.add_argument('--composer', help='Composer name (used as AI context)')
    parser.add_argument('--out',   help='Output base directory')
    parser.add_argument('--bpm',      type=int, help='Override detected BPM')
    parser.add_argument('--time-sig', dest='time_sig',
                        help='Override detected time signature (e.g. "3/8")')
    parser.add_argument('--ai-correct', action='store_true',
                        help='After import, ask an LLM (via browser-ai-bridge) '
                             'to fix obvious OMR errors. Requires the bridge '
                             'running at AI_BRIDGE_URL (default http://localhost:3333). '
                             'Skips quietly if the bridge is unreachable.')
    parser.add_argument('--ai-provider', default='gemini',
                        help='AI provider via browser-ai-bridge for the AI '
                             'stages (default gemini; also chatgpt|copilot|deepseek|grok)')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f'Error: file not found: {inp}', file=sys.stderr)
        sys.exit(1)

    piece_id = args.piece_id or inp.stem.lower().replace(' ', '_').replace('-', '_')
    piece_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in piece_id)

    catalog = convert(
        input_path        = inp,
        piece_id          = piece_id,
        title             = args.title,
        composer          = args.composer,
        out_base          = args.out,
        bpm_override      = args.bpm,
        time_sig_override = args.time_sig,
        provider          = args.ai_provider,
    )

    print('\n-- Catalog entry --')
    print(json.dumps(catalog, indent=2))

    # Mechanical music-theory validation of the finished import (rule-based,
    # no AI) — writes showcase-midi/<id>/theory.json for the dashboard badge.
    try:
        _sd = str(Path(__file__).parent.resolve())
        if _sd not in sys.path:
            sys.path.insert(0, _sd)
        import theory_check
        result = theory_check.validate_piece(piece_id)
        theory_check.write_result(piece_id, result)
        print(f'[theory] {result["status"]} '
              f'(score {result["score"]}, {len(result["issues"])} issue(s))')
    except Exception as e:
        print(f'[theory] validation skipped: {e}', file=sys.stderr)

    # Optional extra stage: note-level AI cleanup of remaining errors.
    if args.ai_correct:
        # Ensure the dashboard directory (where ai_correct.py lives) is on sys.path
        _script_dir = str(Path(__file__).parent.resolve())
        if _script_dir not in sys.path:
            sys.path.insert(0, _script_dir)
        try:
            import ai_correct
        except ImportError as e:
            print(f'\n[ai-correct] skipped: cannot import ai_correct ({e})',
                  file=sys.stderr)
            return
        if not ai_correct._bridge_ping():
            print(f'\n[ai-correct] skipped: browser-ai-bridge not running at '
                  f'{ai_correct.BRIDGE_URL}', file=sys.stderr)
            print('[ai-correct] start it with: cd c:/Users/Work/browser-ai-bridge && '
                  'npm install && npm start', file=sys.stderr)
            return
        print(f'\n[ai-correct] querying {args.ai_provider} for corrections...')
        try:
            _cat, prompt = ai_correct.serialise_piece(piece_id)
            response    = ai_correct._bridge_ask(prompt, provider=args.ai_provider)
            corrections = ai_correct.parse_corrections(response)
            print(f'[ai-correct] {len(corrections)} corrections suggested')
            if corrections:
                applied, skipped = ai_correct.apply_corrections(piece_id, corrections)
                print(f'[ai-correct] applied {applied}, skipped {skipped}')
        except Exception as e:
            print(f'[ai-correct] failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
