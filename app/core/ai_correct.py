"""
ai_correct.py - Use a browser-driven LLM (via browser-ai-bridge) to identify
and correct errors in an OMR-imported piece. Runs AFTER the mechanical
pdf_to_midi.py pipeline.

Workflow:
  1. Read the imported MIDI tracks (melody.mid, bass.mid) for a piece.
  2. Serialise them to a compact text representation grouped by bar.
  3. Prompt an LLM with the piece's title + composer + serialised notes,
     asking it to identify and fix obviously-wrong notes against the
     canonical score (the model is expected to know famous pieces).
  4. Parse the LLM's JSON response into a list of corrections.
  5. Apply corrections back to the MIDI files.

Requires browser-ai-bridge running at http://localhost:3333 with at least
one active AI provider session (ChatGPT/Gemini/etc.).

Usage:
    python ai_correct.py <piece_id>
    python ai_correct.py <piece_id> --provider gemini
    python ai_correct.py <piece_id> --dry-run  # show prompt + response, don't apply

Returns exit code 0 on success, non-zero on bridge/network/parse failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

DASHBOARD     = Path(__file__).parent
SHOWCASE_MIDI = DASHBOARD / 'showcase-midi'
BRIDGE_URL    = os.environ.get('AI_BRIDGE_URL', 'http://localhost:3333')
NODE_BIN      = os.environ.get('NODE_BIN', r'C:\Program Files\nodejs\node.exe')
# Model tier for transcription/correction. Sheet-music reading is a
# high-volume, mechanical visual task — the fast/cheap model (Gemini Flash,
# GPT-4o, DeepSeek-V3) is plenty and many times quicker than the slow
# "thinking"/Pro tiers. Override with AI_BRIDGE_MODE=pro|thinking if needed.
BRIDGE_MODE   = os.environ.get('AI_BRIDGE_MODE', 'fast')

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Map MIDI tick durations (at div=16) to compact note-type tags.  div=16 means
# 16 ticks per quarter note, so quarter=16, eighth=8, sixteenth=4, 32nd=2.
DUR_TO_TAG = {
    64: 'w',   # whole
    48: 'h.',  # dotted half
    32: 'h',   # half
    24: 'q.',  # dotted quarter
    16: 'q',   # quarter
    12: '8.',  # dotted 8th
    8:  '8',   # 8th
    6:  '16.', # dotted 16th
    4:  '16',  # 16th
    3:  '32.', # dotted 32nd
    2:  '32',  # 32nd
    1:  '64',  # 64th (rare; usually an artefact)
}
# Reverse: note-type tag -> ticks (for parsing AI bar-rewrites).
TAG_TO_TICKS = {tag: ticks for ticks, tag in DUR_TO_TAG.items()}
TAG_TO_TICKS['w.'] = 96  # dotted whole, just in case


def midi_to_name(midi):
    return f'{PITCH_NAMES[midi % 12]}{midi // 12 - 1}'


def name_to_midi(name):
    """Parse 'C#5', 'Bb4' etc. into MIDI number. None on parse failure."""
    m = re.match(r'^([A-Ga-g])([#b]?)(-?\d+)$', name.strip())
    if not m:
        return None
    step, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    base = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[step]
    if accidental == '#':
        base += 1
    elif accidental == 'b':
        base -= 1
    return base + (octave + 1) * 12


def _read_midi(path):
    """Read a MIDI file into a list of {bar, beat_in_bar, pitch_name, dur_tag,
    midi, start_qL, end_qL}. Returns (events, ticks_per_beat)."""
    import mido
    mid = mido.MidiFile(str(path))
    TPB = mid.ticks_per_beat
    events = []
    for tr in mid.tracks:
        abs_t = 0
        ons = {}
        for msg in tr:
            abs_t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                ons[msg.note] = abs_t
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                s = ons.pop(msg.note, None)
                if s is None:
                    continue
                events.append({
                    'midi':     msg.note,
                    'pitch':    midi_to_name(msg.note),
                    'start_qL': s / TPB,
                    'end_qL':   abs_t / TPB,
                })
    events.sort(key=lambda e: (e['start_qL'], e['midi']))
    return events, TPB


def _serialise_track(events, time_sig, name='MELODY', max_bars=None, bar_offset=0):
    """Serialise note events to a compact text representation grouped by bar.

    bar_offset (1-indexed) selects which bar to START at; max_bars caps the
    number of bars after that. Pass (0, None) for "all bars from start".
    """
    num, den = map(int, time_sig.split('/'))
    qL_per_bar = num * 4 / den

    by_bar = defaultdict(list)
    for e in events:
        bar = int(e['start_qL'] // qL_per_bar) + 1     # 1-indexed
        beat = e['start_qL'] - (bar - 1) * qL_per_bar  # qL into bar
        dur_qL  = max(0.0625, e['end_qL'] - e['start_qL'])
        dur_t   = int(round(dur_qL * 16))              # at our internal div=16
        tag     = DUR_TO_TAG.get(dur_t, f'{dur_t}t')
        by_bar[bar].append((beat, e['pitch'], tag))

    lines = [f'{name}:']
    sorted_bars = sorted(by_bar.keys())
    if bar_offset:
        sorted_bars = [b for b in sorted_bars if b > bar_offset]
    if max_bars is not None:
        sorted_bars = sorted_bars[:max_bars]
    for bar in sorted_bars:
        notes = sorted(by_bar[bar], key=lambda t: t[0])
        flat  = ' '.join(f'{p}({tag})' for _beat, p, tag in notes)
        lines.append(f' {bar:3d}: {flat}')
    return '\n'.join(lines)


def _notes_by_bar(events, qL_per_bar):
    """Group note events into {bar_number: 'C5(q) E5(8) ...'} compact strings."""
    by_bar = defaultdict(list)
    for e in events:
        bar  = int(e['start_qL'] // qL_per_bar) + 1
        beat = e['start_qL'] - (bar - 1) * qL_per_bar
        dur_t = int(round(max(0.0625, e['end_qL'] - e['start_qL']) * 16))
        tag   = DUR_TO_TAG.get(dur_t, f'{dur_t}t')
        by_bar[bar].append((beat, e['pitch'], tag))
    out = {}
    for bar, notes in by_bar.items():
        notes.sort(key=lambda t: t[0])
        out[bar] = ' '.join(f'{p}({tag})' for _b, p, tag in notes)
    return out


def serialise_bar_context(piece_id, start_bar, n_bars, reference_id=None):
    """Build a bar-by-bar contextual prompt for a window of bars.

    Each bar is presented with its PREV and NEXT neighbours as context, and
    (when a reference is supplied) the reference's version of that bar. The AI
    is asked to judge EACH bar individually and rewrite only the ones that are
    actually wrong. Returns (catalog, prompt, bars_covered).
    """
    catalog, melody, bass = _load_piece(piece_id)
    ts = catalog.get('timeSig', '4/4')
    num, den = map(int, ts.split('/'))
    qL_per_bar = num * 4 / den

    mel_bars  = _notes_by_bar(melody, qL_per_bar)
    bass_bars = _notes_by_bar(bass,   qL_per_bar)
    total_bars = catalog.get('bars') or max([0, *mel_bars, *bass_bars])

    ref_mel_bars = ref_bass_bars = None
    if reference_id:
        ref_cat, ref_mel, ref_bass = _load_piece(reference_id)
        ref_num, ref_den = map(int, ref_cat.get('timeSig', '4/4').split('/'))
        ref_qpb = ref_num * 4 / ref_den
        ref_mel_bars  = _notes_by_bar(ref_mel,  ref_qpb)
        ref_bass_bars = _notes_by_bar(ref_bass, ref_qpb)

    def bar_line(bars_dict, b):
        return bars_dict.get(b, '(empty)')

    end_bar = min(total_bars, start_bar + n_bars - 1)
    blocks  = []
    for b in range(start_bar, end_bar + 1):
        lines = [f'=== Bar {b} ===']
        if b > 1:
            lines.append(f'  PREV  melody: {bar_line(mel_bars, b-1)}')
            lines.append(f'        bass  : {bar_line(bass_bars, b-1)}')
        lines.append(f'  THIS  melody: {bar_line(mel_bars, b)}      <-- evaluate')
        lines.append(f'        bass  : {bar_line(bass_bars, b)}      <-- evaluate')
        lines.append(f'  NEXT  melody: {bar_line(mel_bars, b+1)}')
        lines.append(f'        bass  : {bar_line(bass_bars, b+1)}')
        if ref_mel_bars is not None:
            lines.append(f'  REF   melody: {bar_line(ref_mel_bars, b)}')
            lines.append(f'        bass  : {bar_line(ref_bass_bars, b)}')
        blocks.append('\n'.join(lines))

    title = catalog.get('title', piece_id)
    ref_note = ('A REF line gives the ground-truth version of each bar from a '
                'clean reference recording -- match THIS to REF where they '
                'should agree.\n' if ref_mel_bars is not None else '')
    prompt = textwrap.dedent(f"""
    You are an expert music transcription auditor working BAR BY BAR through an
    OMR (optical music recognition) import of a well-known piece.

    Piece: {title}   Meter: {ts}   Key: {catalog.get('key','C')}   Tempo: {catalog.get('bpm',90)} BPM

    For each bar below you are given the bar to evaluate (THIS) plus its
    neighbouring bars (PREV / NEXT) as musical context. {ref_note}
    Notation: scientific pitch (C4 = middle C). Duration tags in parentheses:
    w=whole h=half q=quarter 8=eighth 16=sixteenth 32=32nd; a trailing dot
    (q. 8.) = dotted. Notes are listed left-to-right in time order.

    Judge EACH bar. If a bar is musically wrong (wrong pitches, wrong octave,
    wrong rhythm, junk notes, missing notes), output a full corrected rewrite
    of that bar. If a bar is already fine, do NOT include it.

    {chr(10).join(blocks)}

    Respond with JSON ONLY (no prose, no markdown fences):
    {{
      "corrections": [
        {{"track": "melody", "bar": N, "rewrite": "C5(q) E5(8) G5(8) ...", "reason": "brief"}},
        {{"track": "bass",   "bar": M, "rewrite": "C3(8) G3(8) ...",        "reason": "brief"}}
      ]
    }}

    Rules:
      - "rewrite" is the COMPLETE corrected note list for that bar (one track).
      - Pitch + duration tag for every note, in time order.
      - Only include bars that genuinely need fixing.
      - If every bar shown is fine, return {{"corrections": []}}.
      - Output JSON only.
    """).strip()
    return catalog, prompt, (start_bar, end_bar)


def _build_correction_prompt(title, composer, key, time_sig, bpm, melody_text, bass_text,
                              ref_melody=None, ref_bass=None, ref_meta=None):
    """Build the AI prompt. When ref_* are provided, the AI is asked to align
    the imported notes against the reference (a known-good MIDI of the same
    piece) instead of guessing from canonical memory."""
    if ref_melody is not None:
        reference_section = textwrap.dedent(f"""
        REFERENCE (this is the CORRECT version of the same piece, from a
        clean MIDI source). Use this as ground truth — match the import to
        the reference where they should agree.

        Reference meter: {ref_meta.get('timeSig', '?')}  BPM: {ref_meta.get('bpm', '?')}  bars: {ref_meta.get('bars', '?')}

        Reference {ref_melody}

        Reference {ref_bass}
        """).strip()
        task = textwrap.dedent("""
        Your task: compare the IMPORTED notes against the REFERENCE and
        produce corrections that align the import to the reference. Focus on
        wrong pitches, wrong octaves, and obviously misplaced notes. Don't try
        to correct the BAR STRUCTURE -- the import may have more bars due to
        OMR cascade errors. Just fix individual notes within the imported
        bars to match what the reference plays at roughly the same point in
        the piece.
        """).strip()
    else:
        reference_section = ''
        task = textwrap.dedent("""
        Your task: identify and correct only the OBVIOUSLY WRONG notes
        -- wrong pitch, wrong octave, or duplicated/missing notes -- compared
        to the canonical score that this piece is famous for. Do not guess at
        ambiguous cases.
        """).strip()

    return textwrap.dedent(f"""
    You are an expert music transcription auditor. I have OMR (optical music
    recognition) output for a well-known classical piece. The OMR engine made
    some errors. {task}

    Piece:    {title}
    Composer: {composer or '(inferred from title)'}
    Meter:    {time_sig}
    Key:      {key}
    Tempo:    {bpm} BPM (quarter-note per minute)

    Notation legend:
      Pitches use scientific notation: C4 = middle C, E5 = E above treble staff.
      Duration tags: q=quarter, q.=dotted quarter, 8=eighth, 8.=dotted 8th,
                     16=sixteenth, 16.=dotted 16th, 32=32nd, h=half.
      Each "bar:" line lists notes left-to-right in time order.

    {reference_section}

    Imported {melody_text}

    Imported {bass_text}

    Respond with JSON ONLY (no prose, no markdown fences):
    {{
      "corrections": [
        {{"track": "melody|bass", "bar": N, "wrong_pitch": "X5", "correct_pitch": "Y5", "reason": "brief"}},
        ...
      ]
    }}

    Rules:
      - Only correct notes you are CERTAIN about. Skip ambiguous cases.
      - Match corrections by track + bar + wrong_pitch; we'll find the note.
      - If multiple notes in the same bar share the wrong_pitch, we'll fix the
        first occurrence.
      - If the piece looks correct, return {{"corrections": []}}.
      - Output JSON only, no other text.
    """).strip()


def _load_piece(piece_id):
    """Read a piece's catalog + melody/bass MIDI events. Returns (catalog, melody_events, bass_events)."""
    piece_dir = SHOWCASE_MIDI / piece_id
    cat_path  = piece_dir / 'catalog.json'
    if not cat_path.exists():
        raise FileNotFoundError(f'No catalog.json at {cat_path}')
    with open(cat_path, encoding='utf-8') as fh:
        catalog = json.load(fh)
    melody_path = piece_dir / 'melody.mid'
    bass_path   = piece_dir / 'bass.mid'
    melody, _   = _read_midi(melody_path) if melody_path.exists() else ([], 0)
    bass, _     = _read_midi(bass_path)   if bass_path.exists() else ([], 0)
    return catalog, melody, bass


def serialise_piece(piece_id, max_bars=None, reference_id=None, bar_offset=0):
    """Return (catalog_dict, prompt_text) for a piece, or raise on missing data.

    If reference_id is given, the prompt also includes the reference piece's
    serialised notes as 'ground truth' for AI to align against.

    bar_offset shifts which bars are serialised (so callers can chunk through a
    long piece: pass bar_offset=0 then 16 then 32 etc.).
    """
    catalog, melody, bass = _load_piece(piece_id)
    ts = catalog.get('timeSig', '4/4')
    melody_text = _serialise_track(melody, ts, 'MELODY', max_bars, bar_offset)
    bass_text   = _serialise_track(bass,   ts, 'BASS',   max_bars, bar_offset)

    ref_melody_text = None
    ref_bass_text   = None
    ref_meta        = None
    if reference_id:
        ref_cat, ref_mel, ref_bass = _load_piece(reference_id)
        ref_ts = ref_cat.get('timeSig', '4/4')
        ref_melody_text = _serialise_track(ref_mel,  ref_ts, 'MELODY', max_bars)
        ref_bass_text   = _serialise_track(ref_bass, ref_ts, 'BASS',   max_bars)
        ref_meta = ref_cat

    # Composer: prefer the catalog field (set from the import form); otherwise
    # guess from the title / importedFrom / piece_id.
    title    = catalog.get('title', piece_id)
    haystack = ' '.join([
        title, catalog.get('importedFrom', ''), piece_id,
    ]).lower()
    composer = catalog.get('composer') or None
    COMPOSERS = (
        ('beethoven', 'Beethoven'), ('mozart', 'Mozart'),
        ('bach', 'Bach'), ('chopin', 'Chopin'),
        ('schubert', 'Schubert'), ('brahms', 'Brahms'),
        ('handel', 'Handel'), ('haydn', 'Haydn'),
        ('debussy', 'Debussy'), ('liszt', 'Liszt'),
        ('schumann', 'Schumann'), ('mendelssohn', 'Mendelssohn'),
        ('tchaikovsky', 'Tchaikovsky'), ('rachmaninoff', 'Rachmaninoff'),
    )
    if composer is None:
        for needle, name in COMPOSERS:
            if needle in haystack:
                composer = name; break
    # Famous-piece nickname fallback (works even when title omits composer)
    NICKNAMES = (
        ('fur elise',          'Beethoven'),
        ('moonlight sonata',   'Beethoven'),
        ('clair de lune',      'Debussy'),
        ('eine kleine',        'Mozart'),
        ('rondo alla turca',   'Mozart'),
        ('canon in d',         'Pachelbel'),
        ('jesu, joy',          'Bach'),
        ('air on the g',       'Bach'),
    )
    if composer is None:
        for needle, name in NICKNAMES:
            if needle in haystack:
                composer = name; break

    prompt = _build_correction_prompt(
        title       = title,
        composer    = composer,
        key         = catalog.get('key', 'C'),
        time_sig    = ts,
        bpm         = catalog.get('bpm', 90),
        melody_text = melody_text,
        bass_text   = bass_text,
        ref_melody  = ref_melody_text,
        ref_bass    = ref_bass_text,
        ref_meta    = ref_meta,
    )
    return catalog, prompt


def align_to_reference(piece_id, reference_id, scale_to_target_bpm=True):
    """Rebuild a piece's melody/bass MIDI from a clean reference of the same
    composition. Used when OMR mis-recognised the score so badly (e.g. 2-3x
    phantom content) that note-level correction can't recover it.

    The reference's note timing is preserved; only the BPM is mapped so the
    catalog's tempo stays consistent. Catalog bars/timeSig/key are synced to
    the reference too. Returns a short summary dict.
    """
    _script_dir = str(Path(__file__).parent.resolve())
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from pdf_to_midi import write_midi_track

    tgt_dir = SHOWCASE_MIDI / piece_id
    ref_dir = SHOWCASE_MIDI / reference_id
    tgt_cat = json.loads((tgt_dir / 'catalog.json').read_text(encoding='utf-8'))
    ref_cat = json.loads((ref_dir / 'catalog.json').read_text(encoding='utf-8'))

    ref_ts  = ref_cat.get('timeSig', '4/4')
    ref_bpm = float(ref_cat.get('bpm', 120) or 120)
    # Keep the target's existing BPM unless it's clearly a placeholder.
    tgt_bpm = float(tgt_cat.get('bpm', 0) or 0) or ref_bpm
    out_bpm = tgt_bpm if scale_to_target_bpm else ref_bpm

    summary = {'tracks': {}}
    for track in ('melody', 'bass', 'pad', 'drums'):
        ref_path = ref_dir / f'{track}.mid'
        tgt_path = tgt_dir / f'{track}.mid'
        if not ref_path.exists():
            continue
        events, _tpb = _read_midi(ref_path)
        # Reference qL positions are tempo-independent; re-time at out_bpm.
        events_sec = [{'note': e['midi'],
                       'start': e['start_qL'] * 60.0 / out_bpm,
                       'end':   e['end_qL']   * 60.0 / out_bpm,
                       'vel':   64}
                      for e in events]
        write_midi_track(events_sec, out_bpm, ref_ts, str(tgt_path),
                         channel=9 if track == 'drums' else 0)
        summary['tracks'][track] = len(events)

    # Sync catalog metadata to the reference's structure.
    tgt_cat['bars']    = ref_cat.get('bars', tgt_cat.get('bars'))
    tgt_cat['timeSig'] = ref_ts
    tgt_cat['key']     = ref_cat.get('key', tgt_cat.get('key'))
    tgt_cat['bpm']     = int(round(out_bpm))
    tgt_cat['description'] = (tgt_cat.get('description', '') +
                              f' [resynced from reference: {reference_id}]')
    (tgt_dir / 'catalog.json').write_text(json.dumps(tgt_cat, indent=2))
    summary['bpm'] = int(round(out_bpm))
    summary['bars'] = tgt_cat['bars']
    return summary


def _bridge_ping():
    """Return True if browser-ai-bridge is up AND connected to Chrome."""
    try:
        with urllib.request.urlopen(f'{BRIDGE_URL}/api/ping', timeout=3) as r:
            data = json.loads(r.read())
    except Exception:
        return False
    if data.get('status') != 'ready':
        return False
    # `browser` may be either a string ("connected") or a dict {"connected": true}
    browser = data.get('browser')
    if isinstance(browser, dict):
        return bool(browser.get('connected'))
    return browser == 'connected'


def _bridge_post(endpoint, payload, timeout, attempts, what):
    """POST JSON to a bridge endpoint, returning data['response'].

    Retries on timeout / connection error / a non-success body. Each attempt
    is a fresh request, so a wedged AI session is abandoned and the bridge
    hands the retry a clean session.
    """
    body = json.dumps(payload).encode('utf-8')
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                f'{BRIDGE_URL}{endpoint}', data=body,
                headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            if not data.get('success'):
                raise RuntimeError(f'bridge {what} failed: {data.get("error") or data}')
            resp = data.get('response', '')
            # Detect a contaminated session (e.g. stuck in Gemini's video/image
            # tool mode) — the retry gets a fresh session.
            low = resp.strip().lower()
            if not low or 'i can only generate' in low or \
               low.startswith('try another prompt'):
                raise RuntimeError(f'unusable {what} response '
                                   f'(wrong tool mode?): {resp[:120]!r}')
            return resp
        except Exception as e:
            last_err = e
            if attempt < attempts:
                print(f'[ai] {what} attempt {attempt}/{attempts} failed '
                      f'({type(e).__name__}: {e}); retrying in 8s...',
                      file=sys.stderr)
                time.sleep(8)
    raise RuntimeError(f'bridge {what} failed after {attempts} attempts: {last_err}')


# Cross-provider fallback order. If the requested provider fails (down, not
# logged in, wedged session, unusable answer) the call retries on the next
# provider — every provider now supports image upload, so sheet-music vision
# survives a Gemini outage. Override/trim with AI_FALLBACK_PROVIDERS; set it to
# a single provider to disable fallback.
FALLBACK_PROVIDERS = [p.strip().lower() for p in os.environ.get(
    'AI_FALLBACK_PROVIDERS', 'gemini,chatgpt,deepseek,copilot').split(',')
    if p.strip()]


def _provider_chain(primary):
    """Ordered, de-duplicated list: the requested provider first, then the
    configured fallbacks."""
    chain = []
    for p in [str(primary or '').lower()] + FALLBACK_PROVIDERS:
        if p and p not in chain:
            chain.append(p)
    return chain or ['gemini']


def _bridge_call(endpoint, base_payload, provider, timeout, attempts, what):
    """POST to a bridge endpoint, falling back across providers on failure."""
    chain = _provider_chain(provider)
    last_err = None
    for i, prov in enumerate(chain):
        try:
            return _bridge_post(endpoint, {**base_payload, 'provider': prov},
                                timeout, attempts, f'{what}[{prov}]')
        except Exception as e:
            last_err = e
            if i + 1 < len(chain):
                print(f'[ai] {what}: provider "{prov}" failed ({e}); '
                      f'falling back to "{chain[i + 1]}"', file=sys.stderr)
    raise RuntimeError(f'{what} failed on all providers {chain}: {last_err}')


def _bridge_ask(prompt, provider='gemini', timeout=420, attempts=3,
                mode=BRIDGE_MODE):
    """Send a prompt to the bridge and return the text response (with retries
    and cross-provider fallback)."""
    return _bridge_call('/api/ask', {'prompt': prompt, 'mode': mode},
                        provider, timeout, attempts, 'ask')


def _bridge_image_ask(image_path, prompt, provider='gemini', timeout=420,
                      attempts=3, mode=BRIDGE_MODE):
    """Upload an image to /api/image-ask and return the response (retries +
    cross-provider fallback)."""
    return _bridge_call('/api/image-ask', {
        'imagePath': str(image_path),
        'prompt': prompt,
        'label': 'sheet-music-visual-heal',
        'mode': mode,
    }, provider, timeout, attempts, 'image-ask')


def _bridge_audio_ask(audio_path, prompt, provider='gemini', timeout=600,
                      attempts=2, mode=BRIDGE_MODE):
    """Upload an audio clip to /api/audio-ask and return the response.

    Used to let the AI *hear* a synth render of a transcription so it can
    catch wrong notes / rhythm by ear. Longer timeout than image-ask because
    Gemini ingests the audio before it can answer.
    """
    return _bridge_call('/api/audio-ask', {
        'audioPath': str(audio_path),
        'prompt': prompt,
        'label': 'sheet-music-audio-validate',
        'mode': mode,
    }, provider, timeout, attempts, 'audio-ask')


def capture_sheet_pages(piece_id, max_pages=4):
    """Render the piece's sheet music to PNG page images via the node helper.
    Returns a list of PNG file paths. Requires the dashboard server running."""
    script = DASHBOARD / 'ss_sheet_pages.mjs'
    if not script.exists():
        raise FileNotFoundError(f'screenshot helper missing: {script}')
    proc = subprocess.run(
        [NODE_BIN, str(script), piece_id, str(max_pages)],
        capture_output=True, text=True, timeout=180,
    )
    pages = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                pages = json.loads(line).get('pages', [])
            except json.JSONDecodeError:
                pass
    if not pages:
        raise RuntimeError(f'no sheet pages rendered (node stderr: {proc.stderr[:300]})')
    return pages


def _build_visual_prompt(title, composer, time_sig, key, bpm, page_num, total_pages):
    return textwrap.dedent(f"""
    The attached image is a screenshot of page {page_num} of {total_pages} of
    rendered sheet music. It is an OMR (optical music recognition) import that
    may contain VISUAL errors. The piece is a well-known work, so you know how
    it should look.

    Piece:    {title}
    Composer: {composer or '(inferred from title)'}
    Meter:    {time_sig}   Key: {key}   Tempo: {bpm} BPM

    Bar numbers are printed at the start of each system (line) of music. The
    upper staff (treble clef) is the MELODY track; the lower staff (bass clef)
    is the BASS track.

    Look at the sheet music and find bars that are clearly WRONG:
      - notes stranded far above/below the staff on extreme ledger lines
      - obviously wrong pitches or octaves for this famous piece
      - garbled or impossible-looking rhythms
      - junk/spurious notes that don't belong

    For every wrong bar, give the CORRECT content of that bar (the version a
    correct score of this piece would have).

    Respond with JSON ONLY (no prose, no markdown fences):
    {{
      "corrections": [
        {{"track": "melody", "bar": N, "rewrite": "C5(q) E5(8) G5(8) ...", "reason": "what looked wrong"}},
        {{"track": "bass",   "bar": M, "rewrite": "C3(8) G3(8) ...",        "reason": "..."}}
      ]
    }}

    Notation for "rewrite": scientific pitch (C4 = middle C); duration tags in
    parentheses -- w=whole h=half q=quarter 8=eighth 16=sixteenth 32=32nd, a
    trailing dot means dotted (q. 8.). List notes left-to-right in time order.

    Only include bars that genuinely look wrong. If the page looks fine,
    return {{"corrections": []}}. Output JSON only.
    """).strip()


def parse_corrections(ai_response):
    """Extract the corrections JSON from the AI's response text."""
    # Strip optional markdown code fences
    txt = ai_response.strip()
    if txt.startswith('```'):
        txt = re.sub(r'^```(?:json)?\s*\n', '', txt)
        txt = re.sub(r'\n?```\s*$', '', txt)
    # Find first JSON object
    m = re.search(r'\{[\s\S]*\}', txt)
    if not m:
        raise ValueError(f'no JSON object found in AI response: {txt[:200]}')
    obj = json.loads(m.group(0))
    return obj.get('corrections', [])


def _ticks_for_tag(tag):
    """Parse a duration tag ('16', 'q.', '8', '5t', ...) into ticks at div=16.
    A grace tag ('g') is non-metrical and contributes 0 ticks."""
    tag = tag.strip()
    if tag == 'g':
        return 0
    if tag in TAG_TO_TICKS:
        return TAG_TO_TICKS[tag]
    m = re.match(r'^(\d+)t$', tag)            # raw "<n>t" form
    if m:
        return max(1, int(m.group(1)))
    if tag.isdigit():                          # bare number = note-value name
        return TAG_TO_TICKS.get(tag, 4)
    return 4  # fall back to a 16th note


def _parse_rewrite(rewrite):
    """Parse a bar rewrite into [(midi, ticks), ...].

    A rest is represented as (None, ticks): it advances time without sounding a
    note, so callers can lay notes at their true beat positions within the bar.

    Accepts either a compact string ('F4(16) R(8) A4(16)') or a list of such
    tokens / dicts ({'pitch':'F4','dur':'16'}).
    """
    tokens = []
    if isinstance(rewrite, str):
        tokens = rewrite.split()
    elif isinstance(rewrite, list):
        for item in rewrite:
            if isinstance(item, str):
                tokens.append(item)
            elif isinstance(item, dict):
                p = item.get('pitch') or item.get('note')
                d = item.get('dur') or item.get('duration') or '16'
                if p:
                    tokens.append(f'{p}({d})')
    out = []
    for tok in tokens:
        tok = tok.strip()
        rest = re.match(r'^[Rr]\(([^)]+)\)$', tok)   # rest: R(8), r(q.) ...
        if rest:
            out.append((None, _ticks_for_tag(rest.group(1))))
            continue
        # A chord ("A4+C5+E5(8)") counts as ONE rhythmic event; represent it by its
        # top note so duration sums stay correct (full chord pitches live in the bar
        # string and are rendered/played by the frontend).
        m = re.match(r'^([A-Ga-g][#b]?-?\d+(?:\+[A-Ga-g][#b]?-?\d+)*)\(([^)]+)\)$', tok)
        if not m:
            continue
        if m.group(2).strip() == 'g':
            continue   # grace note: a non-metrical ornament, not a rhythmic event
        midis = [mm for mm in (name_to_midi(p) for p in m.group(1).split('+')) if mm is not None]
        if not midis:
            continue
        out.append((max(midis), _ticks_for_tag(m.group(2))))
    return out


def apply_corrections(piece_id, corrections):
    """Apply corrections to a piece's MIDI files. Two correction kinds:

      pitch-swap:   {track, bar, wrong_pitch, correct_pitch}
      bar-rewrite:  {track, bar, rewrite: "C5(q) E5(8) G5(8) ..."}

    Bar-rewrite replaces ALL notes in a bar with the given sequence (placed
    consecutively from the bar start), so it fixes wrong rhythms too -- not
    just wrong pitches. Returns (applied_count, skipped_count).
    """
    piece_dir = SHOWCASE_MIDI / piece_id
    cat       = json.loads((piece_dir / 'catalog.json').read_text())
    ts        = cat.get('timeSig', '4/4')
    bpm       = float(cat.get('bpm', 120) or 120)
    num, den  = map(int, ts.split('/'))
    qL_per_bar = num * 4 / den
    DIV = 16  # ticks per quarter note in the compact serialisation

    # Lazy import: write_midi_track lives in pdf_to_midi.py (same directory).
    _script_dir = str(Path(__file__).parent.resolve())
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from pdf_to_midi import write_midi_track

    applied = 0
    skipped = 0

    by_track = defaultdict(list)
    for c in corrections:
        track = (c.get('track') or 'melody').lower()
        if track not in ('melody', 'bass'):
            skipped += 1; continue
        by_track[track].append(c)

    for track_name, track_corrs in by_track.items():
        path = piece_dir / f'{track_name}.mid'
        if not path.exists():
            print(f'[ai] no {path.name} -- skipping {len(track_corrs)} corrections')
            skipped += len(track_corrs); continue

        events, _tpb = _read_midi(path)
        notes = [{'midi': e['midi'], 'start_qL': e['start_qL'], 'end_qL': e['end_qL'],
                  '_swapped': False}
                 for e in events]
        track_applied = 0

        swaps    = [c for c in track_corrs if 'rewrite' not in c]
        rewrites = [c for c in track_corrs if 'rewrite' in c]

        # ── Pitch swaps ──────────────────────────────────────────────────────
        for c in swaps:
            try:
                bar     = int(c['bar'])
                wrong   = name_to_midi(c['wrong_pitch'])
                correct = name_to_midi(c['correct_pitch'])
            except (KeyError, TypeError, ValueError):
                skipped += 1; continue
            if wrong is None or correct is None:
                skipped += 1; continue
            lo, hi = (bar - 1) * qL_per_bar, bar * qL_per_bar
            hit = next((n for n in notes
                        if n['midi'] == wrong and lo <= n['start_qL'] < hi
                        and not n['_swapped']), None)
            if hit is None:
                skipped += 1
                print(f'[ai] could not find {c.get("wrong_pitch")} in bar {bar} of {track_name}')
                continue
            hit['midi'] = correct
            hit['_swapped'] = True
            applied += 1; track_applied += 1

        # ── Bar rewrites ─────────────────────────────────────────────────────
        for c in rewrites:
            try:
                bar = int(c['bar'])
            except (KeyError, TypeError, ValueError):
                skipped += 1; continue
            parsed = _parse_rewrite(c['rewrite'])
            if not parsed:
                skipped += 1
                print(f'[ai] empty/invalid rewrite for bar {bar} of {track_name}')
                continue
            lo, hi = (bar - 1) * qL_per_bar, bar * qL_per_bar
            # Drop existing notes in this bar, then lay the new ones in sequence.
            # A rest (midi is None) advances the cursor without sounding a note.
            # Durations are normalised to fill exactly one measure (the LLM's
            # durations rarely sum exactly), keeping every bar on the grid.
            notes = [n for n in notes if not (lo <= n['start_qL'] < hi)]
            total_qL = sum(t for _m, t in parsed) / DIV
            ratio = total_qL / qL_per_bar if qL_per_bar else 0
            scale = (qL_per_bar / total_qL) if (0.5 <= ratio <= 2.0) else 1.0
            pos = lo
            for midi, ticks in parsed:
                dur_qL = (ticks / DIV) * scale
                if pos + dur_qL > hi + 1e-6:
                    dur_qL = hi - pos
                if dur_qL <= 0:
                    break
                if midi is not None:
                    notes.append({'midi': midi, 'start_qL': pos,
                                  'end_qL': pos + dur_qL, '_swapped': True})
                pos += dur_qL
            applied += 1; track_applied += 1

        if track_applied > 0:
            notes.sort(key=lambda n: n['start_qL'])
            events_sec = [{'note': n['midi'],
                           'start': n['start_qL'] * 60.0 / bpm,
                           'end':   n['end_qL']   * 60.0 / bpm,
                           'vel':   64}
                          for n in notes]
            write_midi_track(events_sec, bpm, ts, str(path),
                             channel=9 if track_name == 'drums' else 0)

    return applied, skipped


def main():
    p = argparse.ArgumentParser(description='AI-correct an OMR-imported piece via browser-ai-bridge')
    p.add_argument('piece_id', help='ID of the piece in showcase-midi/')
    p.add_argument('--provider', default='gemini',
                   help='AI provider via browser-ai-bridge (default gemini; '
                        'also chatgpt|copilot|deepseek|grok)')
    p.add_argument('--max-bars', type=int, default=None,
                   help='Limit serialised bars to first N (smaller prompt; default: all)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print prompt + AI response, do not modify MIDI files')
    p.add_argument('--out-json', metavar='PATH',
                   help='Also write parsed corrections to this JSON path')
    p.add_argument('--apply-json', metavar='PATH',
                   help='Skip the AI call; apply corrections from this JSON file')
    p.add_argument('--reference', metavar='PIECE_ID',
                   help='Use another imported piece as a ground-truth reference '
                        '(e.g. a clean Mutopia MIDI of the same composition). '
                        'AI will align imported notes against the reference.')
    p.add_argument('--start-bar', type=int, default=0,
                   help='Serialise bars starting after this 1-indexed bar number '
                        '(default 0 = start of piece). Use with --max-bars to '
                        'chunk through a long piece in multiple passes.')
    p.add_argument('--chunk-all', action='store_true',
                   help='Walk through the whole piece in N-bar chunks (N = --max-bars '
                        'or 32) and run one AI pass per chunk. Each chunk reuses the '
                        'same --reference if provided. Implies --max-bars defaults to 32.')
    p.add_argument('--align-to-reference', action='store_true',
                   help='Rebuild the piece directly from --reference (rather than '
                        'note-level AI correction). Use when OMR mis-recognised the '
                        'score so badly that correction cannot recover it. Requires '
                        '--reference.')
    p.add_argument('--auto', action='store_true',
                   help='Pick the strategy automatically: if a --reference is given '
                        'and the import note-count diverges from it by more than 40%%, '
                        'rebuild from the reference; otherwise run a chunked AI '
                        'correction pass.')
    p.add_argument('--bar-by-bar', action='store_true',
                   help='Contextual bar-by-bar heal: walk the whole piece, showing '
                        'the AI each bar together with its PREV/NEXT neighbours (and '
                        'the reference bar if --reference is set), and let it rewrite '
                        'only the bars that need fixing. Slower but more thorough '
                        'than --chunk-all. --max-bars sets the window size (default 8).')
    p.add_argument('--visual', action='store_true',
                   help='Visual heal: screenshot the rendered sheet music page '
                        'by page and send each image to the AI, which inspects '
                        'the notation visually and rewrites bars that look wrong. '
                        '--max-bars caps how many pages are sent (default 4). '
                        'Requires the dashboard server running.')
    args = p.parse_args()

    # Skip serialisation + AI call if user supplied pre-computed corrections.
    if args.apply_json:
        try:
            data = json.loads(Path(args.apply_json).read_text(encoding='utf-8'))
        except Exception as e:
            print(f'[ai] cannot read --apply-json: {e}', file=sys.stderr)
            return 2
        corrections = data.get('corrections', data) if isinstance(data, dict) else data
        if not isinstance(corrections, list):
            print(f'[ai] --apply-json: expected list or {{"corrections": [...]}}, '
                  f'got {type(corrections).__name__}', file=sys.stderr)
            return 2
        print(f'[ai] applying {len(corrections)} pre-computed corrections from {args.apply_json}')
        applied, skipped = apply_corrections(args.piece_id, corrections)
        print(f'[ai] applied {applied}, skipped {skipped}')
        return 0

    # Auto mode: decide between align-to-reference and AI correction by
    # comparing the import's note count against the reference's.
    if args.auto:
        if args.reference:
            try:
                _tc, t_mel, t_bas = _load_piece(args.piece_id)
                _rc, r_mel, r_bas = _load_piece(args.reference)
            except Exception as e:
                print(f'[ai] auto: cannot load pieces: {e}', file=sys.stderr)
                return 2
            t_total = len(t_mel) + len(t_bas)
            r_total = len(r_mel) + len(r_bas)
            ratio   = (t_total / r_total) if r_total else 999
            print(f'[ai] auto: import has {t_total} notes, reference {r_total} '
                  f'(ratio {ratio:.2f})')
            if ratio > 1.4 or ratio < 0.7:
                print(f'[ai] auto: import diverges too far from reference '
                      f'-> rebuilding from reference')
                args.align_to_reference = True
            else:
                print(f'[ai] auto: import is close to reference '
                      f'-> running chunked AI correction')
                args.chunk_all = True
        else:
            print('[ai] auto: no reference -> running chunked AI correction')
            args.chunk_all = True

    # Bar-by-bar contextual heal: walk the piece, evaluating each bar with its
    # neighbours (and reference bar) as context.
    if args.bar_by_bar:
        window = args.max_bars or 8
        try:
            catalog, _, _ = serialise_bar_context(args.piece_id, 1, 1,
                                                  reference_id=args.reference)
        except Exception as e:
            print(f'[ai] cannot prepare piece: {e}', file=sys.stderr)
            return 2
        total_bars = catalog.get('bars', 0)
        if not _bridge_ping():
            print(f'[ai] browser-ai-bridge is not reachable at {BRIDGE_URL}.',
                  file=sys.stderr)
            return 3
        print(f'[ai] bar-by-bar heal: {total_bars} bars, window {window}, '
              f'provider {args.provider}'
              + (f', reference {args.reference}' if args.reference else ''))
        total_applied = total_skipped = 0
        start = 1
        win_num = 0
        while start <= total_bars:
            win_num += 1
            try:
                _cat, prompt, (lo, hi) = serialise_bar_context(
                    args.piece_id, start, window, reference_id=args.reference)
            except Exception as e:
                print(f'[ai] window prep failed at bar {start}: {e}', file=sys.stderr)
                break
            print(f'\n[ai] === window {win_num}: bars {lo}..{hi} of {total_bars} ===')
            try:
                response = _bridge_ask(prompt, provider=args.provider)
                corrections = parse_corrections(response)
            except Exception as e:
                print(f'[ai] window failed, skipping: {e}', file=sys.stderr)
                start = hi + 1
                continue
            print(f'[ai] {len(corrections)} bar(s) flagged for correction')
            if corrections:
                applied, skipped = apply_corrections(args.piece_id, corrections)
                total_applied += applied
                total_skipped += skipped
                print(f'[ai] window applied {applied}, skipped {skipped}')
            start = hi + 1
        print(f'\n[ai] TOTAL: rewrote {total_applied} bars, skipped {total_skipped} '
              f'across {win_num} windows')
        return 0

    # Visual heal: screenshot the rendered sheet music and let the AI inspect
    # the notation visually, page by page.
    if args.visual:
        try:
            catalog, _mel, _bass = _load_piece(args.piece_id)
        except Exception as e:
            print(f'[ai] cannot load piece: {e}', file=sys.stderr)
            return 2
        if not _bridge_ping():
            print(f'[ai] browser-ai-bridge is not reachable at {BRIDGE_URL}.',
                  file=sys.stderr)
            return 3
        title    = catalog.get('title', args.piece_id)
        haystack = ' '.join([title, catalog.get('importedFrom', ''),
                             args.piece_id]).lower()
        composer = None
        for needle, name in (('beethoven', 'Beethoven'), ('mozart', 'Mozart'),
                             ('bach', 'Bach'), ('chopin', 'Chopin'),
                             ('schubert', 'Schubert'), ('debussy', 'Debussy'),
                             ('liszt', 'Liszt'), ('pachelbel', 'Pachelbel')):
            if needle in haystack:
                composer = name; break
        max_pages = args.max_bars or 4
        print(f'[ai] visual heal: rendering up to {max_pages} sheet page(s) '
              f'for "{title}"...')
        try:
            pages = capture_sheet_pages(args.piece_id, max_pages)
        except Exception as e:
            print(f'[ai] sheet capture failed: {e}', file=sys.stderr)
            return 2
        print(f'[ai] captured {len(pages)} page image(s); provider {args.provider}')
        total_applied = total_skipped = 0
        for i, page_png in enumerate(pages, 1):
            print(f'\n[ai] === page {i}/{len(pages)}: {page_png} ===')
            prompt = _build_visual_prompt(
                title, composer, catalog.get('timeSig', '4/4'),
                catalog.get('key', 'C'), catalog.get('bpm', 90),
                i, len(pages))
            try:
                response = _bridge_image_ask(page_png, prompt,
                                             provider=args.provider)
                corrections = parse_corrections(response)
            except Exception as e:
                print(f'[ai] page failed, skipping: {e}', file=sys.stderr)
                continue
            print(f'[ai] {len(corrections)} bar(s) flagged for correction')
            if corrections:
                applied, skipped = apply_corrections(args.piece_id, corrections)
                total_applied += applied
                total_skipped += skipped
                print(f'[ai] page applied {applied}, skipped {skipped}')
        print(f'\n[ai] TOTAL: rewrote {total_applied} bars, skipped '
              f'{total_skipped} across {len(pages)} page(s)')
        return 0

    # Align-to-reference mode: rebuild the piece from a clean reference.
    if args.align_to_reference:
        if not args.reference:
            print('[ai] --align-to-reference requires --reference', file=sys.stderr)
            return 2
        try:
            summary = align_to_reference(args.piece_id, args.reference)
        except Exception as e:
            print(f'[ai] align-to-reference failed: {e}', file=sys.stderr)
            return 2
        tracks = ', '.join(f'{k}:{v}' for k, v in summary['tracks'].items())
        print(f'[ai] aligned "{args.piece_id}" to reference "{args.reference}"')
        print(f'[ai] rebuilt tracks [{tracks}], bars={summary["bars"]}, bpm={summary["bpm"]}')
        return 0

    # Chunk-all mode: iterate through the whole piece in fixed-size windows.
    if args.chunk_all:
        chunk_size = args.max_bars or 32
        try:
            catalog, _ = serialise_piece(args.piece_id, max_bars=1)
        except Exception as e:
            print(f'[ai] cannot prepare piece: {e}', file=sys.stderr)
            return 2
        total_bars = catalog.get('bars', 0)
        if not _bridge_ping():
            print(f'[ai] browser-ai-bridge is not reachable at {BRIDGE_URL}.',
                  file=sys.stderr)
            return 3
        total_applied = 0
        total_skipped = 0
        chunk_start = 0
        chunk_num   = 0
        while chunk_start < total_bars:
            chunk_num += 1
            print(f'\n[ai] === chunk {chunk_num}: bars {chunk_start+1}..{chunk_start+chunk_size} of {total_bars} ===')
            try:
                _cat, prompt = serialise_piece(args.piece_id,
                                                max_bars=chunk_size,
                                                reference_id=args.reference,
                                                bar_offset=chunk_start)
            except Exception as e:
                print(f'[ai] chunk prep failed: {e}', file=sys.stderr)
                break
            print(f'[ai] prompt length: {len(prompt)} chars')
            try:
                response = _bridge_ask(prompt, provider=args.provider)
            except Exception as e:
                print(f'[ai] chunk failed, skipping: {e}', file=sys.stderr)
                chunk_start += chunk_size
                continue
            try:
                corrections = parse_corrections(response)
            except Exception as e:
                print(f'[ai] cannot parse response: {e}', file=sys.stderr)
                chunk_start += chunk_size
                continue
            print(f'[ai] {len(corrections)} corrections suggested')
            if corrections:
                applied, skipped = apply_corrections(args.piece_id, corrections)
                total_applied += applied
                total_skipped += skipped
                print(f'[ai] chunk applied {applied}, skipped {skipped}')
            chunk_start += chunk_size
        print(f'\n[ai] TOTAL: applied {total_applied}, skipped {total_skipped} across {chunk_num} chunks')
        return 0

    try:
        catalog, prompt = serialise_piece(args.piece_id,
                                          max_bars=args.max_bars,
                                          reference_id=args.reference,
                                          bar_offset=args.start_bar)
    except Exception as e:
        print(f'[ai] cannot prepare piece: {e}', file=sys.stderr)
        return 2

    print(f'[ai] piece: {catalog.get("title")} ({catalog.get("timeSig")}, '
          f'{catalog.get("bars")} bars, {catalog.get("bpm")} BPM)')
    print(f'[ai] prompt length: {len(prompt)} chars')
    if args.dry_run:
        print('--- PROMPT ---')
        print(prompt)
        print('--- (dry-run: not calling AI) ---')
        return 0

    if not _bridge_ping():
        print(f'[ai] browser-ai-bridge is not reachable at {BRIDGE_URL}.',
              file=sys.stderr)
        print(f'[ai] start it with: cd c:/Users/Work/browser-ai-bridge && '
              f'npm install && npm start', file=sys.stderr)
        return 3

    print(f'[ai] sending to {args.provider}...')
    t0 = time.time()
    try:
        response = _bridge_ask(prompt, provider=args.provider)
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        print(f'[ai] bridge HTTP {e.code}: {body[:500]}', file=sys.stderr)
        return 4
    except Exception as e:
        print(f'[ai] bridge call failed: {e}', file=sys.stderr)
        return 4

    print(f'[ai] response received in {time.time()-t0:.1f}s ({len(response)} chars)')

    try:
        corrections = parse_corrections(response)
    except Exception as e:
        print(f'[ai] failed to parse AI response: {e}', file=sys.stderr)
        print('--- AI response (first 1000 chars) ---', file=sys.stderr)
        print(response[:1000], file=sys.stderr)
        return 5

    print(f'[ai] {len(corrections)} corrections suggested')
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(corrections, indent=2))
        print(f'[ai] wrote corrections to {args.out_json}')

    if not corrections:
        return 0

    applied, skipped = apply_corrections(args.piece_id, corrections)
    print(f'[ai] applied {applied}, skipped {skipped}')
    return 0 if applied > 0 or not corrections else 1


if __name__ == '__main__':
    sys.exit(main())
