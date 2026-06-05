"""
ai_transcribe.py - Hybrid (mechanical + AI) transcription of a sheet-music PDF.

The design splits work by what each side is good at: mechanical code owns
structure (it is fast, deterministic, reliable), the AI owns content (it can
actually read noteheads). The piece is processed one page at a time:

  1. [mechanical] Render the PDF page; split it into system strips at the
     white gaps between staves; detect barline x-positions in each strip.
  2. [AI] Each strip is transcribed by a browser-driven LLM (Gemini) into the
     project's compact note format.
  3. [mechanical] Barline positions crop every bar into its own image; a
     per-bar confidence model scores each transcribed bar on rhythm, density,
     key-fit and octave range.
  4. [AI] Low-confidence bars are re-read FRESH — their high-resolution crops
     are montaged (labelled "BAR N") and sent in one call. The AI reads dense
     music far better from an isolated measure than from a whole strip. This
     loops until the flagged set stops shrinking.
  5. [mechanical+AI] Holistic validation: the original page is shown beside our
     rendered sheet + piano roll; then a synth render is sent so the AI can
     also hear errors. Remaining mismatches are corrected.
  6. [mechanical] Re-normalise; after the last page a rule-based music-theory
     check (theory_check.py) scores the import.

Page images and a page->bar manifest are written to <out_dir>/_pages/.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path


def _log(msg):
    """Timestamped progress log so a long-running import is never opaque."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ai_correct  # noqa: E402  -- shared bridge + note-parsing helpers
import ai_engine   # noqa: E402  -- pluggable transcription engine (bridge|claude)

DIV = 16  # ticks per quarter note in the compact "C5(q)" note format
NODE_BIN = ai_correct.NODE_BIN
DASHBOARD = ai_correct.DASHBOARD
REFINE_ROUNDS  = 3  # confidence-routed per-bar re-read cycles per page
VALIDATE_ROUNDS = 1  # holistic sheet/roll validation rounds per page
REFINE_CHUNK   = 8  # max bar crops per montage (one AI call)


# ── PDF + rendering helpers ───────────────────────────────────────────────────

def _render_pdf_pages(pdf_path, pages_dir, dpi=300, want_pages=None):
    """Render PDF pages to pages_dir/page_NN.png. Returns one PNG path per page
    in the document (in order).

    want_pages: optional set of 1-based page numbers to actually rasterise.
      Pages outside it are still listed (so page counts and manifests stay
      correct) but not rendered. A bar-limit preview or page-range import only
      ever reads its in-scope pages, and rasterising every page of a long score
      at 300 DPI up front was the main reason a "first 2 bars" request crawled.
      None = render every page.
    """
    import fitz  # PyMuPDF
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    out = []
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for i in range(doc.page_count):
        n = i + 1
        png = pages_dir / f'page_{n:02d}.png'
        out.append(png)
        if want_pages is None or n in want_pages:
            doc[i].get_pixmap(matrix=mat, alpha=False).save(str(png))
    doc.close()
    return out


def _detect_systems(png_path):
    """Detect grand-staff systems (treble+bass pairs) on a sheet-music page.

    Robust to scan quality (faint, broken or tightly-packed staff lines):
    works from the *periodicity* of the horizontal ink profile, not from
    detecting individual staff lines.

      1. Autocorrelate the ink-per-row profile — its dominant period P is the
         system-to-system spacing (engravers lay systems out evenly).
      2. Phase-lock that period grid to the page: the offset whose period
         multiples land on the most ink. Title/tempo text does not align to
         the music period, so those grid points score low and are dropped.
      3. Each surviving grid point is a system centre; the band around it is
         ~one period tall.

    Returns a list of (top, bottom) pixel bands, one per system, or None when
    detection is inconclusive (caller falls back to an even split).
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(png_path).convert('L')
    W, H = img.size
    px = img.load()
    if H < 600:
        return None
    step = max(1, W // 300)
    cols = list(range(0, W, step))
    ncol = len(cols)
    ink = [sum(1 for x in cols if px[x, y] < 140) / ncol for y in range(H)]
    sig = [y for y in range(H) if ink[y] > 0.05]
    if len(sig) < 200:
        return None
    top, bot = sig[0], sig[-1]
    seg = ink[top:bot + 1]
    n = len(seg)
    if n < 600:
        return None
    mean = sum(seg) / n
    d = [v - mean for v in seg]
    # Dominant vertical period = system spacing.
    best = (0.0, 0)
    for lag in range(250, min(950, n // 2)):
        c = sum(d[i] * d[i + lag] for i in range(0, n - lag, 2))
        if c > best[0]:
            best = (c, lag)
    period = best[1]
    if period < 250:
        return None
    # Smoothed ink profile — how 'system-like' each row is.
    win = max(20, period // 3)
    sm, acc = [0.0] * H, 0.0
    for y in range(H):
        acc += ink[y]
        if y >= win:
            acc -= ink[y - win]
        sm[y] = acc / min(y + 1, win)
    mx = max(sm) or 1.0
    # Phase-lock the period grid to the system centres.
    best_ph = (-1.0, 0)
    for ph in range(0, period, 4):
        centres = [top + ph + k * period for k in range(40)
                   if top + ph + k * period <= bot]
        score = sum(sm[c] for c in centres)
        if score > best_ph[0]:
            best_ph = (score, ph)
    centres = [top + best_ph[1] + k * period for k in range(40)
               if top + best_ph[1] + k * period <= bot]
    centres = [c for c in centres if sm[c] > 0.32 * mx]   # drop title/footer
    if len(centres) < 2:
        return None
    half = int(period * 0.46)
    return [(max(0, c - half), min(H, c + half)) for c in centres]


def _split_page_into_systems(png_path, out_dir, page_num, target_strips=6):
    """Split a sheet-music page into one strip per grand-staff system.

    Detecting real systems (rather than cutting the page into N even slices)
    means each strip holds exactly one line of music — which is what the
    per-bar crop + refinement pass relies on. Falls back to an even split at
    low-ink rows when staff detection is inconclusive.
    Returns a list of strip PNG paths.
    """
    try:
        from PIL import Image
    except ImportError:
        return [Path(png_path)]
    img = Image.open(png_path).convert('L')
    W, H = img.size
    px = img.load()

    bands = _detect_systems(png_path)
    if bands:
        strips = []
        for idx, (top, bot) in enumerate(bands):
            if bot - top < H * 0.02:
                continue
            sp = Path(out_dir) / f'page_{page_num:02d}_sys_{idx + 1:02d}.png'
            img.crop((0, max(0, top), W, min(H, bot))).save(str(sp))
            strips.append(sp)
        if strips:
            return strips

    # ── Fallback: even split snapped to low-ink rows ────────────────────────
    step = max(1, W // 400)
    ink = [sum(1 for x in range(0, W, step) if px[x, y] < 150) for y in range(H)]
    if not ink or max(ink) == 0 or H < 400:
        return [Path(png_path)]

    n = max(1, target_strips)
    cuts = [0]
    for k in range(1, n):
        target_y = k * H // n
        reach = H // (n * 2)
        lo = max(1, target_y - reach)
        hi = min(H - 1, target_y + reach)
        cuts.append(min(range(lo, hi), key=lambda y: ink[y]))
    cuts.append(H)
    cuts = sorted(set(cuts))

    pad = max(4, H // 280)
    strips = []
    for idx in range(len(cuts) - 1):
        top = max(0, cuts[idx] - pad)
        bot = min(H, cuts[idx + 1] + pad)
        if bot - top < H * 0.03:  # skip slivers
            continue
        sp = Path(out_dir) / f'page_{page_num:02d}_sys_{idx + 1:02d}.png'
        img.crop((0, top, W, bot)).save(str(sp))
        strips.append(sp)
    return strips or [Path(png_path)]


def _detect_barline_positions(strip_png):
    """Find barline x-positions in a system strip via vertical projection.

    A barline is a tall vertical ink run spanning the staff; note stems are
    much shorter. Returns a sorted list of barline-group centre x-positions
    (the system's left edge and every barline), or None if inconclusive.
    The bars of the system lie between consecutive positions.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(strip_png).convert('L')
    W, H = img.size
    px = img.load()
    runs = []
    for x in range(W):
        run = best = 0
        for y in range(H):
            if px[x, y] < 110:
                run += 1
                best = best if best > run else run
            else:
                run = 0
        runs.append(best)
    peak = max(runs) if runs else 0
    if peak < 80:                       # no staff-height verticals — no music
        return None
    thr = max(peak * 0.62, 90)
    cols = [x for x in range(W) if runs[x] >= thr]
    if not cols:
        return None
    # Group adjacent tall-ink columns (a barline is a few px wide) and take
    # each group's centre.
    groups, cur = [], [cols[0]]
    for a, b in zip(cols, cols[1:]):
        if b - a > 12:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    centres = [sum(g) // len(g) for g in groups]
    # Merge only centres a hair apart — a barline drawn thick or doubled spans
    # ~20-30px, whereas the narrowest real measure is far wider. (Earlier this
    # used 8% of the width, which wrongly swallowed close volta measures.)
    min_gap = max(35, int(W * 0.012))
    merged = [centres[0]]
    for c in centres[1:]:
        if c - merged[-1] < min_gap:
            merged[-1] = (merged[-1] + c) // 2
        else:
            merged.append(c)
    if len(merged) < 2:
        return None
    # Recover missed barlines: a measure far wider than the typical one almost
    # always hides a faint barline the projection didn't catch. Split any gap
    # above ~1.7x the median into equal pieces so the measure count is right.
    gaps = [merged[i + 1] - merged[i] for i in range(len(merged) - 1)]
    med = sorted(gaps)[len(gaps) // 2]
    out = [merged[0]]
    for i, g in enumerate(gaps):
        if med > 0 and g > med * 1.7:
            parts = int(round(g / med))
            for k in range(1, parts):
                out.append(merged[i] + g * k // parts)
        out.append(merged[i + 1])
    return out


def _detect_barlines(strip_png):
    """Estimated measure count of a system strip, or None (a soft prompt hint).
    N barline groups delimit N-1 measures."""
    pos = _detect_barline_positions(strip_png)
    if not pos:
        return None
    bars = len(pos) - 1
    return bars if bars >= 1 else None


def _bar_crops(strip_png, expected_bars, out_dir, page_num, sys_num):
    """Crop a system strip into exactly `expected_bars` per-measure images.

    Mechanical bar segmentation: barline positions split the strip into bars;
    if the detected barline count disagrees with `expected_bars` (the count the
    AI transcribed) the music span is divided evenly instead. Each crop is the
    full strip height (grand staff) between two barlines, lightly padded.
    Returns a list of crop Paths of length `expected_bars`, or None on failure.
    """
    if expected_bars < 1:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(strip_png).convert('RGB')
    except Exception:
        return None
    W, H = img.size
    pos = _detect_barline_positions(strip_png)
    if pos and len(pos) - 1 == expected_bars:
        bounds = [float(p) for p in pos]
    else:
        # Counts disagree (or no barlines) — divide the music span evenly.
        x0, x1 = (pos[0], pos[-1]) if pos else (0.0, float(W))
        bounds = [x0 + (x1 - x0) * k / expected_bars
                  for k in range(expected_bars + 1)]
    padx = max(8, W // 120)
    out_dir = Path(out_dir)
    crops = []
    for k in range(expected_bars):
        left = max(0, int(bounds[k]) - padx)
        right = min(W, int(bounds[k + 1]) + padx)
        if right - left < 4:
            right = min(W, left + 4)
        cp = out_dir / f'bar_{page_num:02d}_{sys_num:02d}_{k + 1:02d}.png'
        try:
            img.crop((left, 0, right, H)).save(str(cp))
        except Exception:
            return None
        crops.append(cp)
    return crops


def _build_montage(items, out_png):
    """Stack labelled per-bar crops into one image for a single refine call.

    `items` = [(bar_no, crop_path)]. Each crop gets a dark label band ("BAR N")
    directly above it so the AI can map every measure to its bar number in one
    shot. Returns out_png, or None.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    LABEL_H, MAXW = 44, 1500
    font = None
    for fp in ('arialbd.ttf', 'arial.ttf', 'C:/Windows/Fonts/arial.ttf',
               'DejaVuSans-Bold.ttf'):
        try:
            font = ImageFont.truetype(fp, 30)
            break
        except Exception:
            continue
    tiles = []
    for bar_no, cp in items:
        try:
            im = Image.open(cp).convert('RGB')
        except Exception:
            continue
        if im.width > MAXW:
            im = im.resize((MAXW, max(1, int(im.height * MAXW / im.width))))
        tiles.append((bar_no, im))
    if not tiles:
        return None
    W = max(im.width for _b, im in tiles)
    H = sum(im.height + LABEL_H for _b, im in tiles)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for bar_no, im in tiles:
        draw.rectangle([0, y, W, y + LABEL_H], fill=(28, 28, 40))
        draw.text((12, y + 7), f'BAR {bar_no}', fill=(255, 255, 255), font=font)
        y += LABEL_H
        canvas.paste(im, (0, y))
        y += im.height
    try:
        canvas.save(str(out_png))
    except Exception:
        return None
    return out_png


def _render_piece_sheet(piece_id, max_pages=8):
    """Render the current state of a piece to sheet-music PNGs via the
    dashboard (ss_sheet_pages.mjs). Returns a list of PNG paths (may be empty
    if the dashboard server is not running)."""
    script = DASHBOARD / 'ss_sheet_pages.mjs'
    if not script.exists():
        return []
    try:
        proc = subprocess.run([NODE_BIN, str(script), piece_id, str(max_pages)],
                              capture_output=True, text=True, timeout=180)
    except Exception as e:
        print(f'[transcribe] sheet render failed: {e}', file=sys.stderr)
        return []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line).get('pages', [])
            except json.JSONDecodeError:
                pass
    return []


def _stack_vertical(pngs, out_png):
    """Stack images top-to-bottom into out_png. Returns out_png or None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    imgs = []
    for p in pngs:
        try:
            imgs.append(Image.open(p).convert('RGB'))
        except Exception:
            pass
    if not imgs:
        return None
    w = max(im.width for im in imgs)
    canvas = Image.new('RGB', (w, sum(im.height for im in imgs)), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(str(out_png))
    return out_png


def _render_piano_roll(batch_bars, qL_per_bar, out_png):
    """Draw a piano-roll image of the transcribed batch (mechanical, no AI).

    Time runs left-to-right, pitch bottom-to-top; melody notes are blue, bass
    green, barlines grey. Repeated figures, octave jumps and rhythm patterns
    stand out here in a way they don't in notation, which helps the LLM spot
    transcription errors during validation.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    mel, bas = [], []
    for idx, bar in enumerate(batch_bars):
        bs = idx * qL_per_bar
        for m, s, e in _bar_to_events(bar.get('melody', ''), bs, qL_per_bar):
            mel.append((m, s, e))
        for m, s, e in _bar_to_events(bar.get('bass', ''), bs, qL_per_bar):
            bas.append((m, s, e))
    notes = mel + bas
    if not notes:
        return None
    total_qL = max(1e-6, len(batch_bars) * qL_per_bar)
    pmin = min(m for m, _s, _e in notes) - 2
    pmax = max(m for m, _s, _e in notes) + 2
    ROW = 9
    pxq = min(48, max(8, 3600 / total_qL))
    W = max(400, int(total_qL * pxq))
    H = (pmax - pmin + 1) * ROW
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = lambda m: int((pmax - m) * ROW)
    for m in range(pmin, pmax + 1):                       # octave gridlines
        if m % 12 == 0:
            d.line([(0, y(m)), (W, y(m))], fill=(225, 225, 225))
    for b in range(len(batch_bars) + 1):                  # barlines
        x = int(b * qL_per_bar * pxq)
        d.line([(x, 0), (x, H)], fill=(170, 170, 195))
    for m, s, e in bas:
        d.rectangle([s * pxq, y(m), e * pxq - 1, y(m) + ROW - 1], fill=(70, 150, 95))
    for m, s, e in mel:
        d.rectangle([s * pxq, y(m), e * pxq - 1, y(m) + ROW - 1], fill=(55, 95, 210))
    img.save(str(out_png))
    return out_png


def _render_batch_sheet(batch_bars, meta):
    """Render just this batch's bars to sheet music via a throwaway piece.
    Returns a list of PNG paths (the PNGs live in screenshots/heal/, so they
    survive the temp piece being deleted)."""
    import shutil
    tmp_id = '_tmpbatch'
    tmp_dir = ai_correct.SHOWCASE_MIDI / tmp_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        tmp_meta = dict(meta)
        tmp_meta['title'] = 'batch'
        _write_piece(tmp_id, tmp_dir, batch_bars, tmp_meta)
        return _render_piece_sheet(tmp_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _render_batch_audio(batch_bars, meta, out_wav):
    """Synthesize this batch's bars to a WAV so the AI can audit it by ear.

    Renders melody + bass through the project synth (the same engine that
    produces showcase audio). Returns the WAV path on success, or None when
    audio rendering is unavailable.
    """
    try:
        from pdf_to_midi import write_midi_track
        import midi_synth
    except Exception as e:
        _log(f'audio render unavailable: {e}')
        return None
    num, den = map(int, str(meta['timeSig']).split('/'))
    qL_per_bar = num * 4 / den
    bpm = meta['bpm'] or 100
    tracks = _mechanical_cleanup(_bars_to_tracks(batch_bars, qL_per_bar, bpm))
    # Absolute path — the bridge resolves it from its own cwd, not ours.
    out_wav = Path(out_wav).resolve()
    stems = {}
    for role in ('melody', 'bass', 'pad', 'drums'):
        p = out_wav.parent / f'_audio_{role}.mid'
        write_midi_track(tracks[role], bpm, meta['timeSig'], p,
                         channel=9 if role == 'drums' else 0)
        stems[role] = p
    try:
        written = midi_synth.synth_piece(
            str(stems['bass']), str(stems['pad']), str(stems['melody']),
            str(stems['drums']), str(out_wav), style='Classical')
        return Path(written) if written else None
    except Exception as e:
        _log(f'batch audio render failed: {e}')
        return None


def _composite(left_png, right_png, out_png):
    """Place two images side by side (scaled to a common height) into out_png.
    Returns out_png on success, or None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        a = Image.open(left_png).convert('RGB')
        b = Image.open(right_png).convert('RGB')
    except Exception:
        return None
    h = 1100
    aw = max(1, int(a.width * h / a.height))
    bw = max(1, int(b.width * h / b.height))
    a = a.resize((aw, h))
    b = b.resize((bw, h))
    gap = 24
    canvas = Image.new('RGB', (aw + bw + gap, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (aw + gap, 0))
    canvas.save(str(out_png))
    return out_png


# ── Prompts ───────────────────────────────────────────────────────────────────

def _transcribe_prompt(title, composer, page_num, total_pages, bar_hint=None,
                       strict=False, time_sig=None, key=None):
    hint = ''
    if bar_hint and strict:
        hint = (f'\nIMPORTANT: a barline scan of this slice counted EXACTLY '
                f'{bar_hint} measures. Your "bars" list MUST contain exactly '
                f'{bar_hint} entries — re-check your bar splitting.\n')
    elif bar_hint:
        hint = (f'\nA barline scan suggests this slice contains roughly '
                f'{bar_hint} measure(s) — count the barlines yourself to '
                f'confirm, and make sure your "bars" list has that many.\n')

    # When the meter/key are known up front (user-supplied), state them as
    # ground truth — a misread time signature is the single biggest source of
    # error: it makes the model pad a pickup with rests and overfill bars.
    if time_sig:
        try:
            _n, _d = map(int, str(time_sig).split('/'))
            _beat = {1: 'whole', 2: 'half', 4: 'quarter', 8: 'eighth',
                     16: 'sixteenth'}.get(_d, f'1/{_d}')
            meter_line = (
                f'\nTHE TIME SIGNATURE IS {time_sig} (given — do not change it). '
                f'Every full bar holds exactly {_n} {_beat} note'
                f'{"s" if _n != 1 else ""} of value, in BOTH the melody and the '
                f'bass. Set "timeSig" to "{time_sig}" and make every full bar sum '
                f'to exactly that.\n')
        except Exception:
            meter_line = f'\nThe time signature is {time_sig}. Use it.\n'
    else:
        meter_line = (
            '\nREAD THE TIME SIGNATURE CAREFULLY from the start of the first '
            'system (e.g. 3/8, 6/8, 2/4, 3/4, 4/4) — a wrong meter corrupts '
            'every bar. Many pieces (e.g. Für Elise) are in 3/8, not 4/4.\n')
    key_line = (f'The key is {key}.\n' if key else '')

    return f"""You are an expert music engraver transcribing printed sheet music.

The attached image is a horizontal slice (one or two systems / lines) from
page {page_num} of {total_pages} of "{title}" by {composer}. It may also be a
title/header strip with no music.
{hint}{meter_line}{key_line}
"{title}" by {composer} is a well-known piece. Use your knowledge of the
canonical score to resolve genuinely ambiguous or smudged notes — but always
transcribe what is actually printed, not what you remember, when they differ.

Transcribe EVERY bar visible in this slice. For each bar give the upper-staff
(treble / melody) notes and the lower-staff (bass) notes. If the slice has no
music, return an empty "bars" list.

Respond with JSON ONLY (no prose, no markdown fences):
{{
  "key": "C major",
  "timeSig": "4/4",
  "bpm": 120,
  "bars": [
    {{"melody": "E5(8) D#5(8) E5(8) D#5(8) E5(8) B4(8)", "bass": "(empty)"}},
    {{"melody": "...", "bass": "A2(16) E3(16) A3(8)", "melody2": "", "bass2": ""}}
  ]
}}

Notation rules:
  - Scientific pitch: C4 = middle C, accidentals as # or b (e.g. F#5, Bb3).
  - Duration tag in parentheses after every note: w=whole h=half q=quarter
    8=eighth 16=sixteenth 32=32nd; a trailing dot means dotted (q. 8.).
  - RESTS ARE REQUIRED. Write a rest as R(dur), e.g. R(q) R(8) R(16). When a
    note is held or a staff is silent for part of a bar, fill the remaining
    time with rests — do not just omit it.
  - CRITICAL: every FULL bar's melody string AND bass string must each add up
    to exactly one measure of the time signature. In 3/8 a bar totals three
    eighths (= six sixteenths); in 4/4 it totals four quarters. Use rests only
    to fill genuine silences WITHIN the bar — never add notes or rests beyond
    what is printed just to pad a bar. A bar that sums to MORE than one measure
    (e.g. an extra half/whole note tacked on the end) is an error: transcribe
    only what is printed.
  - PICKUP / ANACRUSIS: if the piece opens with a short incomplete bar before
    the first full barline (Für Elise opens with just two sixteenths, "E5 D#5"),
    transcribe ONLY those few printed notes as a short first bar. Do NOT add
    leading rests to pad it to a full measure, and do NOT merge it into bar 2.
    If the other staff has nothing printed during that pickup (e.g. the left
    hand has not entered yet), write that staff as exactly "(empty)" — never as
    a rest like "R(8)".
  - The "melody" and "bass" in one bar entry are the SAME measure, vertically
    aligned by the barline — do not pair one bar's treble with another's bass.
  - VOICING: if a staff has TWO independent simultaneous voices (e.g. a held
    melody note over a moving inner line, or stems pointing both up and down),
    do NOT cram them into one impossible string. Put the upper / stems-up voice
    in "melody" (treble) or "bass" (bass clef), and the lower / stems-down voice
    in "melody2" / "bass2". EACH voice must independently sum to a full measure
    (pad that voice with rests). Leave "melody2"/"bass2" as "" when the staff is
    a single voice (the usual case) — e.g. Für Elise's opening run is one voice,
    but a bar where the right hand sustains a note while playing faster notes
    beneath needs melody + melody2.
  - List notes left-to-right in time order; for a chord (notes struck together
    in ONE voice) list its notes in order. Two different RHYTHMS at once = two
    voices (use melody2/bass2), not a chord.
  - Use "(empty)" only for a staff that is silent for the WHOLE bar.
  - "key"/"timeSig"/"bpm" describe the piece; estimate bpm from the tempo
    marking. Only the FIRST page's values are used.
  - One entry in "bars" per printed bar on the page, in reading order.
  - Write the JSON directly in your reply. Do NOT open a Canvas or code
    panel — keep the whole answer inline in the chat.
  - Output JSON only."""


def _meter_key_prompt(title, composer):
    """A tiny, focused question: read ONLY the meter + key off the first staff.
    A narrow task is far more reliable than meter-as-afterthought inside a full
    transcription, so we establish it once up front and lock it."""
    return f"""You are reading the very beginning of printed sheet music.

The attached image is the first system (top line) of "{title}" by {composer}.

Look ONLY at the start of the staff, right after the clef, and report:
  - "timeSig": the TIME SIGNATURE printed there, as "n/d" (e.g. "3/8", "6/8",
    "2/4", "3/4", "4/4"; a common-time C = "4/4", cut-time = "2/2"). Look
    carefully — a small "3/8" is easy to misread as "4/4".
  - "key": the KEY implied by the key signature (count the sharps/flats), as
    e.g. "A minor", "C major", "G major", "F major". If a piece is famous (e.g.
    Für Elise is A minor, 3/8) use that to break ties, but trust the print.
  - "bpm": tempo in beats/min if a metronome or tempo word is printed, else null.

Respond with JSON ONLY, no prose, no code fences:
{{"timeSig": "3/8", "key": "A minor", "bpm": null}}"""


def _detect_meter_key(strip_png, title, composer, provider, engine=None):
    """One focused engine call → (timeSig|None, key|None, bpm|None)."""
    try:
        data = _parse_json(ai_engine.image_ask(
            strip_png, _meter_key_prompt(title, composer),
            engine=engine, provider=provider, label='detect-meter-key'))
    except Exception as e:
        _log(f'meter/key detection failed ({e}); falling back to in-pass reads')
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None
    ts = _sane_timesig(data.get('timeSig'), None)
    key = (str(data.get('key')).strip() or None) if data.get('key') else None
    try:
        bpm = int(round(float(data.get('bpm'))))
    except (TypeError, ValueError):
        bpm = None
    return ts, key, bpm


def _validate_prompt(title, composer, page_num, lo, hi, key=None, flags=None):
    keyline = (f'\nThe piece is in {key}. A note far outside that key is usually '
               f'either a deliberate accidental that IS printed (keep it) or a '
               f'misread — check the original carefully for these.\n'
               if key else '')
    flagline = ''
    if flags:
        flagline = ('\nMECHANICAL ANALYSIS — a rule-based check flagged these '
                    'bars as statistically suspect; verify them especially '
                    'closely against the original:\n  '
                    + '\n  '.join(flags) + '\n')
    return f"""You are auditing an automatic transcription of sheet music.

The attached image has a LEFT half and a RIGHT half:
  LEFT  = page {page_num} of the ORIGINAL printed score of "{title}" by {composer}.
  RIGHT = our transcription of those bars, shown two ways stacked vertically:
          (top) rendered sheet music, and
          (bottom) a PIANO ROLL — time left-to-right, pitch bottom-to-top,
          melody in blue, bass in green, grey barlines.
{keyline}{flagline}
Compare our transcription to the original. Use the sheet music to check
pitches and rhythm, and use the piano roll to spot PATTERN errors — a note
that jumps an octave out of an otherwise smooth line, a repeated figure that
breaks, a bar that looks rhythmically unlike its neighbours.

The bars on the right are numbered {lo}-{hi}. For every bar that does not match
the LEFT original (wrong pitches, wrong octave, wrong rhythm, missing or extra
notes, broken pattern), give a corrected rewrite of that bar.

Respond with JSON ONLY (no prose, no markdown fences):
{{
  "ok": true | false,
  "corrections": [
    {{"track": "melody", "bar": N, "rewrite": "C5(q) E5(8) ...", "reason": "..."}},
    {{"track": "bass",   "bar": M, "rewrite": "C3(8) ...",        "reason": "..."}}
  ]
}}

Notation: scientific pitch (C4 = middle C); duration tags w/h/q/8/16/32, a
trailing dot = dotted; rests are R(dur) e.g. R(8). Each "rewrite" must be the
COMPLETE bar and must sum to exactly one full measure of the time signature —
use rests to fill held notes and silences. (A short pickup/anacrusis bar at
the very start, or its complement at the very end, is the exception — leave it
short; do NOT pad it with rests.) "ok" is true only if every bar
already matches; then "corrections" must be empty. Write the JSON directly in
your reply — do NOT open a Canvas or code panel. Output JSON only."""


def _refine_prompt(title, composer, key, timesig, flagged):
    """Prompt for the confidence-routed refinement pass — a montage of
    cropped single measures, each to be re-transcribed fresh."""
    reasons_block = '\n'.join(
        f'  BAR {bn}: ' + '; '.join(rs)
        for bn, rs in sorted(flagged.items()))
    keyline = f' The piece is in {key}.' if key else ''
    return f"""You are correcting specific measures of a music transcription.

The attached image is a vertical stack of CROPPED single measures from the
printed score of "{title}" by {composer}. Each crop has a dark label band
directly above it (e.g. "BAR 7"). Every crop shows exactly ONE measure of a
grand staff:
  - the UPPER staff is the melody (treble clef),
  - the LOWER staff is the bass (bass clef).

The music is in {timesig} time.{keyline} Transcribe each labelled measure
FRESH and carefully from its own crop — do NOT assume any earlier
transcription was correct. A mechanical check flagged these bars as likely
wrong, for the reasons given:
{reasons_block}

Respond with JSON ONLY (no prose, no markdown fences):
{{
  "bars": [
    {{"bar": 7,  "melody": "C5(8) E5(8) G5(q)", "bass": "C3(h)"}},
    {{"bar": 12, "melody": "...",               "bass": "..."}}
  ]
}}

Include one entry for EVERY labelled bar. Notation: scientific pitch
(C4 = middle C), accidentals as # or b; a duration tag follows every note —
w=whole h=half q=quarter 8=eighth 16=sixteenth 32=32nd, a trailing dot = dotted
(q. 8.); rests are R(dur) e.g. R(8). Each "melody" and "bass" string must sum
to exactly one full {timesig} measure — use rests to fill held notes and
silences. (A short pickup/anacrusis bar at the very start of the piece is the
exception — leave it short, don't pad it.) If a staff is empty in a crop, use
"(empty)". Write the JSON directly in your reply — do NOT open a Canvas or code
panel. Output JSON only."""


def _audio_validate_prompt(title, composer, page_num, lo, hi, key, timesig, bpm):
    keyline = f' in the key of {key}' if key else ''
    return f"""You are auditing an automatic transcription of sheet music BY EAR.

The attached AUDIO is our transcription of bars {lo}-{hi} from page {page_num}
of "{title}" by {composer}. It is a plain piano-like synth render of melody
and bass only — {bpm} BPM, {timesig}{keyline}. The bars play strictly in
order, evenly spaced, starting at bar {lo}.

Listen for transcription ERRORS — moments that sound WRONG, not stylistic
choices:
  - a note that clashes harshly or is clearly out of key / out of harmony
  - a note an octave too high or too low that breaks an otherwise smooth line
  - a rhythm that stumbles — a bar that runs long, short, or lurches
  - a repeated figure that breaks on one of its repetitions

Be CONSERVATIVE: only flag a bar when the error is clearly audible. Because
bars are evenly spaced from bar {lo}, you can count to locate the bar number.
For every bar that sounds wrong, give a corrected rewrite of the whole bar.

Respond with JSON ONLY (no prose, no markdown fences):
{{
  "ok": true | false,
  "corrections": [
    {{"track": "melody", "bar": N, "rewrite": "C5(q) E5(8) ...", "reason": "..."}},
    {{"track": "bass",   "bar": M, "rewrite": "C3(8) ...",        "reason": "..."}}
  ]
}}

Notation: scientific pitch (C4 = middle C); duration tags w/h/q/8/16/32, a
trailing dot = dotted; rests are R(dur) e.g. R(8). Each "rewrite" must be the
COMPLETE bar and must sum to exactly one full measure of the time signature.
"ok" is true only if every bar sounds correct; then "corrections" must be
empty. Write the JSON directly in your reply — do NOT open a Canvas or code
panel. Output JSON only."""


def _parse_json(response):
    txt = response.strip()
    if txt.startswith('```'):
        txt = re.sub(r'^```(?:json)?\s*\n', '', txt)
        txt = re.sub(r'\n?```\s*$', '', txt)
    m = re.search(r'\{[\s\S]*\}', txt)
    if not m:
        raise ValueError(f'no JSON object in response: {txt[:200]}')
    return json.loads(m.group(0))


# ── Note assembly ─────────────────────────────────────────────────────────────

def _bar_to_events(note_str, bar_start_qL, qL_per_bar,
                   allow_short=False, align_end=False):
    """Compact bar string ('C5(q) R(8) E5(8)') -> [(midi, start_qL, end_qL)].

    The LLM's note durations rarely add up to a full measure exactly. Rather
    than proportionally scaling every note (which turns clean 8ths/16ths into
    ugly values), the residual is absorbed only at the END of the bar — the
    last note is extended or trimmed — keeping every other note's printed
    duration intact. Proportional scaling is the fallback for gross mismatch.

    allow_short: this bar may be a pickup/anacrusis (first bar) or its
      complement (last bar) — keep its printed durations as-is, never stretch
      a short bar to fill the measure.
    align_end: place a short pickup's notes at the END of the bar so they lead
      into the next downbeat (the musical position of an anacrusis).
    """
    if not note_str or str(note_str).strip().lower() in ('(empty)', 'empty', ''):
        return []
    parsed = ai_correct._parse_rewrite(note_str)  # [(midi, ticks@div16), ...]
    if not parsed:
        return []
    bar_ticks = qL_per_bar * DIV
    total = sum(t for _m, t in parsed)
    if total <= 0:
        return []
    ratio = total / bar_ticks
    pickup = allow_short and ratio < 0.95
    start_qL = bar_start_qL
    if pickup:
        # Keep printed durations; optionally right-align as a lead-in.
        if align_end:
            start_qL = bar_start_qL + max(0.0, qL_per_bar - total / DIV)
    elif ratio < 0.5 or ratio > 2.0:
        # Gross mismatch — fall back to proportional scaling.
        scale = bar_ticks / total
        parsed = [(m, t * scale) for m, t in parsed]
    else:
        # Absorb the small residual at the end: extend or trim the last items.
        diff = bar_ticks - total
        if diff > 0:
            m, t = parsed[-1]
            parsed[-1] = (m, t + diff)
        elif diff < 0:
            over = -diff
            while over > 1e-6 and parsed:
                m, t = parsed[-1]
                if t > over:
                    parsed[-1] = (m, t - over)
                    over = 0
                else:
                    over -= t
                    parsed.pop()
    out, pos = [], start_qL
    bar_end = bar_start_qL + qL_per_bar
    for midi, ticks in parsed:
        dur = ticks / DIV
        if pos + dur > bar_end + 1e-6:
            dur = bar_end - pos
        if dur <= 0:
            break
        # A rest (midi is None) advances the cursor without sounding a note;
        # a sub-64th sliver is a transcription artifact — skip it but keep the
        # cursor moving so following notes stay on the grid.
        if midi is not None and dur >= 0.04:
            out.append((midi, pos, pos + dur))
        pos += dur
    return out


def _bars_to_tracks(all_bars, qL_per_bar, bpm):
    """Build {'melody':[...sec events], 'bass':[...], 'pad':[], 'drums':[]}."""
    tracks = {'melody': [], 'bass': [], 'pad': [], 'drums': []}
    to_sec = lambda qL: qL * 60.0 / bpm
    n = len(all_bars)
    for idx, bar in enumerate(all_bars):
        bar_start = idx * qL_per_bar
        # First/last bar may be a pickup; the first one right-aligns as a
        # lead-in to bar 2's downbeat.
        allow_short = (idx == 0 or idx == n - 1)
        align_end = (idx == 0)
        # Each staff may carry a second voice (melody2/bass2); both sound on the
        # same track so playback/MIDI includes the inner voice.
        for role, field in (('melody', 'melody'), ('melody', 'melody2'),
                            ('bass', 'bass'), ('bass', 'bass2')):
            for midi, s, e in _bar_to_events(bar.get(field, ''), bar_start,
                                             qL_per_bar, allow_short=allow_short,
                                             align_end=align_end):
                tracks[role].append({'note': midi, 'start': to_sec(s),
                                     'end': to_sec(e), 'vel': 72})
    return tracks


# ── Mechanical cleanup ─────────────────────────────────────────────────────────
# Rule-based post-processing of the AI transcription. The LLM reads the score
# well but makes systematic, mechanically-fixable slips: notes an octave off,
# the same note transcribed twice. These passes don't need the LLM.

_TREBLE_BAND = (48, 100)   # C3 .. E7  — plausible MIDI range for the melody staff
_BASS_BAND   = (24, 67)    # C1 .. G4  — plausible MIDI range for the bass staff


def _octave_fix(events, band):
    """Pull octave-outlier notes back toward the body of the track, then clamp
    into the staff's plausible band. Catches the LLM's octave slips."""
    if not events:
        return events
    lo, hi = band
    pitches = sorted(e['note'] for e in events)
    median = pitches[len(pitches) // 2]
    for e in events:
        for _ in range(3):
            if e['note'] - median > 13:
                e['note'] -= 12
            elif median - e['note'] > 13:
                e['note'] += 12
            else:
                break
        while e['note'] > hi:
            e['note'] -= 12
        while e['note'] < lo:
            e['note'] += 12
    return events


def _dedup(events):
    """Drop notes that duplicate another note at (almost) the same time."""
    seen, out = set(), []
    for e in sorted(events, key=lambda x: (x['start'], x['note'])):
        key = (round(e['start'] * 32), e['note'])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _smooth_octaves(events):
    """Fix a single note that jumped a full octave out of an otherwise smooth
    line. Strict on purpose so genuine melodic leaps are never flattened: the
    two neighbours must themselves be close (a smooth line), the note must be
    a full octave-plus from both, and one octave shift must land it within a
    major 3rd of both. Returns (events, fixed_count)."""
    if len(events) < 3:
        return events, 0
    ev = sorted(events, key=lambda e: e['start'])
    fixed = 0
    for i in range(1, len(ev) - 1):
        cur, prev, nxt = ev[i]['note'], ev[i - 1]['note'], ev[i + 1]['note']
        if abs(prev - nxt) > 5:            # neighbours not a smooth line
            continue
        if abs(cur - prev) <= 12 or abs(cur - nxt) <= 12:   # not an octave spike
            continue
        for shift in (-12, 12, -24, 24):
            s = cur + shift
            if abs(s - prev) <= 4 and abs(s - nxt) <= 4:
                ev[i]['note'] = s
                fixed += 1
                break
    return events, fixed


_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR_DEG = [0, 2, 4, 5, 7, 9, 11]
_MINOR_DEG = [0, 2, 3, 5, 7, 8, 10, 11]   # natural + harmonic 7th


def _scale_pcs(key_str):
    """Pitch-class set of a 'Root mode' key, or None if unparseable."""
    m = re.match(r'^\s*([A-Ga-g][#b]?)\s*(major|maj|minor|min|m)?\s*$',
                 str(key_str or ''), re.I)
    if not m:
        return None
    flats = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
    name = flats.get(m.group(1).upper(), m.group(1).upper())
    if name not in _NOTE_NAMES:
        return None
    root = _NOTE_NAMES.index(name)
    is_minor = (m.group(2) or '').lower() in ('minor', 'min', 'm')
    degs = _MINOR_DEG if is_minor else _MAJOR_DEG
    return {(root + d) % 12 for d in degs}


def _bar_quality_flags(bars_slice, lo, key):
    """Mechanical per-bar correctness checks for one page's bars. Returns a
    list of 'bar N: <issue>' strings for bars whose objective measures look
    wrong — fed to the validation prompt so the LLM knows where to look."""
    parsed = []
    for bar in bars_slice:
        mel = [m for m, _t in ai_correct._parse_rewrite(bar.get('melody', ''))
               if m is not None]
        bas = [m for m, _t in ai_correct._parse_rewrite(bar.get('bass', ''))
               if m is not None]
        parsed.append(mel + bas)
    counts = sorted(len(p) for p in parsed)
    median = counts[len(counts) // 2] if counts else 0
    allowed = _scale_pcs(key)
    flags = []
    for i, notes in enumerate(parsed):
        barno = lo + i
        if allowed:
            oot = sum(1 for n in notes if n % 12 not in allowed)
            if oot >= 3:
                flags.append(f'bar {barno}: {oot} notes outside {key} '
                             f'(possible misread accidentals)')
        c = len(notes)
        if median >= 4 and (c < median * 0.35 or c > median * 2.6):
            flags.append(f'bar {barno}: note count {c} unlike its '
                         f'neighbours (~{median})')
    return flags


# ── Per-bar confidence model ───────────────────────────────────────────────────
# Mechanical scoring that decides which bars the AI should re-read from a
# high-resolution crop. The AI reads dense music far better from a single
# isolated measure than from a whole system strip, so the pipeline routes only
# low-confidence bars into a focused refinement pass.

def _bar_confidence(bar, median_density, key_pcs, bar_ticks):
    """Return a list of reason strings for why a bar looks wrong (empty list =
    confident). Combines rhythm, density, key-fit and octave-range checks."""
    reasons = []
    mel = ai_correct._parse_rewrite(bar.get('melody', ''))
    bas = ai_correct._parse_rewrite(bar.get('bass', ''))
    notes = [m for m, _t in mel + bas if m is not None]
    # Rhythm — each non-empty staff should fill close to one whole measure.
    for name, parsed in (('melody', mel), ('bass', bas)):
        toks = [t for _m, t in parsed]
        total = sum(toks)
        if total <= 0:
            continue
        ratio = total / bar_ticks
        if ratio < 0.62 or ratio > 1.6:
            reasons.append(f'{name} rhythm fills ~{ratio * 100:.0f}% of the bar')
    # Density — a bar with far more/fewer notes than its neighbours is suspect.
    c = len(notes)
    if median_density >= 4 and (c < median_density * 0.35 or
                                c > median_density * 2.6):
        reasons.append(f'note count {c} unlike neighbours (~{median_density})')
    # Key fit — several out-of-key notes usually mean misread accidentals.
    if key_pcs:
        oot = sum(1 for n in notes if n % 12 not in key_pcs)
        if oot >= 3:
            reasons.append(f'{oot} notes out of key')
    # Octave — a note well outside its staff's plausible band is an octave slip.
    for name, parsed, band in (('melody', mel, _TREBLE_BAND),
                               ('bass', bas, _BASS_BAND)):
        ps = [m for m, _t in parsed if m is not None]
        far = sum(1 for p in ps if p < band[0] - 12 or p > band[1] + 12)
        if far:
            reasons.append(f'{name} has {far} note(s) outside its staff range')
    return reasons


def _low_confidence_bars(bars_slice, lo, key, timesig):
    """Score every bar of a page; return {global_bar_no: [reasons]} for the
    bars a mechanical check considers likely-wrong."""
    try:
        num, den = map(int, str(timesig).split('/'))
        bar_ticks = (num * 4 / den) * DIV
    except (ValueError, AttributeError):
        bar_ticks = 4 * DIV
    key_pcs = _scale_pcs(key)
    counts = []
    for bar in bars_slice:
        mel = ai_correct._parse_rewrite(bar.get('melody', ''))
        bas = ai_correct._parse_rewrite(bar.get('bass', ''))
        counts.append(len([m for m, _t in mel + bas if m is not None]))
    sc = sorted(counts)
    median = sc[len(sc) // 2] if sc else 0
    flagged = {}
    for idx, bar in enumerate(bars_slice):
        reasons = _bar_confidence(bar, median, key_pcs, bar_ticks)
        if reasons:
            flagged[lo + idx] = reasons
    return flagged


def _key_fit(hist, key_str):
    """In-key note fraction of the histogram for a given 'Root mode' key, or
    None if the key string can't be parsed."""
    total = sum(hist)
    if not total or not key_str:
        return None
    m = re.match(r'^\s*([A-Ga-g][#b]?)\s*(major|maj|minor|min|m)?\s*$',
                 str(key_str), re.I)
    if not m:
        return None
    flats = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
    name = m.group(1).upper()
    name = flats.get(name, name)
    if name not in _NOTE_NAMES:
        return None
    root = _NOTE_NAMES.index(name)
    is_minor = (m.group(2) or '').lower() in ('minor', 'min', 'm')
    degs = _MINOR_DEG if is_minor else _MAJOR_DEG
    return sum(hist[(root + d) % 12] for d in degs) / total


def _detect_key(all_bars, qL_per_bar):
    """Detect the key from the transcription's pitch-class histogram
    (mechanical, objective). Returns (key_string, in_key_fraction, hist)
    or None."""
    hist = [0] * 12
    for bar in all_bars:
        for role in ('melody', 'bass'):
            for m, _s, _e in _bar_to_events(bar.get(role, ''), 0, qL_per_bar):
                hist[m % 12] += 1
    total = sum(hist)
    if total < 24:
        return None
    best, best_score = None, -1.0
    for root in range(12):
        for degs, mode in ((_MAJOR_DEG, 'major'), (_MINOR_DEG, 'minor')):
            allowed = {(root + d) % 12 for d in degs}
            in_key = sum(hist[pc] for pc in allowed)
            # in-key fraction, with a small bonus for tonic emphasis
            score = in_key / total + 0.04 * (hist[root] / total)
            if score > best_score:
                best, best_score = (root, mode), score
    root, mode = best
    in_frac = sum(hist[(root + d) % 12]
                  for d in (_MINOR_DEG if mode == 'minor' else _MAJOR_DEG)) / total
    return f'{_NOTE_NAMES[root]} {mode}', in_frac, hist


def _sane_timesig(value, fallback='4/4'):
    """Validate a time signature string. A malformed timeSig silently breaks
    every bar-duration calculation, so reject implausible ones."""
    try:
        num, den = (str(value).strip().split('/') + [''])[:2]
        num, den = int(num), int(den)
        if 1 <= num <= 32 and den in (1, 2, 4, 8, 16, 32):
            return f'{num}/{den}'
    except (ValueError, AttributeError, TypeError):
        pass
    return fallback


def _bar_has_notes(bar):
    """True if a bar dict has at least one sounding note on either staff."""
    for role in ('melody', 'bass'):
        if any(m is not None for m, _t in
               ai_correct._parse_rewrite(bar.get(role, ''))):
            return True
    return False


def _trim_trailing_empty(all_bars):
    """Drop phantom fully-empty bars the LLM sometimes appends at the end —
    they inflate the bar count and add trailing silence. Returns count dropped."""
    dropped = 0
    while all_bars and not _bar_has_notes(all_bars[-1]):
        all_bars.pop()
        dropped += 1
    return dropped


def _fix_staff_swaps(all_bars):
    """If a bar's melody staff sits clearly below its bass staff, the LLM
    swapped the treble and bass for that bar — swap them back. Returns the
    number of bars corrected."""
    fixed = 0
    for bar in all_bars:
        mel = [m for m, _t in ai_correct._parse_rewrite(bar.get('melody', ''))
               if m is not None]
        bas = [m for m, _t in ai_correct._parse_rewrite(bar.get('bass', ''))
               if m is not None]
        if len(mel) >= 2 and len(bas) >= 2:
            mel_avg = sum(mel) / len(mel)
            bas_avg = sum(bas) / len(bas)
            if mel_avg + 5 < bas_avg:        # melody a 4th+ below bass: swapped
                bar['melody'], bar['bass'] = bar['bass'], bar['melody']
                fixed += 1
    return fixed


def _mechanical_cleanup(tracks):
    """Rule-based cleanup of the AI transcription: octave-correct outliers
    (global + local), de-duplicate. Bar-duration normalisation already
    happened in _bar_to_events."""
    mel0, bas0 = len(tracks['melody']), len(tracks['bass'])
    fixed = 0
    for role, band in (('melody', _TREBLE_BAND), ('bass', _BASS_BAND)):
        ev = _octave_fix(tracks[role], band)
        ev, f = _smooth_octaves(ev)
        fixed += f
        tracks[role] = _dedup(ev)
    drop = (mel0 - len(tracks['melody'])) + (bas0 - len(tracks['bass']))
    if drop or fixed:
        _log(f'mechanical cleanup: {fixed} octave jump(s) smoothed, '
             f'{drop} duplicate(s) removed')
    return tracks


_TOK_RE = re.compile(r'([A-Ga-gR][#b]?-?\d*)\([^)]*\)')


def _is_empty_staff(s):
    return not s or str(s).strip().lower() in ('(empty)', 'empty', '')


def _is_rest_only(s):
    """True if the staff string contains only rests (no pitched note)."""
    if _is_empty_staff(s):
        return False
    toks = _TOK_RE.findall(str(s))
    return bool(toks) and all(t.upper().startswith('R') for t in toks)


def _strip_leading_rests(s):
    """Drop any rest tokens at the very start of a staff string."""
    if _is_empty_staff(s):
        return s
    return re.sub(r'^\s*(?:R[#b]?-?\d*\([^)]*\)\s*)+', '', str(s)).strip()


def _normalize_bar_rests(bars):
    """Silence is silence — a staff that is only rests becomes '(empty)' (the
    prompt's own rule, enforced deterministically). The first bar is a
    pickup/anacrusis: it never starts with a printed rest before its notes, so
    strip any leading rests there too. Fixes the classic Für Elise pickup where
    the model writes a stray bass 'R(8)' (or pads the melody with leading rests)
    instead of leaving the silent staff empty."""
    for idx, bar in enumerate(bars):
        if not isinstance(bar, dict):
            continue
        for tr in ('melody', 'melody2', 'bass', 'bass2'):
            s = bar.get(tr, '')
            if idx == 0 and not _is_empty_staff(s):
                stripped = _strip_leading_rests(s)
                if stripped != s:
                    s = stripped or '(empty)'
                    bar[tr] = s
            if _is_rest_only(s):
                bar[tr] = '(empty)'
    return bars


def _write_piece(piece_id, out_dir, all_bars, meta):
    """Write melody/bass/pad/drums MIDI + catalog.json for the current bars."""
    from pdf_to_midi import write_midi_track
    num, den = map(int, str(meta['timeSig']).split('/'))
    qL_per_bar = num * 4 / den
    bpm = meta['bpm']
    tracks = _mechanical_cleanup(_bars_to_tracks(all_bars, qL_per_bar, bpm))
    for role in ('melody', 'bass', 'pad', 'drums'):
        write_midi_track(tracks[role], bpm, meta['timeSig'],
                         out_dir / f'{role}.mid',
                         channel=9 if role == 'drums' else 0)
    catalog = {
        'id': piece_id, 'title': meta['title'], 'composer': meta.get('composer', ''),
        'genre': 'Classical', 'mood': 'Expressive', 'bpm': bpm,
        'key': meta['key'], 'bars': len(all_bars), 'timeSig': meta['timeSig'],
        'source': 'human', 'importedFrom': meta.get('importedFrom', ''),
        'description': 'Imported from PDF via AI-vision transcription (ai_transcribe.py)',
    }
    (out_dir / 'catalog.json').write_text(json.dumps(catalog, indent=2),
                                          encoding='utf-8')
    return tracks


def _apply_bar_corrections(all_bars, corrections):
    """Apply [{track,bar,rewrite}] rewrites in-place to the all_bars list."""
    applied = 0
    for c in corrections:
        try:
            bar_idx = int(c['bar']) - 1
            track = (c.get('track') or 'melody').lower()
            rewrite = c.get('rewrite')
        except (KeyError, TypeError, ValueError):
            continue
        if track not in ('melody', 'bass') or not rewrite:
            continue
        if 0 <= bar_idx < len(all_bars):
            all_bars[bar_idx][track] = rewrite
            applied += 1
    return applied


def _apply_refinements(all_bars, refinements):
    """Apply whole-bar rewrites [{bar, melody, bass}] from the refinement pass.
    Returns the number of bars actually changed."""
    applied = 0
    for r in refinements:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r['bar']) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= idx < len(all_bars)):
            continue
        changed = False
        for track in ('melody', 'bass'):
            val = r.get(track)
            if val and str(val).strip() and all_bars[idx].get(track) != val:
                all_bars[idx][track] = val
                changed = True
        if changed:
            applied += 1
    return applied


# ── Main entry point ──────────────────────────────────────────────────────────

def transcribe_pdf(pdf_path, out_dir, piece_id, title, composer,
                   bpm_override=None, provider='gemini', only_pages=None,
                   max_bars=None, time_sig=None, key=None, engine=None):
    """Batched AI transcription with render-and-validate. Returns (tracks, bpm, meta).

    only_pages: optional set/iterable of 1-based page numbers to TRANSCRIBE.
      Pages already cached (page_NN.done.json) are always loaded so partial
      compiles accumulate; pages outside the set that aren't cached are left
      'pending' (no bars) so the user can compile them later. None = all pages.
    max_bars: optional cap — stop after this many bars (a fast preview, e.g.
      the first 2 bars). The page it stops on is marked 'partial' so it can be
      recompiled in full later. None = no limit.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    pages_dir = out_dir / '_pages'
    only_pages = set(only_pages) if only_pages else None
    try:
        max_bars = int(max_bars) if max_bars else None
    except (TypeError, ValueError):
        max_bars = None

    # A bar-limit preview only ever reads page 1 (matching the detect scope), so
    # bound it explicitly — we then neither rasterise nor scan later pages.
    if max_bars and only_pages is None:
        only_pages = {1}

    eng_ok, eng_detail = ai_engine.available(engine)
    if not eng_ok:
        raise RuntimeError(eng_detail)
    _log(f'transcription engine: {eng_detail}')

    # Rasterise only the pages this run will actually read.
    page_pngs = _render_pdf_pages(pdf_path, pages_dir, want_pages=only_pages)
    if not page_pngs:
        raise RuntimeError('PDF has no pages')
    print(f'[transcribe] {len(page_pngs)} page(s) rendered; provider {provider}')

    # User-supplied meter/key are authoritative ground truth — seed meta with
    # them and lock so the AI's own (often wrong) reading can't override.
    time_sig = _sane_timesig(time_sig, None) if time_sig else None
    key = (str(key).strip() or None) if key else None
    locked = set()
    meta = {'key': key or 'C', 'timeSig': time_sig or '4/4', 'bpm': None,
            'title': title, 'composer': composer, 'importedFrom': pdf_path.name}
    if time_sig:
        locked.add('timeSig')
    if key:
        locked.add('key')
    all_bars = []
    page_manifest = []
    audio_ok = True  # set False if the bridge has no /api/audio-ask route

    # ── Resume support ──────────────────────────────────────────────────────
    # A re-run reuses cached work from an interrupted import so we never start
    # from scratch:  meta.json = key/timeSig/bpm;  page_NN.done.json = a fully
    # finished page's bars;  sys_NN_MM.json = one system's raw transcription.
    meta_cache = pages_dir / 'meta.json'
    got_meta = False
    if meta_cache.exists():
        try:
            saved = json.loads(meta_cache.read_text(encoding='utf-8'))
            for k in ('key', 'timeSig', 'bpm'):
                if saved.get(k):
                    meta[k] = saved[k]
            got_meta = True
            _log(f'resume: loaded cached meta ({meta["key"]}, {meta["timeSig"]})')
        except Exception:
            pass

    def _save_meta():
        try:
            meta_cache.write_text(json.dumps(
                {k: meta[k] for k in ('key', 'timeSig', 'bpm')}), encoding='utf-8')
        except Exception:
            pass

    for i, png in enumerate(page_pngs, 1):
        page_done = pages_dir / f'page_{i:02d}.done.json'
        if page_done.exists():
            try:
                _cache = json.loads(page_done.read_text(encoding='utf-8'))
                cbars = [b for b in (_cache.get('bars') or []) if isinstance(b, dict)]
                lo = len(all_bars) + 1
                all_bars.extend(cbars)
                hi = len(all_bars)
                # Preserve a partial (bar-limited preview) page's status so the
                # UI keeps offering to recompile it in full.
                page_manifest.append({'file': png.name, 'page': i,
                                      'startBar': lo, 'endBar': hi,
                                      'status': 'partial' if _cache.get('partial')
                                      else 'done'})
                _log(f'page {i}/{len(page_pngs)}: resumed from cache '
                     f'({len(cbars)} bars{", partial" if _cache.get("partial") else ""})')
                continue
            except Exception:
                _log(f'page {i}: cached result unreadable -- re-transcribing')

        # ── Page-range filter ───────────────────────────────────────────────
        # When a subset was requested, leave out-of-range uncached pages
        # 'pending' so they can be compiled in a later run.
        if only_pages is not None and i not in only_pages:
            page_manifest.append({'file': png.name, 'page': i,
                                  'startBar': 0, 'endBar': 0, 'status': 'pending'})
            _log(f'page {i}/{len(page_pngs)}: skipped (not in requested range)')
            continue

        # ── 1. Transcribe this page system-by-system (small fast batches) ────
        _log(f'=== page {i}/{len(page_pngs)} ===')
        strips = _split_page_into_systems(png, pages_dir, i)
        _log(f'page {i}: split into {len(strips)} system strip(s)')

        # Establish meter/key ONCE, up front, from the first system — a focused
        # single-purpose read is far more reliable than the meter the model
        # mentions in passing while transcribing notes. No human input needed;
        # a user-supplied value (in `locked`) always wins and skips this.
        if i == 1 and not got_meta and strips and (
                'timeSig' not in locked or 'key' not in locked):
            det_ts, det_key, det_bpm = _detect_meter_key(
                strips[0], title, composer, provider, engine=engine)
            if det_ts and 'timeSig' not in locked:
                meta['timeSig'] = det_ts
                time_sig = det_ts        # feed it into every system's prompt
                _log(f'detected time signature: {det_ts}')
            if det_key and 'key' not in locked:
                meta['key'] = det_key
                key = det_key
                _log(f'detected key: {det_key}')
            if det_bpm and not bpm_override and not meta.get('bpm'):
                meta['bpm'] = det_bpm
            if det_ts or det_key:
                got_meta = True          # don't let in-pass reads override it
                _save_meta()

        page_start = len(all_bars) + 1
        bar_crop = {}  # global_bar_no -> per-bar crop PNG, for refinement
        truncated = False  # set when a max_bars cap stops this page early
        for j, strip in enumerate(strips, 1):
            sys_cache = pages_dir / f'sys_{i:02d}_{j:02d}.json'
            data = None
            if sys_cache.exists():
                try:
                    data = json.loads(sys_cache.read_text(encoding='utf-8'))
                    _log(f'page {i} system {j}/{len(strips)}: loaded from cache')
                except Exception:
                    data = None
            if data is None:
                bar_hint = _detect_barlines(strip)   # mechanical bar-count hint
                _log(f'page {i} system {j}/{len(strips)}: transcribing'
                     + (f' [~{bar_hint} bars]' if bar_hint else '') + '...')
                t0 = time.time()
                try:
                    data = _parse_json(ai_engine.image_ask(
                        strip, _transcribe_prompt(title, composer, i,
                                                  len(page_pngs), bar_hint,
                                                  time_sig=time_sig, key=key),
                        engine=engine, provider=provider,
                        label=f'page{i}-sys{j}'))
                except Exception as e:
                    _log(f'page {i} system {j} failed after {time.time()-t0:.0f}s: {e}')
                    continue
                n = len([b for b in (data.get('bars') or []) if isinstance(b, dict)])
                # Mechanical cross-check: if the AI's bar count disagrees with
                # the barline scan by 2+, re-transcribe once with the count
                # enforced and keep whichever lands closer to the scan.
                if bar_hint and abs(n - bar_hint) >= 2:
                    _log(f'page {i} system {j}: bar-count mismatch '
                         f'(ai {n} vs scan {bar_hint}) -- re-transcribing')
                    try:
                        retry = _parse_json(ai_engine.image_ask(
                            strip, _transcribe_prompt(title, composer, i,
                                     len(page_pngs), bar_hint, strict=True,
                                     time_sig=time_sig, key=key),
                            engine=engine, provider=provider,
                            label=f'page{i}-sys{j}-strict'))
                        rn = len([b for b in (retry.get('bars') or [])
                                  if isinstance(b, dict)])
                        if abs(rn - bar_hint) < abs(n - bar_hint):
                            data, n = retry, rn
                            _log(f'page {i} system {j}: re-transcription kept '
                                 f'({rn} bars)')
                    except Exception as e:
                        _log(f'page {i} system {j}: re-transcription failed: {e}')
                try:
                    sys_cache.write_text(json.dumps(data), encoding='utf-8')
                except Exception:
                    pass
                _log(f'page {i} system {j}: {n} bar(s) in {time.time()-t0:.0f}s')
            if not got_meta and (data.get('key') or data.get('timeSig')
                                 or data.get('bpm')):
                # Locked fields are user-supplied ground truth — never let the
                # AI's reading overwrite them.
                if 'key' not in locked:
                    meta['key'] = data.get('key') or meta['key']
                if 'timeSig' not in locked:
                    meta['timeSig'] = _sane_timesig(data.get('timeSig'),
                                                    meta['timeSig'])
                try:
                    meta['bpm'] = int(round(float(data.get('bpm'))))
                except (TypeError, ValueError):
                    meta['bpm'] = None
                got_meta = True
                _save_meta()
            sys_bars = [b for b in (data.get('bars') or [])
                        if isinstance(b, dict)]
            if sys_bars:
                gstart = len(all_bars) + 1
                all_bars.extend(sys_bars)
                # Mechanically crop each bar of this system so the refinement
                # pass can re-read low-confidence bars at full resolution.
                crops = _bar_crops(strip, len(sys_bars), pages_dir, i, j)
                if crops and len(crops) == len(sys_bars):
                    for k, cp in enumerate(crops):
                        bar_crop[gstart + k] = cp
            # Bar-limit preview: stop as soon as we have enough bars.
            if max_bars and len(all_bars) >= max_bars:
                del all_bars[max_bars:]
                truncated = True
                _log(f'page {i}: reached bar limit ({max_bars}) — stopping early '
                     f'(preview; this page is marked partial)')
                break
        lo, hi = page_start, len(all_bars)
        page_manifest.append({'file': png.name, 'page': i, 'startBar': lo,
                              'endBar': hi,
                              'status': 'partial' if truncated
                              else ('done' if hi >= lo else 'empty')})
        _log(f'page {i}: {hi - lo + 1 if hi >= lo else 0} bar(s) total '
             f'(bars {lo}-{hi})')
        if hi < lo:
            continue

        # A bar-limited preview skips the refinement/validation passes and
        # stops here — the partial page caches what it has and can be
        # recompiled in full later.
        if truncated:
            meta['bpm'] = bpm_override or meta['bpm'] or 100
            _save_meta()
            _normalize_bar_rests(all_bars)
            _write_piece(piece_id, out_dir, all_bars, meta)
            try:
                page_done.write_text(
                    json.dumps({'bars': all_bars[lo - 1:hi], 'partial': True},
                               indent=1), encoding='utf-8')
            except Exception:
                pass
            # Remaining pages stay pending so the UI can compile them later.
            for pj in range(i + 1, len(page_pngs) + 1):
                page_manifest.append({'file': page_pngs[pj - 1].name, 'page': pj,
                                      'startBar': 0, 'endBar': 0,
                                      'status': 'pending'})
            break
        meta['bpm'] = bpm_override or meta['bpm'] or 100
        _save_meta()

        # ── 2. Confidence-routed refinement ──────────────────────────────────
        # Mechanical scoring flags low-confidence bars; each is re-read by the
        # AI from a high-resolution crop of that single measure (montaged so a
        # batch of bars is fixed in one call). Loops until the flagged set
        # stops shrinking — the AI focuses precisely where it is weak.
        _write_piece(piece_id, out_dir, all_bars, meta)
        prev_count = None
        for rnd in range(1, REFINE_ROUNDS + 1):
            flagged = _low_confidence_bars(all_bars[lo - 1:hi], lo,
                                           meta.get('key'), meta['timeSig'])
            if not flagged:
                _log(f'page {i}: all bars confident'
                     + (f' after {rnd - 1} refine round(s)' if rnd > 1 else ''))
                break
            items = [(bn, bar_crop[bn]) for bn in sorted(flagged)
                     if bn in bar_crop]
            if not items:
                _log(f'page {i}: {len(flagged)} low-confidence bar(s) but no '
                     f'crops -- deferring to holistic validation')
                break
            if prev_count is not None and len(flagged) >= prev_count:
                _log(f'page {i}: refinement converged '
                     f'({len(flagged)} bar(s) still flagged)')
                break
            prev_count = len(flagged)
            _log(f'page {i}: refine round {rnd} -- re-reading '
                 f'{len(items)} low-confidence bar(s) from crops')
            applied = 0
            for c0 in range(0, len(items), REFINE_CHUNK):
                chunk = items[c0:c0 + REFINE_CHUNK]
                montage = _build_montage(
                    chunk,
                    pages_dir / f'_refine_p{i:02d}_r{rnd}_{c0 // REFINE_CHUNK}.png')
                if not montage:
                    continue
                try:
                    rdata = _parse_json(ai_engine.image_ask(
                        str(montage),
                        _refine_prompt(title, composer, meta.get('key'),
                                       meta['timeSig'],
                                       {bn: flagged[bn] for bn, _cp in chunk}),
                        engine=engine, provider=provider, label=f'refine-p{i}'))
                except Exception as e:
                    _log(f'page {i}: refine round {rnd} chunk failed: {e}')
                    continue
                # Only apply rewrites for bars that were actually in this
                # montage — guards against a hallucinated bar number clobbering
                # a bar the AI never saw.
                chunk_bars = {bn for bn, _cp in chunk}
                refs = []
                for r in (rdata.get('bars') or []):
                    if not isinstance(r, dict):
                        continue
                    try:
                        if int(r['bar']) in chunk_bars:
                            refs.append(r)
                    except (KeyError, TypeError, ValueError):
                        continue
                applied += _apply_refinements(all_bars, refs)
            _log(f'page {i}: refine round {rnd} applied {applied} bar rewrite(s)')
            _write_piece(piece_id, out_dir, all_bars, meta)
            if not applied:
                break

        # ── 3. Holistic validation — full page sheet + piano roll vs original.
        for rnd in range(1, VALIDATE_ROUNDS + 1):
            _write_piece(piece_id, out_dir, all_bars, meta)
            _log(f'page {i}: holistic validation (sheet + piano roll)...')
            batch_bars = all_bars[lo - 1:hi]
            our_pngs = _render_batch_sheet(batch_bars, meta)
            if not our_pngs:
                _log(f'page {i}: could not render sheet -- skipping validation')
                break
            num, den = map(int, str(meta['timeSig']).split('/'))
            roll = _render_piano_roll(batch_bars, num * 4 / den,
                                      pages_dir / f'_roll_p{i:02d}.png')
            stacked = pages_dir / f'_our_p{i:02d}.png'
            combo = pages_dir / f'_validate_p{i:02d}.png'
            panels = list(our_pngs) + ([roll] if roll else [])
            if not _stack_vertical(panels, stacked) or \
               not _composite(png, stacked, combo):
                _log(f'page {i}: image compositing failed -- skipping validation')
                break
            flags = _bar_quality_flags(batch_bars, lo, meta.get('key'))
            try:
                res = _parse_json(ai_engine.image_ask(
                    str(combo),
                    _validate_prompt(title, composer, i, lo, hi,
                                     meta.get('key'), flags),
                    engine=engine, provider=provider, label=f'validate-p{i}'))
            except Exception as e:
                _log(f'page {i}: holistic validation failed: {e}')
                break
            corrections = res.get('corrections') or []
            if res.get('ok') and not corrections:
                _log(f'page {i}: holistic validation clean')
                break
            applied = _apply_bar_corrections(all_bars, corrections)
            _log(f'page {i}: holistic validation applied {applied} correction(s)')
            if not applied:
                break

        # ── 4. Audio validation — let the AI HEAR the transcription ──────────
        # A synth render exposes errors the eye misses on the staff: a note
        # that looks plausible but clashes audibly, a rhythm that lurches.
        if audio_ok and hi >= lo:
            batch_bars = all_bars[lo - 1:hi]
            page_wav = _render_batch_audio(batch_bars, meta,
                                           pages_dir / f'_audio_p{i:02d}.wav')
            if page_wav and page_wav.exists():
                _log(f'page {i}: rendering audio + validating by ear...')
                ares = None
                try:
                    ares = _parse_json(ai_engine.audio_ask(
                        str(page_wav),
                        _audio_validate_prompt(title, composer, i, lo, hi,
                                               meta.get('key'), meta['timeSig'],
                                               meta['bpm']),
                        engine=engine, provider=provider))
                except Exception as e:
                    audio_ok = False
                    _log(f'page {i}: audio validation unavailable ({e}) -- '
                         f'skipping audio checks for the rest of this run')
                if ares is not None:
                    acorr = ares.get('corrections') or []
                    if ares.get('ok') and not acorr:
                        _log(f'page {i}: audio validation found nothing wrong')
                    else:
                        applied = _apply_bar_corrections(all_bars, acorr)
                        _log(f'page {i}: audio validation applied {applied} '
                             f'correction(s)')
                        if applied:
                            _write_piece(piece_id, out_dir, all_bars, meta)

        # ── Cache the finished page so a re-run skips it entirely ────────────
        _normalize_bar_rests(all_bars)
        try:
            page_done.write_text(json.dumps({'bars': all_bars[lo - 1:hi]},
                                            indent=1), encoding='utf-8')
            for sc in pages_dir.glob(f'sys_{i:02d}_*.json'):
                sc.unlink()  # page-level cache supersedes the per-system ones
        except Exception:
            pass

    if not all_bars:
        raise RuntimeError('AI transcription produced no bars')

    trimmed = _trim_trailing_empty(all_bars)
    if trimmed:
        _log(f'trimmed {trimmed} phantom empty bar(s) from the end')
    if not all_bars:
        raise RuntimeError('AI transcription produced no bars')

    # Clamp the tempo to a musically plausible range — the LLM occasionally
    # reads a wild value off a tempo marking.
    meta['bpm'] = bpm_override or meta['bpm'] or 100
    if not bpm_override:
        meta['bpm'] = min(220, max(30, int(meta['bpm'])))
    meta['bars'] = len(all_bars)
    try:
        num, den = map(int, str(meta['timeSig']).split('/'))
        qL_per_bar = num * 4 / den
    except Exception:
        meta['timeSig'], qL_per_bar = '4/4', 4.0

    swapped = _fix_staff_swaps(all_bars)
    if swapped:
        _log(f'fixed {swapped} bar(s) with swapped treble/bass staves')

    # Reconcile the key. The AI reads it straight off the key signature and is
    # usually right; only override it when the AI's key genuinely fits the
    # note distribution POORLY and a detected key fits clearly better. This
    # avoids being fooled by chromatic neighbour tones (e.g. Fur Elise's D#,
    # which makes a naive scale-fit favour E minor over the true A minor).
    detected = _detect_key(all_bars, qL_per_bar) if 'key' not in locked else None
    if 'key' in locked:
        _log(f'key locked to user-supplied "{meta["key"]}"')
    if detected:
        det_key, det_frac, hist = detected
        ai_fit = _key_fit(hist, meta.get('key'))
        if (det_key.lower() != str(meta['key']).strip().lower()
                and (ai_fit is None or ai_fit < 0.80)
                and det_frac >= (ai_fit or 0) + 0.10):
            _log(f'key reconciled: AI "{meta["key"]}" fits '
                 f'{("%.0f%%" % (ai_fit*100)) if ai_fit is not None else "n/a"}, '
                 f'"{det_key}" fits {det_frac*100:.0f}% -- using detected')
            meta['key'] = det_key
        else:
            _log(f'key kept: "{meta["key"]}" '
                 f'(fit {("%.0f%%" % (ai_fit*100)) if ai_fit is not None else "n/a"})')

    _normalize_bar_rests(all_bars)
    tracks = _write_piece(piece_id, out_dir, all_bars, meta)
    (pages_dir / 'pages.json').write_text(
        json.dumps({'pages': page_manifest, 'bars': len(all_bars)}, indent=2),
        encoding='utf-8')

    print(f'\n[transcribe] done: {len(all_bars)} bars, key {meta["key"]}, '
          f'{meta["timeSig"]}, {meta["bpm"]} BPM, '
          f'melody {len(tracks["melody"])} / bass {len(tracks["bass"])} notes')
    return tracks, meta['bpm'], meta
