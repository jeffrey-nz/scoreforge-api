"""Locate where a bar's transcribed glyphs sit on its source crop.

The OMR MusicXML carries no pixel coordinates, so we recover geometry straight
from the crop image: detect the two staves (treble + bass) by their full-width
ink lines, then map each token's pitch to its real vertical position on those
staves. Horizontal position comes from musical onset. This lets the UI draw a
box on each glyph and a leader line to its label — "what was picked up, and
where it maps on the page."

Pure numpy/PIL (no oemer, no ML). Returns normalised (0..1) coordinates so the
client can scale to whatever size the crop is displayed at.
"""
from __future__ import annotations
import re
from typing import Optional, List, Dict

_STEP = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
_DUR = {'w': 4, 'h': 2, 'q': 1, '8': 0.5, '16': 0.25, '32': 0.125, '64': 0.0625}
# diatonic index (octave*7 + step) of each staff's BOTTOM line
_TREBLE_BOTTOM = 4 * 7 + _STEP['E']   # E4 = 30
_BASS_BOTTOM = 2 * 7 + _STEP['G']     # G2 = 18


def _diatonic(name: str) -> Optional[int]:
    m = re.match(r'^([A-Ga-g])[#b]?(-?\d+)$', name.strip())
    if not m:
        return None
    return int(m.group(2)) * 7 + _STEP[m.group(1).upper()]


def _dur_beats(tag: str, dot: bool) -> float:
    b = _DUR.get(tag, 0.0)
    return b * 1.5 if dot else b


def _parse_events(s: str):
    """[(heads:list[str] (empty=rest), onset, dur, is_rest)] in quarter-beats."""
    out, cur = [], 0.0
    for tok in str(s or '').strip().split():
        m = re.match(r'^(.+)\((w|h|q|8|16|32|64|g)(\.?)\)$', tok)
        if not m:
            continue
        head, tag, dot = m.group(1), m.group(2), m.group(3) == '.'
        dur = 0.0 if tag == 'g' else _dur_beats(tag, dot)
        if re.match(r'^[Rr]$', head):
            out.append(([], cur, dur, True))
        else:
            out.append(([h for h in head.split('+') if h], cur, dur, False))
        cur += dur
    return out, cur


def detect_staff_lines(gray) -> List[float]:
    """y-centres of full-width horizontal ink lines (staff lines), top→bottom.
    Full-width requirement rejects beams/noteheads/ledger lines."""
    import numpy as np
    H, W = gray.shape
    dark = gray < 140
    rowdark = dark.sum(axis=1)
    for frac in (0.55, 0.45, 0.35):       # relax if a faint scan finds too few
        is_line = rowdark > frac * W
        lines, y = [], 0
        while y < H:
            if is_line[y]:
                y0 = y
                while y < H and is_line[y]:
                    y += 1
                lines.append((y0 + y - 1) / 2.0)
            else:
                y += 1
        if len(lines) >= 10:
            return lines
    return lines


def _two_staves(lines: List[float]):
    """Split detected lines into (treble5, bass5) by the largest gap (the space
    between the two staves of a grand staff). Returns None if not confident."""
    lines = sorted(lines)
    if len(lines) < 8:
        return None
    gaps = [b - a for a, b in zip(lines, lines[1:])]
    gi = max(range(len(gaps)), key=lambda i: gaps[i])   # inter-staff gap
    treble, bass = lines[:gi + 1], lines[gi + 1:]
    treble, bass = treble[-5:], bass[:5]                # 5 nearest the gap each
    if len(treble) < 5 or len(bass) < 5:
        return None
    return treble, bass


def _disk(r):
    import numpy as np
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def detect_noteheads(gray, lines, spacing):
    """Actual notehead blobs on one staff: (cx, cy, area), left→right. Morphological
    opening with a notehead-sized disk drops thin stems + staff lines; size/extent
    filters drop the clef and beams. Used to place labels on the real notes (x AND
    y from the image) instead of guessing x from musical onset."""
    import numpy as np
    from scipy import ndimage
    H, W = gray.shape
    sp = spacing
    top = max(0, int(min(lines) - 1.6 * sp))
    bot = min(H, int(max(lines) + 1.6 * sp))
    sub = gray[top:bot] < 110
    opened = ndimage.binary_opening(sub, structure=_disk(max(2, int(sp * 0.42))))
    lbl, n = ndimage.label(opened)
    if not n:
        return []
    out = []
    amin, amax = (sp * 0.42) ** 2, (sp * 1.8) ** 2
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        ys, xs = sl
        hh, ww = ys.stop - ys.start, xs.stop - xs.start
        area = int((lbl[sl] == i).sum())
        if area < amin or area > amax or hh > 2.3 * sp or ww > 2.3 * sp:
            continue                          # too small (noise) / too big (clef/beam)
        out.append(((xs.start + xs.stop) / 2.0, (ys.start + ys.stop) / 2.0 + top, area))
    # Drop blobs much smaller than the typical notehead — these are accidental
    # glyphs (a ♯/♭ to the left of a note) that would otherwise inflate the count.
    if len(out) >= 3:
        import statistics
        med = statistics.median(a for _, _, a in out)
        out = [t for t in out if t[2] >= 0.55 * med]
    out.sort()
    return out


_STEP_R = 'CDEFGAB'
def _ydia_pitch(cy, bottom_y, bottom_dia, sp):
    """Natural pitch name from a notehead's staff height (no accidental)."""
    d = int(round(bottom_dia + (bottom_y - cy) * 2.0 / sp))
    octv, st = divmod(d, 7)
    return f'{_STEP_R[st]}{octv}'


def _has_accidental(gray, cx, cy, sp):
    """True if a real accidental glyph (♯/♭/♮ — TALL ~2 staff-spaces and solid)
    sits immediately left of a notehead. Height+area reject the previous notehead
    / stem fragments that also fall in that region."""
    from scipy import ndimage
    x0, x1 = int(cx - 1.5 * sp), int(cx - 0.35 * sp)
    y0, y1 = int(cy - 1.0 * sp), int(cy + 1.0 * sp)
    if x1 <= x0:
        return False
    reg = gray[max(0, y0):y1, max(0, x0):x1] < 110
    lbl, nb = ndimage.label(reg)
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        ys = sl[0]
        hh = ys.stop - ys.start
        if hh >= 1.7 * sp and int((lbl[sl] == i).sum()) >= 0.8 * sp * sp:
            return True
    return False


def _accidental_bbox(gray, cx, cy, sp):
    """Pixel bbox [x0,y0,x1,y1] of a real accidental glyph (♯/♭/♮) sitting just left
    of a notehead, or None — same TALL+solid test as `_has_accidental`, but returns
    the glyph's box so the UI can draw and link it. Picks the component nearest the
    notehead when several qualify."""
    from scipy import ndimage
    x0, x1 = int(cx - 1.7 * sp), int(cx - 0.3 * sp)
    y0, y1 = int(cy - 1.2 * sp), int(cy + 1.2 * sp)
    if x1 <= x0:
        return None
    ox0, oy0 = max(0, x0), max(0, y0)
    reg = gray[oy0:y1, ox0:x1] < 110
    lbl, nb = ndimage.label(reg)
    best = None
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        ys, xs = sl
        hh, ww = ys.stop - ys.start, xs.stop - xs.start
        area = int((lbl[sl] == i).sum())
        # A real ♯/♭/♮ is TALL, sizeable, and reasonably SOLID. Stricter than the
        # boolean `_has_accidental` (which only gates pitch carry-through): the overlay
        # must not box stems/beam-stubs left of a note (sparse: small area, low fill).
        if hh >= 1.9 * sp and area >= 1.1 * sp * sp and area >= 0.34 * hh * ww:
            bb = [ox0 + xs.start, oy0 + ys.start, ox0 + xs.stop, oy0 + ys.stop]
            if best is None or bb[2] > best[2]:    # rightmost = closest to the notehead
                best = bb
    return best


def detect_rests(gray, lines, sp, noteheads):
    """Rest glyphs on one staff: (cx, cy, [x0,y0,x1,y1]), x-sorted. Strips staff lines,
    stems (tall-thin) and beams (wide), then keeps isolated, reasonably-solid, rest-
    sized blobs sitting within the staff and clear of any notehead. Decouples rests from
    musical onset (which lands on noteheads) by finding the actual engraved glyph."""
    import numpy as np
    from scipy import ndimage
    H, W = gray.shape
    top, bot = max(0, int(min(lines) - 1.4 * sp)), min(H, int(max(lines) + 1.4 * sp))
    ink = (gray[top:bot] < 110)
    m = ink.copy()
    for ly in lines:                              # strip the staff lines
        r = int(ly) - top
        if 0 <= r < m.shape[0]:
            m[max(0, r - 2):r + 3, :] = False
    m = ndimage.binary_closing(m, structure=_disk(1))
    m &= ~ndimage.binary_opening(ink, structure=np.ones((max(3, int(2.0 * sp)), 1)))  # stems
    m &= ~ndimage.binary_opening(ink, structure=np.ones((1, max(3, int(2.4 * sp)))))  # beams/lines
    lbl, n = ndimage.label(m)
    lo, hi = min(lines) - 0.4 * sp, max(lines) + 0.4 * sp
    out = []
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        ys, xs = sl
        hh, ww = ys.stop - ys.start, xs.stop - xs.start
        area = int((lbl[sl] == i).sum())
        cx, cy = (xs.start + xs.stop) / 2.0, top + (ys.start + ys.stop) / 2.0
        if not (0.4 * sp * sp <= area <= 2.6 * sp * sp):
            continue
        if not (0.6 * sp <= ww <= 2.2 * sp and 0.45 * sp <= hh <= 2.3 * sp):
            continue
        if area < 0.5 * hh * ww:                  # a rest is fairly solid
            continue
        if not (lo <= cy <= hi):                  # sits within the staff
            continue
        if min((abs(cx - c) for c, _, _ in noteheads), default=9e9) < 1.4 * sp:
            continue                              # on a notehead → it's a note, not a rest
        out.append((cx, cy, area, [xs.start, top + ys.start, xs.stop, top + ys.stop]))
    # Merge fragments of one rest (a staff line can split a rest into two vertical
    # pieces with near-identical x): same column, close in y → one glyph.
    out.sort()
    merged = []
    for cx, cy, area, bb in out:
        if merged and abs(cx - merged[-1][0]) < 1.3 * sp and abs(cy - merged[-1][1]) < 2.2 * sp:
            pcx, pcy, parea, pbb = merged[-1]
            nb = [min(pbb[0], bb[0]), min(pbb[1], bb[1]), max(pbb[2], bb[2]), max(pbb[3], bb[3])]
            merged[-1] = ((nb[0] + nb[2]) / 2.0, (nb[1] + nb[3]) / 2.0, parea + area, nb)
        else:
            merged.append((cx, cy, area, bb))
    return [(cx, cy, bb) for cx, cy, area, bb in merged]


def _assign_pitches(noteheads, oemer_pitches, bottom_y, bottom_dia, sp, gray=None):
    """One pitch per detected notehead (x-sorted): the OMR pitch whose staff height
    matches (within ~1 step, keeping its accidental), else the height-derived
    natural — so a note OMR missed is recovered and an out-of-order run is fixed.
    Accidentals are resolved against real glyphs WITH carry-through (an accidental
    applies to later same-pitch notes in the bar): a sharp/flat with a detected
    glyph is kept and remembered; one with neither a glyph nor a carried accidental
    is stripped (oemer hallucination); a carried accidental is re-applied."""
    pool = [(_diatonic(p), p) for p in oemer_pitches if _diatonic(p) is not None]
    used = [False] * len(pool)
    active = {}                                  # (letter+octave) -> '#'/'b' seen this bar
    out = []
    for cx, cy, _a in noteheads:
        want = bottom_dia + (bottom_y - cy) * 2.0 / sp
        best, cost = -1, 0.7                    # only if OMR agrees with the notehead
        #                                         (>~half a step apart → trust the notehead)
        for i, (pd, _ps) in enumerate(pool):
            if used[i]:
                continue
            if abs(pd - want) < cost:
                cost, best = abs(pd - want), i
        if best >= 0:
            used[best] = True
            pitch = pool[best][1]
        else:
            pitch = _ydia_pitch(cy, bottom_y, bottom_dia, sp)
        if gray is not None:
            mm = re.match(r'^([A-G])([#b]?)(-?\d+)$', pitch)
            if mm:
                letter, acc, octv = mm.group(1), mm.group(2), mm.group(3)
                key = letter + octv
                if _has_accidental(gray, cx, cy, sp):
                    acc = acc or '#'             # real glyph here
                    active[key] = acc            # ...remember for the rest of the bar
                elif acc:
                    if active.get(key) != acc:   # no glyph and not carried → hallucination
                        acc = ''
                elif key in active:              # natural read but accidental still in force
                    acc = active[key]
                pitch = f'{letter}{acc}{octv}'
        out.append(pitch)
    return out


def _repeat_dots(gray, treble, bass, sp_t, sp_b, side):
    """True if a pair of repeat dots (flanking the middle staff line, same column,
    ~1 space apart) sits on the given `side` ('right' = repeat-end, 'left' =
    repeat-start). Dots are small isolated blobs — a barline is one tall blob and a
    notehead is much bigger, so size + the paired geometry tells them apart."""
    import numpy as np
    from scipy import ndimage
    H, W = gray.shape
    sl_x = slice(int(0.80 * W), W) if side == 'right' else slice(0, int(0.20 * W))
    for lines, s in ((treble, sp_t), (bass, sp_b)):
        mid = lines[2]
        sub = (gray[:, sl_x] < 110).copy()
        for ly in lines:                       # drop staff lines
            sub[max(0, int(ly) - 1):int(ly) + 2, :] = False
        lbl, nb = ndimage.label(sub)
        if not nb:
            continue
        dots = []
        for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
            ys, xs = sl
            hh, ww, area = ys.stop - ys.start, xs.stop - xs.start, int((lbl[sl] == i).sum())
            # dot = small, round, isolated blob (not a tall barline / big notehead)
            if ((0.12 * s) ** 2 < area < (0.5 * s) ** 2 and hh < 0.7 * s and ww < 0.7 * s
                    and 0.5 < (hh + 1) / (ww + 1) < 2.0):
                dots.append(((xs.start + xs.stop) / 2.0, (ys.start + ys.stop) / 2.0))
        # the two repeat dots sit in the SAME x-column, ~1 space apart, one in the
        # space just above and one just below the middle line.
        for xa, ya in dots:
            for xb, yb in dots:
                if (abs(xa - xb) < 0.5 * s and 0.6 * s < yb - ya < 1.5 * s
                        and abs(ya - (mid - 0.5 * s)) < 0.45 * s
                        and abs(yb - (mid + 0.5 * s)) < 0.45 * s):
                    return True
    return False


def detect_repeat_end(gray, treble, bass, sp_t, sp_b):
    return _repeat_dots(gray, treble, bass, sp_t, sp_b, 'right')


def detect_repeat_start(gray, treble, bass, sp_t, sp_b):
    return _repeat_dots(gray, treble, bass, sp_t, sp_b, 'left')


def _beam_span(gray, lines, sp):
    """x-range (x0, x1) of the beam(s) over a staff, or None. A beam is a THICK
    AND WIDE dark bar (2D opening) — the thickness rejects thin ledger/staff lines,
    the width rejects stems and noteheads."""
    import numpy as np
    from scipy import ndimage
    top = max(0, int(min(lines) - 4 * sp))
    bot = min(gray.shape[0], int(max(lines) + 4 * sp))
    sub = (gray[top:bot] < 110).copy()
    for ly in lines:
        r = int(ly) - top
        sub[max(0, r - 2):r + 3, :] = False
    th, k = max(3, int(0.22 * sp)), max(4, int(0.8 * sp))
    beam = ndimage.binary_opening(sub, structure=np.ones((th, k), bool))
    cols = np.where(beam.any(axis=0))[0]
    if cols.size < k:
        return None
    segs = np.split(cols, np.where(np.diff(cols) > 0.8 * sp)[0] + 1)
    seg = max(segs, key=len)
    if len(seg) < 1.2 * sp:
        return None
    return int(seg[0]), int(seg[-1])


_REST16 = {1: '16', 2: '8', 3: '8.', 4: 'q', 6: 'q.'}
_TAG16 = {1: '16', 2: '8', 3: '8.', 4: 'q', 6: 'q.', 8: 'h'}   # sixteenths → note tag
def _rests16(gap, start):
    """Rest tokens for `gap` sixteenths starting at `start` (sixteenths), split on
    the eighth-note beat grid."""
    out, cur, rem = [], start, gap
    while rem > 0:
        to_grid = 2 - (cur % 2) or 2
        chunk = min(to_grid, rem)
        out.append(f'R({_REST16.get(chunk, "16")})')
        cur += chunk
        rem -= chunk
    return out


def melody_rhythm(gray, treble, sp, pitches, meter_q, downbeat_dotted=False, downbeat_quarter=False):
    """Beam-aware melody rhythm. If the downbeat note is a flagged eighth SEPARATE
    from a beamed run (the Alberti-style "note, 16th-rest, run" figure), emit
    `note(8) R(16) run(16…)` — recovering the rest the plain absorb fit drops.
    `downbeat_dotted` (from oemer's detected augmentation dot) means the downbeat is
    a HELD dotted note, not eighth+rest → fall back to absorb (no rest).
    Returns a token string, or None to fall back to fit_to_meter."""
    nh = detect_noteheads(gray, treble, sp)
    beam = _beam_span(gray, treble, sp)
    if not nh or not beam:
        return None
    # Pick the REAL notes by structure: the run is the noteheads under the beam,
    # the downbeat is the leftmost notehead before it. Anything between them (a
    # 16th-rest glyph, stray ink) is ignored — so a rest read as a notehead no
    # longer breaks the count.
    bx0, bx1 = beam[0] - 0.4 * sp, beam[1] + 0.4 * sp
    run = [b for b in nh if bx0 <= b[0] <= bx1]
    left = [b for b in nh if b[0] < bx0]
    if not left or len(run) < 2:               # need a separate downbeat + beamed run
        return None
    meter16 = int(round(meter_q / 0.25))
    # Detected noteheads drive the count/positions (recovers notes OMR missed);
    # OMR supplies the downbeat pitch (read first, reliable) and any accidentals.
    db_pitch = pitches[0] if pitches else _ydia_pitch(left[0][1], treble[-1], _TREBLE_BOTTOM, sp)
    if ('#' in db_pitch or 'b' in db_pitch) and not _has_accidental(gray, left[0][0], left[0][1], sp):
        db_pitch = _ydia_pitch(left[0][1], treble[-1], _TREBLE_BOTTOM, sp)
    run_pitches = _assign_pitches(run, pitches[1:], treble[-1], _TREBLE_BOTTOM, sp, gray)
    # Held downbeat (no rest): a dotted note, OR a plain quarter over a 2-note run
    # (quarter + 2 sixteenths fills the bar; a 3-note run would overflow, so that
    # stays the detached accompaniment figure).
    held = downbeat_dotted or (downbeat_quarter and len(run) == 2)
    if held:
        down = meter16 - len(run)              # held: downbeat absorbs, NO rest
        if down < 2 or down not in _TAG16:
            return None
        toks = [f'{db_pitch}({_TAG16[down]})'] + [f'{p}(16)' for p in run_pitches]
    else:
        used = 2 + len(run)                     # detached: eighth + rest + run
        if used > meter16:
            return None
        toks = [f'{db_pitch}(8)'] + _rests16(meter16 - used, 2) + [f'{p}(16)' for p in run_pitches]
    return ' '.join(toks)


def bass_rhythm(gray, bass, sp, oemer_pitches, meter_q):
    """Recover the bass arpeggio from detected noteheads (oemer often misreads the
    bass — wrong octaves, phantom chords). Group noteheads into onsets by x (chords
    share an x), pitch each from staff height (OMR pitch where it matches, keeping
    real accidentals), and lay them as 16ths + a beat-grid trailing rest.
    Returns a token string, or None to fall back to fit_to_meter."""
    nh = detect_noteheads(gray, bass, sp)
    if len(nh) < 2:
        return None
    onsets = []                                 # cluster noteheads sharing an x = a chord
    for b in nh:
        if onsets and b[0] - onsets[-1][-1][0] <= 0.6 * sp:
            onsets[-1].append(b)
        else:
            onsets.append([b])
    meter16 = int(round(meter_q / 0.25))
    if not (2 <= len(onsets) <= meter16):
        return None
    flat = [p for tok in oemer_pitches for p in tok.split('+')]
    toks = []
    for grp in onsets:
        ps = _assign_pitches(grp, flat, bass[-1], _BASS_BOTTOM, sp, gray)
        ps = sorted(set(ps), key=lambda p: _diatonic(p) or 0)   # chord low→high
        toks.append('+'.join(ps) + '(16)')
    if len(onsets) < meter16:
        toks += _rests16(meter16 - len(onsets), len(onsets))
    return ' '.join(toks)


def _mapper(five, bottom_diatonic):
    """pitch diatonic-index -> y on this staff (5 line-centres top→bottom)."""
    spacing = (five[-1] - five[0]) / 4.0 or 1.0
    bottom_y = five[-1]
    return lambda dia: bottom_y - (dia - bottom_diatonic) * (spacing / 2.0), spacing


def _detect_structural(gray, treble, bass, tsp, bsp, prefix_end, timesig, allow_tempo=True):
    """Detect the non-note glyphs in a bar — clef, time signature, tempo/character
    text, dynamic marking — each as a real bounding box. The time-sig value is known
    and the clef by staff; text/dynamics can't be OCR'd so they're tagged by type.
    Returns elements with pixel bbox [x0,y0,x1,y1]."""
    import numpy as np
    from scipy import ndimage
    H, W = gray.shape
    out = []

    # ── Clef + time signature: tall solid glyphs in the prefix before note 1 ──
    # Only a SYSTEM-START bar has them, which means a wide prefix (clef + time-sig
    # ≈ 4+ staff-spaces). A mid-system bar's first note sits right after the
    # barline (~1-2 spaces of margin), so don't mistake that margin for a clef.
    if int(prefix_end) > 4.0 * tsp:
        for staff, lines, sp, sym in (('treble', treble, tsp, '\U0001D11E'),
                                      ('bass', bass, bsp, '\U0001D122')):
            top, bot = max(0, int(min(lines) - 1.4 * sp)), min(H, int(max(lines) + 1.4 * sp))
            # Pull the prefix back off the first notehead so its left edge doesn't leak
            # in as a phantom glyph after the time-sig.
            pre = max(3, int(prefix_end - 0.6 * sp))
            band = gray[top:bot, :pre] < 120
            # Column ink density: staff lines give every column a baseline; the clef
            # and time-sig sit well above it. (A clef crosses the staff lines, so
            # masking them would fragment it — column density keeps it whole.)
            col = band.sum(axis=0).astype(float)
            if col.size < 3:
                continue
            base = float(np.percentile(col, 30))
            hot = np.where(col > base + 0.4 * sp)[0]
            if hot.size < 2:
                continue
            # Group hot columns into glyph clusters. The gap threshold (~0.85 space) is
            # wide enough that a clef's own internal density dips (e.g. the treble clef's
            # loops at high resolution) stay ONE cluster, but the clef→time-sig gap still
            # separates them.
            groups, s, prev = [], int(hot[0]), int(hot[0])
            for x in hot[1:]:
                x = int(x)
                if x - prev > 0.85 * sp:
                    groups.append((s, prev))
                    s = x
                prev = x
            groups.append((s, prev))
            # Drop sub-space noise slivers and the thin left-edge barline spike (a
            # narrow group hugging x≈0), but KEEP wider clef sub-parts.
            groups = [g for g in groups if (g[1] - g[0]) >= 0.3 * sp
                      and not ((g[1] - g[0]) < 0.55 * sp and g[1] < 0.3 * sp)]
            if not groups:
                continue

            def _yb(x0, x1):
                ys = np.where(band[:, x0:x1 + 1].any(axis=1))[0]
                return (top + int(ys.min()), top + int(ys.max())) if ys.size else (top, bot)
            # clef = first cluster; time-sig = the next cluster (close behind it).
            clef = groups[0]
            tsg = groups[1] if len(groups) > 1 else None
            cy0, cy1 = _yb(clef[0], clef[1])
            out.append({'type': 'clef', 'voice': staff, 'label': sym,
                        'bbox': [clef[0], cy0, clef[1], cy1]})
            if tsg and tsg[0] - clef[1] < 6.0 * sp:    # the next glyph = the time-sig
                ty0, ty1 = _yb(tsg[0], tsg[1])
                out.append({'type': 'timesig', 'voice': staff, 'label': timesig,
                            'bbox': [tsg[0], ty0, tsg[1], ty1]})

    def _text_blob(y0, y1, label, typ, min_w):
        """Union of text-like components in a horizontal band, excluding full-width
        staff/beam lines and edge bleed — so a marking is boxed, not the whole row."""
        y0, y1 = max(0, y0), min(H, y1)
        if y1 - y0 < 6:
            return
        lbl, n = ndimage.label(gray[y0:y1] < 110)
        boxes = []
        for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
            ys, xs = sl
            w = xs.stop - xs.start
            if int((lbl[sl] == i).sum()) < 30 or w > 0.5 * W:
                continue                          # noise / staff-line-or-beam
            if xs.start < 0.02 * W or xs.stop > 0.98 * W:
                continue                          # bleed from an adjacent bar
            boxes.append((xs.start, y0 + ys.start, xs.stop, y0 + ys.stop))
        if not boxes:
            return
        x0, x1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
        if x1 - x0 < min_w:
            return
        out.append({'type': typ, 'label': label,
                    'bbox': [x0, min(b[1] for b in boxes), x1, max(b[3] for b in boxes)]})

    if allow_tempo:   # the tempo/title sits above the FIRST system; on later bars it
        #               just bleeds in from the row above, so only look on bar 1.
        _text_blob(int(min(treble) - 4 * tsp), int(min(treble) - 1.3 * tsp), 'tempo', 'text', 2.0 * tsp)
    _text_blob(int(max(treble) + 1.7 * tsp), int(min(bass) - 1.7 * bsp), 'dynamic', 'dynamic', 1.0 * tsp)
    return out


def bar_layout(crop_path: str, voices: Dict[str, str], bar_n=None) -> Optional[Dict]:
    """Geometry for one bar. `voices` maps role->token string; melody/melody2 go
    on the treble staff, bass/bass2 on the bass staff. `bar_n` (1-indexed) gates
    once-per-piece markings (tempo). Returns dict with image size, staff bands, and
    a flat list of placed elements (normalised coords)."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(crop_path).convert('L')
    W, H = img.size
    gray = np.asarray(img)
    staves = _two_staves(detect_staff_lines(gray))
    if not staves:
        return None
    treble, bass = staves
    ty, tsp = _mapper(treble, _TREBLE_BOTTOM)
    by, bsp = _mapper(bass, _BASS_BOTTOM)

    # Horizontal note region: project ink in the staff band but MASK OUT the
    # staff-line rows (they span the full width and would otherwise light up
    # every column). What remains is noteheads/stems/rests, whose left/right
    # extent bounds the playable width.
    band_top = max(0, int(min(treble) - tsp))
    band_bot = min(H, int(max(bass) + bsp))
    rowmask = np.zeros(H, dtype=bool)
    rowmask[band_top:band_bot] = True
    for ly in treble + bass:                  # blank ±2px around each staff line
        rowmask[max(0, int(ly) - 2):min(H, int(ly) + 3)] = False
    coldark = ((gray < 120) & rowmask[:, None]).sum(axis=0)
    thr = max(4, 0.05 * (band_bot - band_top))
    m = int(0.03 * W)
    strong = [x for x in range(m, W - m) if coldark[x] > thr]
    if strong and (strong[-1] - strong[0]) > 0.35 * W:
        x_lo, x_hi = strong[0], strong[-1]
    else:                                     # inconclusive → typical bar margins
        x_lo, x_hi = int(0.08 * W), int(0.92 * W)
    usable = x_hi - x_lo

    staff_bands = {
        'treble': {'top': (min(treble) - tsp) / H, 'bottom': (max(treble) + tsp) / H,
                   'mid': (sum(treble) / 5) / H},
        'bass': {'top': (min(bass) - bsp) / H, 'bottom': (max(bass) + bsp) / H,
                 'mid': (sum(bass) / 5) / H},
    }

    # Actual notehead x-positions per staff — used to place labels on the real
    # notes instead of guessing x from onset (onset only works when notes fill
    # the bar evenly; it breaks on system-start bars where a clef+time prefix
    # crams the notes to the right). Cluster blobs that share an x (a chord).
    def _clusters(nhlist, sp):
        cl = []
        for cx, _cy, _a in nhlist:
            if cl and cx - cl[-1][-1] <= 0.6 * sp:
                cl[-1].append(cx)
            else:
                cl.append([cx])
        return [sum(c) / len(c) for c in cl]
    clusters = {'treble': _clusters(detect_noteheads(gray, treble, tsp), tsp),
                'bass': _clusters(detect_noteheads(gray, bass, bsp), bsp)}
    primary = {'treble': 'melody', 'bass': 'bass'}

    elements = []
    # (No clef detection: on this engraving a leading beamed-note group looks the
    # same to a tall/wide CV test as a clef, so it can't be told apart per-bar
    # without full OMR. Notes + rests + staff-accurate placement are reliable.)
    plan = [('melody', 'treble', ty, tsp), ('melody2', 'treble', ty, tsp),
            ('bass', 'bass', by, bsp), ('bass2', 'bass', by, bsp)]
    for role, staff, ymap, sp in plan:
        events, total = _parse_events(voices.get(role, ''))
        if not events:
            continue
        span = total or 1.0
        # Map detected notehead clusters onto this voice's note events (in order)
        # when their counts reconcile; extra leftmost clusters (a clef) are dropped.
        xmap = {}
        if role == primary[staff]:
            note_idx = [i for i, e in enumerate(events) if not e[3]]
            cl = clusters[staff]
            use = cl if len(cl) == len(note_idx) else (
                cl[-len(note_idx):] if note_idx and len(cl) > len(note_idx) else None)
            if use is not None:
                for k, i in enumerate(note_idx):
                    xmap[i] = use[k] / W
        for i, (heads, onset, dur, is_rest) in enumerate(events):
            xf = xmap.get(i, (x_lo + (onset / span) * usable) / W)
            cx = xf * W
            if is_rest:
                continue              # rests are placed from detected glyphs below, not onset
            dias = [d for d in (_diatonic(h) for h in heads) if d is not None]
            if not dias:
                continue
            ys = [ymap(d) for d in dias]
            yt, yb = min(ys), max(ys)
            rad = 0.62 * sp                       # box scales with the staff (dynamic size)
            elements.append({
                'voice': staff, 'type': 'note', 'label': ' '.join(heads),
                'x': round(xf, 4),
                'y': round((sum(ys) / len(ys)) / H, 4),
                'yTop': round(yt / H, 4), 'yBot': round(yb / H, 4), 'grace': dur == 0.0,
                'bbox': [round((cx - rad) / W, 4), round((yt - rad) / H, 4),
                         round((cx + rad) / W, 4), round((yb + rad) / H, 4)]})
            # Accidentals: a ♯/♭/♮ glyph sits just left of its notehead. Box it and
            # link it (a leader line) to the note it alters, so the pairing is shown.
            # Per-head so a chord's accidentals each box near their own notehead.
            for h in heads:
                d = _diatonic(h)
                if d is None:
                    continue
                hy = ymap(d)
                ab = _accidental_bbox(gray, cx, hy, sp)
                if not ab:
                    continue
                sym = '♯' if '#' in h else ('♭' if re.match(r'^[A-Ga-g]b', h) else '♮')
                elements.append({
                    'voice': staff, 'type': 'accidental', 'label': sym, 'note': h,
                    'link': [round(cx / W, 4), round(hy / H, 4)],
                    'bbox': [round(ab[0] / W, 4), round(ab[1] / H, 4),
                             round(ab[2] / W, 4), round(ab[3] / H, 4)]})

    # Prefix = space before the first PLACED note (note placement already drops the
    # clef's false notehead detections via rightmost-N), so the clef/time-sig live in
    # [0, prefix_end] on a system-start bar.
    note_xs = [el['x'] for el in elements if el['type'] == 'note']
    prefix_end = (min(note_xs) * W) if note_xs else x_lo
    elements += _detect_structural(gray, treble, bass, tsp, bsp, prefix_end,
                                   voices.get('timeSig', '3/8'),
                                   allow_tempo=(bar_n is None or bar_n == 1))

    # Rests: box the actual rest GLYPHS (compact, solid, isolated blobs found by
    # detect_rests) rather than guessing from onset — onset lands on noteheads, and
    # many rest tokens are rhythm-fill with no engraved glyph. Run a staff only when it
    # has rest tokens or no notes, so a note-only staff can't sprout phantom rests.
    for staff, lines, sp in (('treble', treble, tsp), ('bass', bass, bsp)):
        roles = ('melody', 'melody2') if staff == 'treble' else ('bass', 'bass2')
        n_rest_tok = sum(1 for r in roles for ev in _parse_events(voices.get(r, ''))[0] if ev[3])
        has_note = any(e.get('voice') == staff and e['type'] == 'note' for e in elements)
        if not n_rest_tok and has_note:
            continue
        found = detect_rests(gray, lines, sp, detect_noteheads(gray, lines, sp))
        cap = n_rest_tok if n_rest_tok else 2     # tokens say how many; empty staff → ≤2
        if len(found) > cap:                      # keep the most solid (largest-area) glyphs
            found = sorted(sorted(found, key=lambda r: -(r[2][2] - r[2][0]) * (r[2][3] - r[2][1]))[:cap])
        for cx, cy, bb in found:
            elements.append({'type': 'rest', 'voice': staff, 'label': 'rest',
                             'x': round(cx / W, 4), 'y': round(cy / H, 4),
                             'bbox': [bb[0], bb[1], bb[2], bb[3]]})

    # normalise structural bboxes (detector returns pixels) + an anchor x/y
    for el in elements:
        if 'bbox' in el and max(el['bbox']) > 1.5:    # still in pixels
            x0, y0, x1, y1 = el['bbox']
            el['bbox'] = [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)]
        if 'x' not in el and 'bbox' in el:
            el['x'] = round((el['bbox'][0] + el['bbox'][2]) / 2, 4)
            el['y'] = round((el['bbox'][1] + el['bbox'][3]) / 2, 4)
    return {'w': W, 'h': H, 'staves': staff_bands, 'elements': elements,
            'noteRegion': {'lo': round(x_lo / W, 4), 'hi': round(x_hi / W, 4)}}


def debug_render(crop_path: str, voices: Dict[str, str], out_path: str):
    """Draw detected staves + placed labels onto the crop for visual validation."""
    from PIL import Image, ImageDraw
    lay = bar_layout(crop_path, voices)
    img = Image.open(crop_path).convert('RGB')
    if not lay:
        img.save(out_path)
        return False
    W, H = img.size
    d = ImageDraw.Draw(img)
    COL = {'rest': (220, 60, 60), 'clef': (150, 60, 200), 'timesig': (20, 150, 90),
           'dynamic': (0, 150, 170), 'text': (120, 120, 120)}
    for el in lay['elements']:
        col = COL.get(el['type']) or ((37, 99, 235) if el.get('voice') == 'treble' else (176, 96, 32))
        bb = el.get('bbox')
        if bb:
            d.rectangle([bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H], outline=col, width=3)
            d.text((bb[0] * W, bb[1] * H - 12), el['label'], fill=col)
    img.save(out_path)
    return True


if __name__ == '__main__':
    import sys, json
    if len(sys.argv) >= 3 and sys.argv[1] == 'debug':
        # debug <crop> <out> <melody> [bass]
        v = {'melody': sys.argv[3] if len(sys.argv) > 3 else '',
             'bass': sys.argv[4] if len(sys.argv) > 4 else ''}
        print(debug_render(sys.argv[2], v, sys.argv[2].replace('.png', '_dbg.png')))
    else:
        print(json.dumps(bar_layout(sys.argv[1], {'melody': sys.argv[2] if len(sys.argv) > 2 else ''}), indent=1))
