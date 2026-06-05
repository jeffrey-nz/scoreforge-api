"""omr2 — EXPERIMENTAL staged single-measure note recogniser (NOT wired in).

Status: research WIP. Stages 1-3 below are implemented; stem/notehead detection
is ~50-60% on dense beamed runs, so this is NOT used by the pipeline — it stays
isolated from the trustworthy reading aid in ``omr``. Reliable note recognition
needs an ML OMR (e.g. oemer) feeding ``omer_import``; this module is the
classical-CV fallback / sandbox for improving it.

Design goal: many small, individually-reliable steps instead of one hard
"read the measure" leap. Each stage is testable alone; the bar-duration
constraint cross-checks the result.

Pipeline per measure crop (one staff band at a time):
  1. staff grid     -> y -> pitch mapping            (omr._grid, reliable)
  2. stems          -> note count + x positions      (vertical morphology)   [done]
  3. noteheads      -> pitch (blob at each stem end vs the grid)              [done, noisy]
  4. beams/flags    -> duration per note                                     [TODO]
  5. rests          -> rest glyphs in the gaps                               [TODO]
  6. assemble+check -> tokens; validate sum vs meter, flag low confidence    [TODO]
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import omr   # reuse staff detection + pitch grid + name helpers


def _prep(gray, y_lo, y_hi, sp):
    """Binary (ink=255) with staff lines removed, for one staff band."""
    sub = gray[max(0, y_lo):y_hi, :].astype(np.uint8)
    _, bw = cv2.threshold(sub, 115, 255, cv2.THRESH_BINARY_INV)
    # remove long horizontal staff lines so vertical morphology sees only stems
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(sp * 2.5)), 1)))
    return cv2.subtract(bw, horiz)


def detect_stems(bw, sp):
    """Stage 2: vertical ink runs ~>=1 staff-space tall and thin = stems.
    Returns [(x_center, y_top, y_bottom)] left->right — one per note."""
    # fill only the thin holes a staff line left in a stem (~line thickness),
    # NOT the wider gaps between separate noteheads, then keep tall-thin runs
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(sp * 0.18)))))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(sp * 1.15))))
    vert = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)
    n, _l, stats, _c = cv2.connectedComponentsWithStats(vert, connectivity=8)
    stems = []
    for i in range(1, n):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if h < sp * 1.3 or w > sp * 0.6:      # too short or too thick to be a stem
            continue
        stems.append((x + w // 2, y, y + h))
    stems.sort(key=lambda s: s[0])
    # merge stems closer than ~0.5sp (same note detected twice)
    merged = []
    for s in stems:
        if merged and s[0] - merged[-1][0] < sp * 0.5:
            continue
        merged.append(s)
    return merged


def detect_notes(bw, sp, grid):
    """Stages 2-3: stems that have a notehead at one end = real notes. Rejects
    accidentals/barlines (no head). Returns [(x, pitch_name, filled)] L->R."""
    H, W = bw.shape
    notes = []
    for (sx, ytop, ybot) in detect_stems(bw, sp):
        x0, x1 = max(0, int(sx - sp * 0.7)), min(W, int(sx + sp * 0.7))
        # a beam continues horizontally past the head; sample a wider strip to
        # tell a local notehead (ink stays near the stem) from a beam end.
        wx0, wx1 = max(0, int(sx - sp * 1.6)), min(W, int(sx + sp * 1.6))
        best = None
        for (lo, hi) in ((ybot - int(sp * 0.3), min(H, ybot + int(sp * 0.9))),
                         (max(0, ytop - int(sp * 0.9)), ytop + int(sp * 0.3))):
            lo = max(0, lo)
            band = bw[lo:hi, x0:x1]
            wide_band = bw[lo:hi, wx0:wx1]
            if band.size == 0:
                continue
            rw = (band > 0).sum(axis=1)
            wide = np.where(rw >= 0.55 * (x1 - x0))[0]
            if len(wide) < 2:
                continue
            # reject beam ends: ink spanning most of the wide strip = a beam
            wide_rows = (wide_band > 0).sum(axis=1)
            if wide_rows.max() >= 0.85 * (wx1 - wx0):
                continue
            hy = (wide[0] + wide[-1]) / 2 + lo
            fill = band[wide[0]:wide[-1] + 1].mean()
            if best is None or len(wide) * fill > best[0]:
                best = (len(wide) * fill, hy, fill)
        if best is None:
            continue                                  # no head -> accidental/barline/beam
        _s, hy, fill = best
        dia = omr._y_to_dia(hy, grid)
        notes.append((sx, omr._dia_to_name(dia), fill > 140))
    return notes


if __name__ == '__main__':
    import sys, json
    gray = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
    grp = omr._group_staves(omr._find_staff_lines(binary))
    if not grp:
        print('no staff'); sys.exit()
    treble, bass = grp
    sp = (sorted(treble)[-1] - sorted(treble)[0]) / 4.0
    mid = int((sorted(treble)[-1] + sorted(bass)[0]) / 2)
    tb = _prep(gray, 0, mid, sp)
    bb = _prep(gray, mid, gray.shape[0], sp)
    tg = omr._grid(treble, 30)
    bg = omr._grid([y - mid for y in bass], 18)  # bass band starts at mid
    tnotes = detect_notes(tb, sp, tg)
    bnotes = detect_notes(bb, sp, bg)
    print('MEL :', ' '.join(p for _x, p, _f in tnotes))
    print('BASS:', ' '.join(p for _x, p, _f in bnotes))
