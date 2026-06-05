"""Staff detection + pitch-grid reading aid for clean piano engravings.

The reliable half of mechanical transcription: find the grand-staff lines in a
measure crop, build a diatonic pitch grid for each staff from its clef, and
render a pitch-LABELLED overlay so a human/Claude reads a note's pitch off the
label instead of counting ledger lines. Pitch/duration *recognition* (the hard,
unreliable half) lives in the experimental ``omr2`` module — kept separate so
this aid stays trustworthy.
"""
from __future__ import annotations
import numpy as np
import cv2

# A "diatonic index" is octave*7 + step, where step is C=0 .. B=6. It counts
# staff positions (line/space), which is exactly what a pitch grid measures.
_LET = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


def _dia_to_name(dia: int) -> str:
    octave, pos = divmod(dia, 7)
    return f'{_LET[pos]}{octave}'


def _find_staff_lines(binary: np.ndarray):
    """Row indices of staff lines: rows whose dark run spans most of the width."""
    h, w = binary.shape
    rowsum = (binary > 0).sum(axis=1)
    thr = 0.55 * w
    rows = np.where(rowsum >= thr)[0]
    if len(rows) == 0:
        return []
    # group contiguous rows into single lines (a printed line is a few px thick)
    lines, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b - a > 3:
            lines.append(int(np.mean(cur))); cur = []
        cur.append(b)
    lines.append(int(np.mean(cur)))
    return lines


def _five_line_groups(lines):
    """Find every run of 5 lines with near-constant spacing (a staff). Robust to
    spurious lines from captions/lyrics/ledger marks, which break the spacing."""
    groups = []
    n = len(lines)
    for i in range(n - 4):
        five = lines[i:i + 5]
        gaps = [five[k + 1] - five[k] for k in range(4)]
        g = sum(gaps) / 4.0
        if g <= 2:
            continue
        if all(abs(x - g) <= 0.28 * g for x in gaps):   # evenly spaced
            groups.append((five, g))
    # de-dupe overlapping windows: keep the one whose spacing is most uniform
    groups.sort(key=lambda fg: fg[0][0])
    dedup = []
    for five, g in groups:
        if dedup and five[0] - dedup[-1][0][0] < g * 2:
            continue
        dedup.append((five, g))
    return dedup


def _group_staves(lines):
    """Split detected lines into the treble (upper) and bass (lower) staff,
    each a 5-line evenly-spaced group."""
    if len(lines) < 10:
        return None
    groups = _five_line_groups(lines)
    if len(groups) >= 2:
        treble = groups[0][0]
        bass = groups[-1][0]
        return treble, bass
    # fallback: largest gap split
    gaps = sorted(((lines[i + 1] - lines[i], i) for i in range(len(lines) - 1)),
                  reverse=True)
    split = gaps[0][1] + 1
    treble, bass = lines[max(0, split - 5):split], lines[split:split + 5]
    if len(treble) == 5 and len(bass) == 5:
        return treble, bass
    return lines[:5], lines[-5:]


def _grid(staff5, bottom_dia):
    """(y0, step, bottom_dia): map y -> diatonic index. staff5 top→bottom."""
    ys = sorted(staff5)                  # ascending y = top line first
    spacing = (ys[-1] - ys[0]) / 4.0     # line-to-line
    step = spacing / 2.0                 # one diatonic step = half a space
    y_bottom = ys[-1]                    # lowest line
    return (y_bottom, step, bottom_dia)


def _y_to_dia(y, grid):
    y_bottom, step, bottom_dia = grid
    return bottom_dia + int(round((y_bottom - y) / step))


# Treble bottom line = E4 (diatonic 4*7+2 = 30); bass bottom line = G2 (2*7+4 = 18).
_TREBLE_BOTTOM_DIA = 30
_BASS_BOTTOM_DIA = 18


def annotate_grid(crop_path: str, out_path: str) -> bool:
    """Write a colour copy of the crop with a pitch-labelled staff grid: a faint
    guide at every diatonic position, with staff lines labelled by pitch name, so
    a reader names a note off the label rather than counting ledger lines.
    Returns True if a grand staff was found and labelled."""
    gray = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return False
    _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    grp = _group_staves(_find_staff_lines(binary))
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not grp:
        cv2.imwrite(str(out_path), vis)
        return False
    treble, bass = grp
    H, W = gray.shape
    # (staff, its bottom-line pitch, diatonic range to draw above/below)
    for staff5, bottom_dia, lo, hi in ((treble, _TREBLE_BOTTOM_DIA, 24, 45),
                                       (bass, _BASS_BOTTOM_DIA, 10, 33)):
        y0, step, bd = _grid(staff5, bottom_dia)
        for dia in range(lo, hi):
            y = int(round(y0 - (dia - bd) * step))
            if not (0 <= y < H):
                continue
            if (dia - bd) % 2 == 0:                # a staff line of this staff
                cv2.line(vis, (34, y), (W, y), (180, 180, 255), 1)
                cv2.putText(vis, _dia_to_name(dia), (1, y + 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 200), 1, cv2.LINE_AA)
            else:                                  # a space — faint guide, no label
                cv2.line(vis, (34, y), (W, y), (235, 235, 235), 1)
    cv2.imwrite(str(out_path), vis)
    return True


if __name__ == '__main__':
    import sys
    ok = annotate_grid(sys.argv[1], sys.argv[2])
    print('grid', ok, '->', sys.argv[2])
