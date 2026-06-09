"""Create the SECOND Fur Elise import: a FULLY MECHANICAL transcription — no AI, no
hand-edits, and NOT cloning the hand-verified original job. Everything is derived from
the source PDF + the cached oemer MusicXML:

  1. render each page from the source PDF (app/core/sources/fur_elise_woo59.pdf);
  2. segment it into systems mechanically — try BOTH the autocorrelation detector
     (ai_transcribe._detect_systems) and a staff-line-grouping detector, and keep the
     one whose detected bar count best matches oemer's measure count for the page
     (two independent mechanical signals agreeing);
  3. detect barlines per system and crop each bar straight off the rendered page;
  4. read notes from oemer (cached MusicXML) with the beam-aware rhythm recovery;
  5. infer the repeat/volta/pickup structure from the crops;
  6. write a self-contained job folder + state.json from scratch.

The only inputs are the gitignored source PDF and the cached oemer output — nothing
from fur_elise_p1 (the corrected first version).
"""
import sys, os, json, uuid, time, shutil, re, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'app', 'core'))
from app.core import omer_import

SHOWCASE = os.path.join(ROOT, '..', 'procmusic', 'dashboard', 'showcase-midi')
DST_DIR = os.path.join(SHOWCASE, 'fur_elise_mech')
MXML_DIR = os.path.join(HERE, 'fixtures', 'omr_out')
PDF_SRC = os.path.join(ROOT, 'app', 'core', 'sources', 'fur_elise_woo59.pdf')
SEG_DPI = 150.0     # segmentation is verified at this DPI (staff/barline detection)
CROP_DPI = 450.0    # crops rendered high-res so the small rest/accidental glyphs read
#                     (matches the ~500-DPI resolution the glyph detection was tuned on)


def _raw_pitches(s):
    out = []
    for t in str(s or '').split():
        head = t.split('(')[0]
        if head and head[:1] not in 'Rr':
            out.append(head)
    return out


# ── Mechanical page segmentation ────────────────────────────────────────────────
def render_page(pdf, page, dst_png, dpi):
    import fitz
    doc = fitz.open(pdf)
    pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
    doc.close()
    pix.save(dst_png)


def _staff_line_systems(gray):
    """Group grand-staff systems straight from the staff lines: detect full-width
    horizontal lines, cluster them into staves (5 close lines), then join staves into
    systems using the bimodal inter-staff gap (treble↔bass within a system is smaller
    than the gap between systems; split at the largest natural jump). Robust to the
    page-1 title block that defeats the autocorrelation detector."""
    import numpy as np
    H, W = gray.shape
    # 0.4·W (not 0.5) catches faint last-system staff lines that would otherwise drop a
    # staff and break the grand-staff pairing (page 2/3 last systems).
    rows = np.where((gray < 140).sum(axis=1) > 0.4 * W)[0]
    if len(rows) < 10:
        return None
    lines, s, prev = [], int(rows[0]), int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r - prev > 4:
            lines.append((s + prev) // 2); s = r
        prev = r
    lines.append((s + prev) // 2)
    gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    small = float(np.median([g for g in gaps if g < 25]) or 10)
    staves, cur = [], [lines[0]]
    for ln in lines[1:]:
        if ln - cur[-1] > 3.0 * small:
            staves.append(cur); cur = [ln]
        else:
            cur.append(ln)
    staves.append(cur)
    staves = [st for st in staves if len(st) >= 3]
    if len(staves) < 2:
        return None
    inter = [staves[i + 1][0] - staves[i][-1] for i in range(len(staves) - 1)]
    sg = sorted(set(inter))
    thresh = 1e9
    if len(sg) >= 2:
        mj, ji = max((sg[i + 1] - sg[i], i) for i in range(len(sg) - 1))
        if mj > 1.5 * small:
            thresh = (sg[ji] + sg[ji + 1]) / 2.0
    if thresh > 1e8:                       # no clear bimodal → assume grand-staff pairs
        groups = [staves[i:i + 2] for i in range(0, len(staves), 2)]
    else:
        groups = [[staves[0]]]
        for i, st in enumerate(staves[1:]):
            (groups.append([st]) if inter[i] > thresh else groups[-1].append(st))
    return [(max(0, int(g[0][0] - 1.6 * small)), min(H, int(g[-1][-1] + 1.6 * small)))
            for g in groups]


def _sys_bar_boxes(img, systems, tmp, page):
    """Per system band, detect barlines → list of (x1,x2) bar boxes."""
    import ai_transcribe as atr
    import numpy as np
    W, H = img.size
    out = []
    for k, (top, bot) in enumerate(systems):
        crop = img.crop((0, max(0, top), W, min(H, bot)))
        strip = os.path.join(tmp, f'_strip_{page}_{k}.png')
        crop.save(strip)
        pos = atr._detect_barline_positions(strip)
        boxes = []
        if pos and len(pos) >= 2:
            # Right-tail recovery: the system's CLOSING barline is sometimes broken (it
            # fails the run-length detector, so no position is emitted past the last bar).
            # If the staff still extends a full bar-width beyond the last barline, add it.
            g = np.asarray(crop.convert('L'))
            bh = g.shape[0]
            colink = (g < 120).sum(axis=0)
            tall = [x for x in range(W) if colink[x] > 0.4 * bh]
            redge = max(tall) if tall else pos[-1]
            widths = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
            med = sorted(widths)[len(widths) // 2] if widths else 0
            if med and redge - pos[-1] > 0.6 * med:
                pos = pos + [min(W, redge)]
            for i in range(len(pos) - 1):
                boxes.append((max(0, pos[i] - 5), min(W, pos[i + 1] + 5)))
        out.append(boxes)
    return out


def segment_page(img, gray, png, oemer_count, tmp, page):
    """Pick system bands. STRONGLY prefer the staff-line detector: its bands are anchored
    to real staff lines (vertically-correct grand staves). The autocorrelation detector
    phase-locks onto the GAPS between systems here (bands span the bottom of one system +
    the top of the next → broken crops), so it's only a last resort if staff-line fails or
    is wildly off. Returns (systems, per-system bar-box lists, total)."""
    import ai_transcribe as atr
    sl = _staff_line_systems(gray)
    if sl:
        boxes = _sys_bar_boxes(img, sl, tmp, page)
        tot = sum(len(bx) for bx in boxes)
        if tot >= 0.8 * oemer_count:          # bands reliable; a small bar-count miss is
            return sl, boxes, tot             # absorbed by per-system measure mapping
    # Fallback only when staff-line detection is unavailable / badly short.
    cands = [c for c in (sl, atr._detect_systems(png)) if c] or [[(0, img.size[1])]]
    best = None
    for sysbands in cands:
        boxes = _sys_bar_boxes(img, sysbands, tmp, page)
        tot = sum(len(bx) for bx in boxes)
        score = abs(tot - oemer_count)
        if best is None or score < best[0]:
            best = (score, sysbands, boxes, tot)
    return best[1], best[2], best[3]


def _distribute(n, counts):
    """Split n oemer measures across systems ∝ their detected bar-box counts."""
    T = sum(counts)
    if T == 0:
        return [0] * len(counts)
    per = [int(round(c * n / T)) for c in counts]
    d = n - sum(per)
    j = 0
    while d != 0 and per:
        k = j % len(per)
        if d > 0:
            per[k] += 1; d -= 1
        elif per[k] > 0:
            per[k] -= 1; d += 1
        j += 1
    return per


def apply_repeat_structure(bars, raw, crops, mq):
    """Detect repeat-end barlines from the crops and infer the repeat structure:
    - mark each detected repeat-end (+ volta 1 on it, volta 2 on the next bar);
    - infer the (un-drawn) repeat-START: bar 1 for the first end, else the bar
      after the previous end's 2nd ending;
    - PICKUP: the piece opens with an anacrusis, so bar 1's notes (laid short, as
      16ths) plus the first repeat-end bar complete one measure — relate them by
      making bar 1 partial and fitting the repeat-end bar to the complement.
    Returns the list of detected repeat-end bar numbers."""
    import numpy as np
    from PIL import Image
    import glyph_detect as gd

    ends = []
    for i, _bar in enumerate(bars):
        crop = crops[i] if i < len(crops) else None
        if not crop or not os.path.exists(crop):
            continue
        gray = np.asarray(Image.open(crop).convert('L'))
        st = gd._two_staves(gd.detect_staff_lines(gray))
        if not st:
            continue
        tre, bas = st
        spt, spb = (tre[-1] - tre[0]) / 4, (bas[-1] - bas[0]) / 4
        if gd.detect_repeat_end(gray, tre, bas, spt, spb):
            ends.append(i + 1)                  # 1-indexed bar number
    if not ends:
        return ends

    for k, e in enumerate(ends):
        bars[e - 1]['repeat_end'] = True
        bars[e - 1]['volta'] = 1
        if e < len(bars):
            bars[e]['volta'] = 2                 # the bar after a 1st ending is the 2nd
        start = 1 if k == 0 else ends[k - 1] + 2  # bar 1, else after prev 2nd ending
        bars[start - 1]['repeat_start'] = True

    # Pickup relationship: bar 1 (anacrusis) ↔ first repeat-end bar.
    e0 = ends[0]
    p1 = _raw_pitches(raw[0].get('melody', '')) if raw else []
    if p1:
        bars[0]['melody'] = ' '.join(f'{p}(16)' for p in p1)   # short pickup, unfilled
        bars[0]['partial'] = True
        comp = mq - len(p1) * 0.25                # remaining beats the end bar completes
        if comp > 0 and e0 - 1 < len(raw):
            bars[e0 - 1]['melody'] = omer_import.fit_to_meter(
                raw[e0 - 1].get('melody', ''), comp, 'absorb')
            bars[e0 - 1]['bass'] = omer_import.fit_to_meter(
                raw[e0 - 1].get('bass', ''), comp, 'trail')
        bars[e0 - 1]['partial'] = True
    return ends


def _beam_aware_melody(crop, raw_mel, mq):
    """Use the crop's beam structure to recover the 'note, rest, run' rhythm;
    fall back to the plain absorb fit when not applicable/confident."""
    import numpy as np
    from PIL import Image
    import glyph_detect as gd
    pitches = _raw_pitches(raw_mel)
    first = next((t for t in raw_mel.split() if t.split('(')[0][:1] not in 'Rr'), '')
    md = re.search(r'\((w|h|q|8|16|32|64)\.\)$', first)
    dotted = bool(md and md.group(1) in ('8', 'q'))
    is_quarter = bool(re.search(r'\(q\)$', first))
    if crop and os.path.exists(crop):
        gray = np.asarray(Image.open(crop).convert('L'))
        st = gd._two_staves(gd.detect_staff_lines(gray))
        if st:
            tre = st[0]
            sp = (tre[-1] - tre[0]) / 4
            r = gd.melody_rhythm(gray, tre, sp, pitches, mq,
                                 downbeat_dotted=dotted, downbeat_quarter=is_quarter)
            if r:
                return r
    return omer_import.fit_to_meter(raw_mel, mq, slack='absorb')


_STEP_MIDI = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
def _bass_garbled(raw_bass):
    """oemer misreads the bass as a tight semitone-cluster chord (e.g. D2+D#2) —
    physically implausible for this arpeggio bass, so trust the crop instead."""
    for tok in raw_bass.split():
        head = tok.split('(')[0]
        if '+' not in head:
            continue
        midis = []
        for p in head.split('+'):
            m = re.match(r'^([A-G])([#b]?)(-?\d+)$', p)
            if m:
                pc = _STEP_MIDI[m.group(1)] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)
                midis.append((int(m.group(3)) + 1) * 12 + pc)
        midis.sort()
        if any(b - a < 2 for a, b in zip(midis, midis[1:])):
            return True
    return False


def _beam_aware_bass(crop, raw_bass, mq):
    """Recover the bass arpeggio from the crop's noteheads when oemer garbled it
    (a semitone-cluster chord); otherwise keep the reliable trail fit."""
    import numpy as np
    from PIL import Image
    import glyph_detect as gd
    oemer = [t.split('(')[0] for t in raw_bass.split() if t.split('(')[0][:1] not in 'Rr']
    if _bass_garbled(raw_bass) and crop and os.path.exists(crop):
        gray = np.asarray(Image.open(crop).convert('L'))
        st = gd._two_staves(gd.detect_staff_lines(gray))
        if st:
            bas = st[1]
            sp = (bas[-1] - bas[0]) / 4
            r = gd.bass_rhythm(gray, bas, sp, oemer, mq)
            if r:
                return r
    return omer_import.fit_to_meter(raw_bass, mq, slack='trail')


def _steps_skeleton():
    idle = lambda: {'status': 'idle', 'pct': 0, 'result': None, 'issues': [], 'log': []}
    s = {k: idle() for k in ('detect', 'read', 'pitch', 'rhythm', 'theory', 'review')}
    s['read'] = {'status': 'done', 'pct': 100, 'result': None, 'issues': [], 'log': []}
    return s


def main():
    from PIL import Image
    mq = omer_import.meter_quarters('3/8')
    if not os.path.exists(PDF_SRC):
        raise SystemExit(f'source PDF not found: {PDF_SRC}')

    keep_id = None
    prev = os.path.join(DST_DIR, '_job', 'state.json')
    if os.path.exists(prev):
        try:
            keep_id = json.load(open(prev, encoding='utf-8')).get('id')
        except Exception:
            keep_id = None

    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    pages_dir = os.path.join(DST_DIR, '_pages')
    job_dir = os.path.join(DST_DIR, '_job')
    os.makedirs(pages_dir); os.makedirs(job_dir)
    mech_pdf = os.path.join(DST_DIR, 'fur_elise.pdf')
    shutil.copy(PDF_SRC, mech_pdf)            # self-contained: own copy of the source

    tmp = tempfile.mkdtemp(prefix='mechseg_')
    bars, raw_all, crop_for, page_ranges, gb = [], [], [], [], 0
    try:
        import omr
    except Exception:
        omr = None

    for page in (1, 2, 3):
        mxml = os.path.join(MXML_DIR, f'fe_page{page}.musicxml')
        if not os.path.exists(mxml):
            print(f'  (page {page}: no MusicXML, skipped)')
            continue
        raw = omer_import.musicxml_to_bars(mxml)
        n = len(raw)
        import numpy as np
        # Two NATIVE renders of the same PDF: 150-DPI for the verified segmentation,
        # 450-DPI for the crops. Same PDF ⇒ coords map by exactly CROP/SEG (×3).
        scale = CROP_DPI / SEG_DPI
        page_png = os.path.join(pages_dir, f'page_{page:02d}.png')
        render_page(PDF_SRC, page, page_png, CROP_DPI)      # high-res page (for crops)
        big = Image.open(page_png).convert('RGB')
        seg_png = os.path.join(tmp, f'seg_{page:02d}.png')
        render_page(PDF_SRC, page, seg_png, SEG_DPI)        # segmentation render
        seg = Image.open(seg_png).convert('RGB')
        gray = np.asarray(seg.convert('L'))

        systems, sysboxes, cvtot = segment_page(seg, gray, seg_png, n, tmp, page)
        # Assign oemer measures to bar boxes PER SYSTEM (so a small count mismatch can't
        # drift across systems): distribute the n measures ∝ each system's box count,
        # then within a system map its measures to its boxes proportionally.
        per = _distribute(n, [len(bx) for bx in sysboxes])
        assign = []                              # one (sys, top, bot, x1, x2) per measure
        for si, ((top, bot), boxes, k) in enumerate(zip(systems, sysboxes, per), start=1):
            for j in range(k):
                if boxes:
                    x1, x2 = boxes[min(len(boxes) - 1, int(j * len(boxes) / max(1, k)))]
                    assign.append((si, top, bot, x1, x2))
                else:
                    assign.append(None)
        assign += [None] * (n - len(assign))     # pad if rounding came up short
        print(f'page {page}: oemer={n} systems={len(systems)} CV bars={cvtot} '
              f'-> {"1:1" if cvtot == n else "scaled"}')

        start = gb + 1
        for i, b in enumerate(raw):
            gb += 1
            crop_path = ''
            sys_i = 0
            if i < len(assign) and assign[i] is not None:
                sys_i, top, bot, x1, x2 = assign[i]
                crop_path = os.path.join(pages_dir, f'page_{page:02d}_bar_{gb:03d}.png')
                big.crop((int(x1 * scale), int(max(0, top) * scale),
                          int(x2 * scale), int(min(seg.height, bot) * scale))).save(crop_path)
                if omr is not None:
                    try:
                        omr.annotate_grid(crop_path,
                                          crop_path.replace('.png', '_grid.png'))
                    except Exception:
                        pass
            bars.append({'n': gb, 'page': page, 'sys': sys_i,
                         'melody': _beam_aware_melody(crop_path, b.get('melody', ''), mq),
                         'bass': _beam_aware_bass(crop_path, b.get('bass', ''), mq)})
            raw_all.append(b)
            crop_for.append(crop_path)
        page_ranges.append((page, start, gb))
    total = gb
    print(f'TOTAL mechanical bars: {total}')
    shutil.rmtree(tmp, ignore_errors=True)

    json.dump({'pages': [{'file': f'page_{p:02d}.png', 'page': p, 'startBar': s,
                          'endBar': e, 'status': 'done'} for (p, s, e) in page_ranges],
               'bars': total},
              open(os.path.join(pages_dir, 'pages.json'), 'w', encoding='utf-8'), indent=2)

    st = {
        'id': keep_id or str(uuid.uuid4()),
        'piece_id': 'fur_elise_mech',
        'pdf_path': os.path.abspath(mech_pdf),
        'title': 'Fur Elise (mechanical OMR)', 'composer': 'Beethoven',
        'bpm': None, 'provider': 'gemini', 'pages_spec': None, 'max_bars': None,
        'time_sig': None, 'key': None, 'engine': 'bridge', 'source': 'pipeline',
        'created': time.time(), 'approved': False,
        'steps': _steps_skeleton(),
        'bars': [{
            'n': bb['n'], 'page': bb['page'], 'sys': bb['sys'],
            'melody': bb['melody'], 'bass': bb['bass'], 'melody2': '', 'bass2': '',
            'issues': [], 'pitch_issues': [], 'rhythm_issues': [],
            'confidence': 0.4, 'verified': False, 'edited': False, 'flags': [],
        } for bb in bars],
        'pages': [{'file': f'page_{p:02d}.png', 'page': p, 'startBar': s, 'endBar': e,
                   'status': 'done'} for (p, s, e) in page_ranges],
        'meta': {'timeSig': '3/8', 'key': 'A minor', 'bpm': 70, 'title': 'Fur Elise (mechanical OMR)'},
    }

    ends = apply_repeat_structure(st['bars'], raw_all, crop_for, mq)
    print(f'repeat-end detected on bars: {ends}')

    json.dump(st, open(os.path.join(job_dir, 'state.json'), 'w', encoding='utf-8'), indent=2)
    print(f"created job id={st['id']} title={st['title']} bars={len(st['bars'])} pages={len(page_ranges)}")
    print(f'dir: {DST_DIR}')


if __name__ == '__main__':
    main()
