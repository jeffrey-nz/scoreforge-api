"""Piece-agnostic fully-mechanical importer: build an importer job for ANY PDF straight
from oemer's cached MusicXML + mechanical page segmentation — no AI, no hand-edits, no
reliance on any corrected version. Reuses the segmentation/crop helpers from
_make_mechanical_job but drops the Fur-Elise-specific repeat/pickup structure.

Usage (after rendering pages to <work>/<stem>_p<N>.png and caching <work>/omr/<stem>_p<N>.musicxml):
  python tests/_make_mechanical_piece.py
Edit the CONFIG block for a different piece.
"""
import sys, os, json, uuid, time, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'app', 'core')); sys.path.insert(0, HERE)
from app.core import omer_import
import _make_mechanical_job as fe   # reuse render_page, segmentation, beam-aware helpers

# ── CONFIG ──────────────────────────────────────────────────────────────────────
PDF_SRC = r'C:/Users/Work/Downloads/IMSLP56442-PMLP01855-Mozart_Werke_Breitkopf_Serie_20_KV545.pdf'
WORK = os.path.join(HERE, 'fixtures', 'k545_work')          # holds k545_p<N>.png + omr/
STEM = 'k545_p'
PIECE_ID = 'k545_mech'
TITLE = 'Mozart K545 (mechanical OMR)'
COMPOSER = 'Mozart'
TIME_SIG = '4/4'                                            # mvt I; later movements differ
KEY = 'C major'
NPAGES = 8
DST_DIR = os.path.join(ROOT, '..', 'procmusic', 'dashboard', 'showcase-midi', PIECE_ID)


def main():
    from PIL import Image
    import numpy as np
    mq = omer_import.meter_quarters(TIME_SIG)
    scale = fe.CROP_DPI / fe.SEG_DPI

    keep_id = None
    prev = os.path.join(DST_DIR, '_job', 'state.json')
    if os.path.exists(prev):
        try: keep_id = json.load(open(prev, encoding='utf-8')).get('id')
        except Exception: pass
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    pages_dir = os.path.join(DST_DIR, '_pages'); job_dir = os.path.join(DST_DIR, '_job')
    os.makedirs(pages_dir); os.makedirs(job_dir)
    shutil.copy(PDF_SRC, os.path.join(DST_DIR, 'source.pdf'))

    tmp = tempfile.mkdtemp(prefix='k545seg_')
    try: import omr
    except Exception: omr = None

    bars, page_ranges, gb = [], [], 0
    for page in range(1, NPAGES + 1):
        mxml = os.path.join(WORK, 'omr', f'{STEM}{page}.musicxml')
        if not os.path.exists(mxml):
            print(f'  (page {page}: no MusicXML yet, skipped)'); continue
        raw = omer_import.musicxml_to_bars(mxml)
        n = len(raw)
        page_png = os.path.join(pages_dir, f'page_{page:02d}.png')
        fe.render_page(PDF_SRC, page, page_png, fe.CROP_DPI)
        big = Image.open(page_png).convert('RGB')
        seg_png = os.path.join(tmp, f'seg_{page:02d}.png')
        fe.render_page(PDF_SRC, page, seg_png, fe.SEG_DPI)
        seg = Image.open(seg_png).convert('RGB'); gray = np.asarray(seg.convert('L'))

        systems, sysboxes, cvtot = fe.segment_page(seg, gray, seg_png, n, tmp, page)
        per = fe._distribute(n, [len(bx) for bx in sysboxes])
        assign = []
        for si, ((top, bot), boxes, k) in enumerate(zip(systems, sysboxes, per), start=1):
            for j in range(k):
                if boxes:
                    x1, x2 = boxes[min(len(boxes) - 1, int(j * len(boxes) / max(1, k)))]
                    assign.append((si, top, bot, x1, x2))
                else:
                    assign.append(None)
        assign += [None] * (n - len(assign))
        print(f'page {page}: oemer={n} systems={len(systems)} CV bars={cvtot} '
              f'-> {"1:1" if cvtot == n else "scaled"}')

        start = gb + 1
        for i, b in enumerate(raw):
            gb += 1
            crop_path, sys_i = '', 0
            if i < len(assign) and assign[i] is not None:
                sys_i, top, bot, x1, x2 = assign[i]
                crop_path = os.path.join(pages_dir, f'page_{page:02d}_bar_{gb:03d}.png')
                big.crop((int(x1 * scale), int(max(0, top) * scale),
                          int(x2 * scale), int(min(seg.height, bot) * scale))).save(crop_path)
                if omr is not None:
                    try: omr.annotate_grid(crop_path, crop_path.replace('.png', '_grid.png'))
                    except Exception: pass
            bars.append({'n': gb, 'page': page, 'sys': sys_i,
                         'melody': fe._beam_aware_melody(crop_path, b.get('melody', ''), mq),
                         'bass': fe._beam_aware_bass(crop_path, b.get('bass', ''), mq)})
        page_ranges.append((page, start, gb))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'TOTAL mechanical bars: {gb}')
    if not page_ranges:
        raise SystemExit('no pages had MusicXML — run oemer first')

    json.dump({'pages': [{'file': f'page_{p:02d}.png', 'page': p, 'startBar': s,
                          'endBar': e, 'status': 'done'} for (p, s, e) in page_ranges],
               'bars': gb},
              open(os.path.join(pages_dir, 'pages.json'), 'w', encoding='utf-8'), indent=2)
    st = {
        'id': keep_id or str(uuid.uuid4()), 'piece_id': PIECE_ID,
        'pdf_path': os.path.abspath(os.path.join(DST_DIR, 'source.pdf')),
        'title': TITLE, 'composer': COMPOSER, 'bpm': None, 'provider': 'gemini',
        'pages_spec': None, 'max_bars': None, 'time_sig': None, 'key': None,
        'engine': 'bridge', 'source': 'pipeline', 'created': time.time(), 'approved': False,
        'steps': fe._steps_skeleton(),
        'bars': [{'n': bb['n'], 'page': bb['page'], 'sys': bb['sys'],
                  'melody': bb['melody'], 'bass': bb['bass'], 'melody2': '', 'bass2': '',
                  'issues': [], 'pitch_issues': [], 'rhythm_issues': [],
                  'confidence': 0.4, 'verified': False, 'edited': False, 'flags': []} for bb in bars],
        'pages': [{'file': f'page_{p:02d}.png', 'page': p, 'startBar': s, 'endBar': e,
                   'status': 'done'} for (p, s, e) in page_ranges],
        'meta': {'timeSig': TIME_SIG, 'key': KEY, 'bpm': 100, 'title': TITLE},
    }
    json.dump(st, open(os.path.join(job_dir, 'state.json'), 'w', encoding='utf-8'), indent=2)
    print(f"created job id={st['id']} bars={len(st['bars'])} pages={len(page_ranges)}\n{DST_DIR}")


if __name__ == '__main__':
    main()
