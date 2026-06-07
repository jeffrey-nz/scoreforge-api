"""Create the SECOND Fur Elise import: a fully mechanical (oemer -> omer_import ->
fit_to_meter, no AI, no hand-edits) transcription of page 1, as its own job that
shows in the importer next to the hand-verified original.

Clones the original job's folder (so the page images / per-bar source crops still
resolve) and swaps in the mechanical bars.
"""
import sys, os, json, uuid, time, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from app.core import omer_import

SHOWCASE = os.path.join(ROOT, '..', 'procmusic', 'dashboard', 'showcase-midi')
SRC_DIR = os.path.join(SHOWCASE, 'fur_elise_p1')
DST_DIR = os.path.join(SHOWCASE, 'fur_elise_mech')
MXML_DIR = os.path.join(HERE, 'fixtures', 'omr_out')


def find_mxml():
    for f in os.listdir(MXML_DIR):
        if f.lower().endswith(('.musicxml', '.xml')):
            return os.path.join(MXML_DIR, f)
    raise SystemExit('no MusicXML in ' + MXML_DIR)


def main():
    mxml = find_mxml()
    raw = omer_import.musicxml_to_bars(mxml)
    mq = omer_import.meter_quarters('3/8')
    mech = [{'melody': omer_import.fit_to_meter(b.get('melody', ''), mq),
             'bass':   omer_import.fit_to_meter(b.get('bass', ''), mq)} for b in raw]
    print(f'mechanical page-1 measures: {len(mech)}')

    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)

    sp = os.path.join(DST_DIR, '_job', 'state.json')
    st = json.load(open(sp, encoding='utf-8'))
    st['id'] = str(uuid.uuid4())
    st['piece_id'] = 'fur_elise_mech'
    st['title'] = 'Fur Elise (mechanical OMR)'
    st['created'] = time.time()
    st['approved'] = False
    st['source'] = 'pipeline'
    st['meta'] = {'timeSig': '3/8', 'key': 'A minor', 'bpm': 70, 'title': 'Fur Elise (mechanical OMR)'}
    st['bars'] = [{
        'n': i, 'page': 1, 'sys': 0,
        'melody': m['melody'], 'bass': m['bass'], 'melody2': '', 'bass2': '',
        'issues': [], 'pitch_issues': [], 'rhythm_issues': [],
        'confidence': 0.4, 'verified': False, 'edited': False, 'flags': [],
    } for i, m in enumerate(mech, start=1)]
    # page-1-only mechanical pass
    st['pages'] = [p for p in st.get('pages', []) if p.get('page') == 1]
    json.dump(st, open(sp, 'w', encoding='utf-8'), indent=2)
    print(f"created job id={st['id']} title={st['title']} bars={len(st['bars'])}")
    print(f"dir: {DST_DIR}")


if __name__ == '__main__':
    main()
