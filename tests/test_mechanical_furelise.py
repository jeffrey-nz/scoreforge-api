"""Unit tests: can the FULLY MECHANICAL pipeline reproduce the hand-verified
bars of Fur Elise? One test per verified bar (bar 1 passes, bar 3, ...).

Run:    python -m pytest tests/test_mechanical_furelise.py -v
Report: python tests/test_mechanical_furelise.py        # per-bar diff + pass rate

Requires the cached oemer MusicXML (tests/fixtures/omr_out/*.musicxml); generate
it once with:  python tests/_run_oemer_page1.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
import mechanical_compare as mc

_GOLD = mc.load_gold()
_PAGE1 = [b for b in _GOLD['bars'] if b.get('page') == 1]
_MECH = mc.mechanical_bars(_GOLD['meta'].get('timeSig', '3/8'))


@pytest.mark.skipif(_MECH is None, reason="no cached oemer MusicXML — run tests/_run_oemer_page1.py first")
@pytest.mark.parametrize('bar', _PAGE1, ids=[f"bar{b['n']}" for b in _PAGE1])
def test_bar_mechanical(bar):
    mech = mc.mech_for_bar(_MECH, bar['n'])
    assert mc.bar_matches(bar, mech), (
        f"bar {bar['n']} mismatch\n"
        f"  gold melody: {bar.get('melody')}\n  mech melody: {mech and mech.get('melody')}\n"
        f"  gold bass  : {bar.get('bass')}\n  mech bass  : {mech and mech.get('bass')}")


if __name__ == '__main__':
    mech = mc.mechanical_bars(_GOLD['meta'].get('timeSig', '3/8'))
    if mech is None:
        print('No cached MusicXML yet. Run: python tests/_run_oemer_page1.py')
        raise SystemExit(1)
    print(f'mechanical measures parsed: {len(mech)}\n')
    npass = npitch = 0
    for b in _PAGE1:
        m = mc.mech_for_bar(mech, b['n'])
        ok = mc.bar_matches(b, m)
        pok = mc.bar_pitch_matches(b, m)
        npass += ok; npitch += pok
        tag = 'PASS' if ok else ('pitch-ok' if pok else 'FAIL')
        print(f"bar {b['n']:>3}: {tag}")
        if not ok:
            print(f"      gold mel: {b.get('melody')}")
            print(f"      mech mel: {m and m.get('melody')}")
            print(f"      gold bas: {b.get('bass')}")
            print(f"      mech bas: {m and m.get('bass')}")
    n = len(_PAGE1)
    print(f"\n=== EXACT (pitch+rhythm): {npass}/{n}   PITCH-ONLY: {npitch}/{n} ===")
    print("    (rhythm is the gap — oemer's durations are unreliable; pitches read far better)")
