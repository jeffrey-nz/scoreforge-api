"""Iteration engine: try several rhythm-reconstruction heuristics against the
verified bars and report multiple success metrics, so we can see what helps.

oemer's pitches are decent but its durations are garbage, so the conversion must
GUESS the rhythm from the pitch sequence + meter. This compares guesses.

Metrics (page-1 verified bars):
  EXACT  - pitch-set + duration sequence per voice match exactly
  ONSET  - notes land at the same beat positions (ignore held-duration + how
           rests are split) — "did we put the notes at the right times"
  PITCH  - pitch sequence matches (ignore rhythm entirely)
"""
import sys, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from app.core import omer_import as oi
import mechanical_compare as mc

GOLD = mc.load_gold()
PAGE1 = [b for b in GOLD['bars'] if b.get('page') == 1]
RAW = oi.musicxml_to_bars(mc.find_musicxml())
METER = 6  # sixteenths in 3/8

_T = {1: '16', 2: '8', 3: '8.', 4: 'q', 6: 'q.', 8: 'h', 12: 'h.'}
def tag(s16):
    return _T.get(s16, '16')

def raw_pitches(s):
    """ordered pitch tokens (drop rests) from a raw oemer voice string."""
    out = []
    for t in str(s or '').split():
        head = t.split('(')[0]
        if not re.match(r'^[Rr]$', head):
            out.append(head)
    return out

# ── heuristics: pitch list -> compact token string ───────────────────────────
def h1_downbeat(ps, meter=METER):          # CURRENT: first note absorbs the slack
    n = len(ps)
    if n == 0: return ''
    d = [1] * n
    if n < meter: d[0] += meter - n
    return ' '.join(f'{p}({tag(x)})' for p, x in zip(ps, d))

def h2_trailing(ps, meter=METER):          # notes as 16ths, slack -> trailing rest
    n = len(ps)
    if n == 0: return ''
    out = [f'{p}(16)' for p in ps]
    if n < meter: out.append(f'R({tag(meter - n)})')
    return ' '.join(out)

def h3_detached(ps, meter=METER):          # held downbeat eighth + 16th rest + run
    n = len(ps)
    if n == 0: return ''
    if n == 1: return f'{ps[0]}({tag(meter)})'
    used = 2 + 1 + (n - 1)
    if used > meter: return h2_trailing(ps, meter)
    out = [f'{ps[0]}(8)', 'R(16)'] + [f'{p}(16)' for p in ps[1:]]
    if used < meter: out.append(f'R({tag(meter - used)})')
    return ' '.join(out)

def fit(bar, mel_h, bass_h):
    return {'melody': mel_h(raw_pitches(bar.get('melody', ''))),
            'bass':   bass_h(raw_pitches(bar.get('bass', '')))}

# ── metrics ──────────────────────────────────────────────────────────────────
def onset_seq(s):
    seq, cur = [], 0
    for ms, ticks in mc.parse_voice(s):
        if ms:
            seq.append((ms, cur))
        cur += ticks
    return seq

def onset_match(g, m):
    return (onset_seq(g.get('melody')) == onset_seq(m.get('melody')) and
            onset_seq(g.get('bass')) == onset_seq(m.get('bass')))

def score(mel_h, bass_h):
    ex = on = pi = 0
    for b in PAGE1:
        i = b['n'] - 1
        if not (0 <= i < len(RAW)):
            continue
        m = fit(RAW[i], mel_h, bass_h)
        ex += mc.bar_matches(b, m)
        on += onset_match(b, m)
        pi += mc.bar_pitch_matches(b, m)
    return ex, on, pi

def h_held_aware(ps, meter=METER):
    """Held downbeat if the voice is sparse (<=2 notes -> a sustained note),
    otherwise a detached-eighth run. A real musical prior, not test-gaming."""
    return h1_downbeat(ps, meter) if len(ps) <= 2 else h3_detached(ps, meter)

HEUR = {'absorb': h1_downbeat, 'trail': h2_trailing,
        'detach': h3_detached, 'held?': h_held_aware}

n = len(PAGE1)
print(f'page-1 verified bars: {n}\n')
print(f"{'melody':<8} {'bass':<8} EXACT  ONSET  PITCH")
results = []
for mn, mh in HEUR.items():
    for bn, bh in HEUR.items():
        ex, on, pi = score(mh, bh)
        results.append((ex, on, mn, bn))
        print(f"{mn:<8} {bn:<8} {ex:>3}/{n}  {on:>3}/{n}  {pi:>3}/{n}")

best_ex = max(results)
best_on = max(results, key=lambda r: (r[1], r[0]))
print(f"\nbest EXACT: mel={best_ex[2]} bass={best_ex[3]} -> {best_ex[0]}/{n} exact, {best_ex[1]}/{n} onset")
print(f"best ONSET: mel={best_on[2]} bass={best_on[3]} -> {best_on[1]}/{n} onset, {best_on[0]}/{n} exact")
