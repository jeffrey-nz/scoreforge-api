"""Run oemer on K545 pages 2-8 sequentially (RAM-safe), caching MusicXML."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core import omr_run
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'k545_work')
out = os.path.join(W, 'omr')
for page in range(2, 9):
    img = os.path.join(W, f'k545_p{page}.png')
    mx = os.path.join(out, f'k545_p{page}.musicxml')
    if os.path.exists(mx):
        print(f'page {page}: cached', flush=True); continue
    t = time.time()
    try:
        omr_run.run_oemer(img, out)
        print(f'page {page}: done in {time.time()-t:.0f}s', flush=True)
    except Exception as e:
        print(f'page {page}: FAILED {type(e).__name__}: {e}', flush=True)
print('ALL DONE', flush=True)
