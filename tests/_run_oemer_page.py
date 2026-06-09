"""Render one Fur Elise page and run the mechanical OMR (oemer), caching the
MusicXML. Usage: python tests/_run_oemer_page.py <page-number>  (1-indexed).
Local fixture only (derived from the gitignored source PDF)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
from app.core import omr_run

page = int(sys.argv[1])
dpi = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
SRC = 'app/core/sources/fur_elise_woo59.pdf'
OUT = 'tests/fixtures'
os.makedirs(f'{OUT}/omr_out', exist_ok=True)

t0 = time.time()
doc = fitz.open(SRC)
zoom = dpi / 72.0
pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
img = f'{OUT}/fe_page{page}.png'
pix.save(img)
print(f'rendered page {page} -> {img} ({pix.w}x{pix.h}) @ {dpi}dpi in {time.time()-t0:.1f}s', flush=True)

if len(sys.argv) > 3 and sys.argv[3] == 'render-only':
    raise SystemExit(0)

t1 = time.time()
mxml = omr_run.run_oemer(img, f'{OUT}/omr_out')
print(f'OEMER DONE page {page} -> {mxml} in {time.time()-t1:.1f}s', flush=True)
