"""Render Fur Elise page 1 and run the mechanical OMR (oemer) once, caching the
MusicXML so the comparison test can iterate on omer_import without re-running the
slow ML engine. Local fixture only (derived from the gitignored source PDF)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
from app.core import omr_run

SRC = 'app/core/sources/fur_elise_woo59.pdf'
OUT = 'tests/fixtures'
os.makedirs(f'{OUT}/omr_out', exist_ok=True)

t0 = time.time()
doc = fitz.open(SRC)
# 150 DPI: 300 DPI (2700x3600) OOMs oemer's onnxruntime (~680MB buffer); a
# quarter of the pixels keeps the allocation in range.
zoom = 150 / 72.0
pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
img = f'{OUT}/fe_page1.png'
pix.save(img)
print(f'rendered page 1 -> {img} ({pix.w}x{pix.h}) in {time.time()-t0:.1f}s', flush=True)

t1 = time.time()
mxml = omr_run.run_oemer(img, f'{OUT}/omr_out')
print(f'OEMER DONE -> {mxml} in {time.time()-t1:.1f}s', flush=True)
