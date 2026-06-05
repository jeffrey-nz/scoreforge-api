"""Source score documents (the music PDFs) and their relationships.

A source PDF sits at the top of the provenance chain:

    source PDF  ->  page render  ->  bar  ->  note token / glyph crop

The PDF binaries live in app/core/sources/ (gitignored); this registry
(app/core/sources.json) and the page renders are version-controlled. Each
source links to the piece(s) transcribed from it, and — via the glyph-sample
provenance manifest (app/core/glyph_samples.json) — to the glyph crops and the
actual compact note tokens taken from its pages.

  GET /sources              -> list sources (+ whether the PDF is embedded locally)
  GET /sources/{id}         -> one source with its related glyphs + pieces + notes
  GET /sources/{id}/pdf     -> the embedded PDF (if present locally)
"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import CORE_DIR

router = APIRouter()

_SOURCES = CORE_DIR / "sources.json"
_SOURCES_DIR = CORE_DIR / "sources"
_SAMPLES = CORE_DIR / "glyph_samples.json"


def _load_sources():
    try:
        return json.loads(_SOURCES.read_text(encoding="utf-8")).get("sources", [])
    except Exception:
        return []


def _load_samples():
    try:
        return json.loads(_SAMPLES.read_text(encoding="utf-8"))
    except Exception:
        return {"pieces": {}, "samples": {}}


def _pdf_present(src) -> bool:
    f = src.get("file")
    return bool(f and (_SOURCES_DIR / f).exists())


def _glyphs_for_pieces(pieces, manifest):
    """Every glyph crop whose provenance piece belongs to this source, with the
    note token it produced — the file<->glyph<->note relationship."""
    want = set(pieces or [])
    out = []
    for gid, samples in manifest.get("samples", {}).items():
        hits = [s for s in samples if s.get("piece") in want]
        if hits:
            out.append({"glyph": gid, "samples": hits})
    return out


@router.get("")
@router.get("/")
def list_sources():
    srcs = _load_sources()
    manifest = _load_samples()
    items = []
    for s in srcs:
        glyphs = _glyphs_for_pieces(s.get("pieces"), manifest)
        items.append({
            **s,
            "pdf_present": _pdf_present(s),
            "pdf_url": f"/sources/{s['id']}/pdf" if _pdf_present(s) else None,
            "glyph_count": len(glyphs),
            "sample_count": sum(len(g["samples"]) for g in glyphs),
        })
    return {"sources": items, "count": len(items)}


@router.get("/{source_id}")
def get_source(source_id: str):
    src = next((s for s in _load_sources() if s.get("id") == source_id), None)
    if not src:
        raise HTTPException(404, "source not found")
    manifest = _load_samples()
    glyphs = _glyphs_for_pieces(src.get("pieces"), manifest)
    # Distinct note tokens this source contributed (file -> note relationship).
    notes = sorted({s["note"] for g in glyphs for s in g["samples"] if s.get("note")})
    return {
        **src,
        "pdf_present": _pdf_present(src),
        "pdf_url": f"/sources/{src['id']}/pdf" if _pdf_present(src) else None,
        "pieces_meta": {p: manifest.get("pieces", {}).get(p) for p in src.get("pieces", [])},
        "glyphs": glyphs,
        "glyph_count": len(glyphs),
        "notes_sampled": notes,
    }


@router.get("/{source_id}/pdf")
def get_source_pdf(source_id: str):
    src = next((s for s in _load_sources() if s.get("id") == source_id), None)
    if not src:
        raise HTTPException(404, "source not found")
    p = _SOURCES_DIR / (src.get("file") or "")
    if not p.exists():
        raise HTTPException(404, "PDF not embedded locally (gitignored — copy it into app/core/sources/)")
    return FileResponse(str(p), media_type="application/pdf", filename=src["file"])
