"""Glyph reference library + collection of unidentified music symbols.

A shared, AI-friendly catalogue of notation glyphs and how each maps to our
compact note format. The seed catalogue (app/core/glyphs.json) is read-only
reference; anything a human/AI clips from a score as "unidentified" is stored in
a writable data dir so it can be catalogued and later identified.

  GET  /glyphs                  -> {glyphs:[...seed + user], categories:[...]}
  GET  /glyphs/by-piece/{piece} -> glyphs that actually occur in a given piece
  GET  /glyphs/{id}             -> one glyph (with its samples + provenance)
  POST /glyphs                  -> add a glyph (typically status 'unidentified' + image)
  PATCH /glyphs/{id}            -> identify/annotate a user glyph
  GET  /glyphs/{id}/image       -> the first sample PNG (back-compat)
  GET  /glyphs/{id}/image/{f}   -> a specific named sample PNG

Each glyph carries a `samples` list — one entry per REAL crop collected from a
score, tagged with provenance (piece/page/system/bar/region) and, where a single
note carries the symbol, the actual compact token it produced (`note`/`maps_to`).
That list is the relationship between the reference library and the music: the
same glyph can hold multiple samples from different pieces. Provenance lives in
app/core/glyph_samples.json; the PNGs in app/core/glyph_images/<id>/<file>.
We never synthesise a glyph — a glyph with no samples is reported as missing.
"""
import base64
import json
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import CORE_DIR

router = APIRouter()

_SEED = CORE_DIR / "glyphs.json"
_SEED_IMG = CORE_DIR / "glyph_images"        # real PNG segments cropped from source scores
_SAMPLES = CORE_DIR / "glyph_samples.json"   # provenance: sample -> piece/page/bar/note
GLYPHS_DIR = Path(__file__).resolve().parent.parent / "data" / "glyphs"
_USER = GLYPHS_DIR / "user_glyphs.json"
_IMG = GLYPHS_DIR / "images"


def _load_manifest():
    """{ pieces: {...}, samples: { glyph_id: [ {file, piece, page, bar, ...} ] } }"""
    try:
        return json.loads(_SAMPLES.read_text(encoding="utf-8"))
    except Exception:
        return {"pieces": {}, "samples": {}}


def _samples_for(gid, manifest):
    """Real cropped samples for a glyph: provenance entries from the manifest
    that have a matching PNG on disk, plus any on-disk PNGs not yet in the
    manifest (so dropping a file in glyph_images/<id>/ is enough). We never
    fabricate an image — no files => empty list => reported as missing."""
    gdir = _SEED_IMG / gid
    by_file = {}
    for s in manifest.get("samples", {}).get(gid, []):
        fn = s.get("file")
        if fn and (gdir / fn).exists():
            by_file[fn] = {**s, "url": f"/glyphs/{gid}/image/{fn}"}
    if gdir.is_dir():
        for p in sorted(gdir.glob("*.png")):
            by_file.setdefault(p.name, {"file": p.name, "piece": p.stem,
                                        "url": f"/glyphs/{gid}/image/{p.name}"})
    return list(by_file.values())


def _load_seed():
    """Seed catalogue. Each glyph gets a `samples` list (real crops + provenance)
    and `image_url`/`image_missing` flags. We never synthesise a glyph image — a
    glyph with no collected samples is reported as missing."""
    try:
        glyphs = json.loads(_SEED.read_text(encoding="utf-8")).get("glyphs", [])
    except Exception:
        return []
    manifest = _load_manifest()
    for g in glyphs:
        samples = _samples_for(g.get("id"), manifest)
        g["samples"] = samples
        g["image_url"] = samples[0]["url"] if samples else None
        g["image_missing"] = not samples
    return glyphs


def _load_user():
    try:
        return json.loads(_USER.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_user(items):
    GLYPHS_DIR.mkdir(parents=True, exist_ok=True)
    _USER.write_text(json.dumps(items, indent=1), encoding="utf-8")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "glyph").lower()).strip("-") or "glyph"


def _decorate_user(user):
    """Give user-collected glyphs the same samples[] shape as seed glyphs."""
    for g in user:
        if g.get("image"):
            g["samples"] = [{"file": g["image"], "piece": g.get("source") or "collected",
                             "url": f"/glyphs/{g['id']}/image", "source": g.get("source")}]
            g["image_url"] = f"/glyphs/{g['id']}/image"
            g["image_missing"] = False
        else:
            g["samples"] = []
            g["image_url"] = None
            g["image_missing"] = True
    return user


@router.get("")
@router.get("/")
def list_glyphs():
    glyphs = _load_seed() + _decorate_user(_load_user())
    cats = sorted({g.get("category", "other") for g in glyphs})
    return {"glyphs": glyphs, "categories": cats, "count": len(glyphs),
            "samples_total": sum(len(g.get("samples", [])) for g in glyphs),
            "with_samples": sum(1 for g in glyphs if g.get("samples")),
            "missing_samples": sum(1 for g in glyphs if not g.get("samples")),
            "pieces": _load_manifest().get("pieces", {}),
            "unidentified": sum(1 for g in glyphs if g.get("status") == "unidentified")}


@router.get("/by-piece/{piece}")
def glyphs_by_piece(piece: str):
    """Reverse lookup: which reference glyphs actually occur in this piece, with
    the specific samples (and note tokens) drawn from it. Futureproofs linking
    transcribed music back to the glyph catalogue."""
    out = []
    for g in _load_seed():
        hits = [s for s in g.get("samples", []) if s.get("piece") == piece]
        if hits:
            out.append({"id": g["id"], "name": g.get("name"), "category": g.get("category"),
                        "compact": g.get("compact"), "samples": hits})
    return {"piece": piece, "meta": _load_manifest().get("pieces", {}).get(piece),
            "count": len(out), "glyphs": out}


@router.get("/{glyph_id}")
def get_glyph(glyph_id: str):
    for g in _load_seed() + _decorate_user(_load_user()):
        if g.get("id") == glyph_id:
            return g
    raise HTTPException(404, "glyph not found")


class NewGlyph(BaseModel):
    name: Optional[str] = None
    category: str = "unidentified"
    meaning: Optional[str] = None
    compact: Optional[str] = None
    smufl: Optional[str] = None
    status: str = "unidentified"
    source: Optional[str] = None        # e.g. "fur_elise_p1 bar 12"
    image_b64: Optional[str] = None      # PNG, optionally a data: URL


@router.post("")
@router.post("/")
def add_glyph(req: NewGlyph):
    """Catalogue a new glyph — usually a clipped, not-yet-identified symbol."""
    user = _load_user()
    base = _slug(req.name or req.category)
    gid = base
    existing = {g["id"] for g in user} | {g["id"] for g in _load_seed()}
    n = 2
    while gid in existing:
        gid = f"{base}-{n}"; n += 1
    entry = {"id": gid, "name": req.name or "(unidentified)", "category": req.category,
             "meaning": req.meaning, "compact": req.compact, "smufl": req.smufl,
             "status": req.status, "source": req.source, "added": time.time(),
             "image": None}
    if req.image_b64:
        _IMG.mkdir(parents=True, exist_ok=True)
        data = req.image_b64.split(",", 1)[-1]   # strip data: URL prefix if present
        try:
            (_IMG / f"{gid}.png").write_bytes(base64.b64decode(data))
            entry["image"] = f"{gid}.png"
        except Exception:
            pass
    user.append(entry)
    _save_user(user)
    return {"ok": True, "id": gid, "glyph": entry}


class GlyphPatch(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    meaning: Optional[str] = None
    compact: Optional[str] = None
    smufl: Optional[str] = None
    status: Optional[str] = None


@router.patch("/{glyph_id}")
def identify_glyph(glyph_id: str, patch: GlyphPatch):
    """Fill in / correct a user glyph (e.g. mark an unidentified one as known)."""
    user = _load_user()
    for g in user:
        if g.get("id") == glyph_id:
            for k, v in patch.dict(exclude_none=True).items():
                g[k] = v
            _save_user(user)
            return {"ok": True, "glyph": g}
    raise HTTPException(404, "glyph not found (seed glyphs are read-only)")


def _safe(name: str) -> str:
    # never let a sample filename escape its glyph dir
    return Path(name).name


@router.get("/{glyph_id}/image/{filename}")
def glyph_image_named(glyph_id: str, filename: str):
    """A specific named sample (one piece's crop of this glyph)."""
    p = _SEED_IMG / _safe(glyph_id) / _safe(filename)
    if p.exists():
        return FileResponse(str(p), media_type="image/png")
    raise HTTPException(404, "no such sample")


@router.get("/{glyph_id}/image")
def glyph_image(glyph_id: str):
    """First available sample (back-compat single-image accessor)."""
    gdir = _SEED_IMG / _safe(glyph_id)
    if gdir.is_dir():
        pngs = sorted(gdir.glob("*.png"))
        if pngs:
            return FileResponse(str(pngs[0]), media_type="image/png")
    # user-collected glyphs
    for g in _load_user():
        if g.get("id") == glyph_id and g.get("image"):
            up = _IMG / g["image"]
            if up.exists():
                return FileResponse(str(up), media_type="image/png")
    raise HTTPException(404, "no image for this glyph")
