"""Glyph reference library + collection of unidentified music symbols.

A shared, AI-friendly catalogue of notation glyphs and how each maps to our
compact note format. The seed catalogue (app/core/glyphs.json) is read-only
reference; anything a human/AI clips from a score as "unidentified" is stored in
a writable data dir so it can be catalogued and later identified.

  GET  /glyphs              -> {glyphs:[...seed + user], categories:[...]}
  GET  /glyphs/{id}         -> one glyph
  POST /glyphs              -> add a glyph (typically status 'unidentified' + image)
  PATCH /glyphs/{id}        -> identify/annotate a user glyph
  GET  /glyphs/{id}/image   -> the clipped PNG for a user glyph
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
GLYPHS_DIR = Path(__file__).resolve().parent.parent / "data" / "glyphs"
_USER = GLYPHS_DIR / "user_glyphs.json"
_IMG = GLYPHS_DIR / "images"


def _load_seed():
    try:
        return json.loads(_SEED.read_text(encoding="utf-8")).get("glyphs", [])
    except Exception:
        return []


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


@router.get("")
@router.get("/")
def list_glyphs():
    seed = _load_seed()
    user = _load_user()
    for g in user:
        g["image_url"] = f"/glyphs/{g['id']}/image"
    glyphs = seed + user
    cats = sorted({g.get("category", "other") for g in glyphs})
    return {"glyphs": glyphs, "categories": cats,
            "count": len(glyphs), "unidentified": sum(1 for g in glyphs if g.get("status") == "unidentified")}


@router.get("/{glyph_id}")
def get_glyph(glyph_id: str):
    for g in _load_seed() + _load_user():
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


@router.get("/{glyph_id}/image")
def glyph_image(glyph_id: str):
    for g in _load_user():
        if g.get("id") == glyph_id and g.get("image"):
            p = _IMG / g["image"]
            if p.exists():
                return FileResponse(str(p), media_type="image/png")
    raise HTTPException(404, "no image for this glyph")
