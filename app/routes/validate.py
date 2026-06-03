"""Music-theory validation via theory_check.py."""
import json
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import CORE_DIR, MIDI_OUTPUT_DIR, PYTHON

router = APIRouter()


class ValidateRequest(BaseModel):
    piece_id: str
    key: str = ""
    time_sig: str = ""
    bars: int = 0


@router.post("")
def validate_piece(req: ValidateRequest):
    """Run mechanical music-theory validation for a single piece."""
    args = [PYTHON, "-u", str(CORE_DIR / "theory_check.py"), req.piece_id, "--json"]
    if req.key:
        args += ["--key", req.key]
    if req.time_sig:
        args += ["--time-sig", req.time_sig]
    if req.bars:
        args += ["--bars", str(req.bars)]

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(CORE_DIR),
    )
    if result.returncode != 0 and not result.stdout.strip():
        raise HTTPException(status_code=500, detail=result.stderr[:500])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout, "stderr": result.stderr}


@router.post("/all")
def validate_all():
    """Run music-theory validation across every piece in the MIDI output directory."""
    args = [PYTHON, "-u", str(CORE_DIR / "theory_check.py"), "--all"]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(CORE_DIR),
    )
    return {"returncode": result.returncode, "output": result.stdout, "errors": result.stderr}
