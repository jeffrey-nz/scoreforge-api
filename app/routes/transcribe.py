"""PDF upload → MIDI transcription via ai_transcribe.py."""
import asyncio
import re
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, Form
from pydantic import BaseModel

from app.config import CORE_DIR, MIDI_OUTPUT_DIR, PYTHON
from app.routes.jobs import create_job, push_log, finish_job

router = APIRouter()


class TranscribeRequest(BaseModel):
    piece_id: str
    title: str = ""
    composer: str = ""
    key: str = ""
    bpm: int = 120


@router.post("")
async def transcribe(
    file: UploadFile,
    piece_id: str = Form(...),
    title: str = Form(""),
    composer: str = Form(""),
    key: str = Form(""),
    bpm: int = Form(120),
):
    """
    Upload a sheet-music PDF; starts an async transcription job.
    Returns job_id — poll GET /jobs/{id} or stream GET /jobs/{id}/stream.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    out_dir = MIDI_OUTPUT_DIR / piece_id

    job_id = create_job()

    async def _run():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            pdf_path = tmp.name

        args = [
            PYTHON, "-u",
            str(CORE_DIR / "ai_transcribe.py"),
            pdf_path,
            str(out_dir),
        ]
        if title:
            args += ["--title", title]
        if composer:
            args += ["--composer", composer]
        if key:
            args += ["--key", key]
        args += ["--bpm", str(bpm)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for raw in proc.stdout:
            push_log(job_id, raw.decode(errors="replace").rstrip())
        await proc.wait()
        finish_job(job_id, proc.returncode == 0)

    asyncio.create_task(_run())
    return {"job_id": job_id, "piece_id": piece_id}
