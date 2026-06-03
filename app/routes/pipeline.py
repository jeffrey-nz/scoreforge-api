"""
Multi-step import pipeline API.

POST  /pipeline/jobs                    upload PDF, create job → {job_id}
GET   /pipeline/jobs                    list recent jobs
GET   /pipeline/jobs/{id}               full job state
GET   /pipeline/jobs/{id}/stream        SSE: real-time step + log events
POST  /pipeline/jobs/{id}/steps/{step}/run   start/rerun a step
POST  /pipeline/jobs/{id}/approve       finalise MIDI files
GET   /pipeline/jobs/{id}/bars          get bars array
PATCH /pipeline/jobs/{id}/bars/{n}      edit a single bar
GET   /pipeline/jobs/{id}/page/{n}      stream page PNG
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from app.config import MIDI_OUTPUT_DIR
from app.pipeline.job import create_job, get_job, list_jobs, Job, STEP_ORDER
from app.pipeline.steps import run_step, run_approve

router = APIRouter()


# ── Job creation ───────────────────────────────────────────────────────────────

@router.post("/jobs", status_code=202)
async def create_import_job(
    file: UploadFile,
    piece_id: str = Form(...),
    title: str = Form(""),
    composer: str = Form(""),
    bpm: Optional[int] = Form(None),
    provider: str = Form("gemini"),
):
    """Upload a PDF and create a new import job. Returns job_id immediately."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files accepted")

    pdf_bytes = await file.read()
    out_dir = MIDI_OUTPUT_DIR / piece_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / file.filename
    pdf_path.write_bytes(pdf_bytes)

    job = create_job(
        piece_id=piece_id,
        pdf_path=str(pdf_path),
        out_dir=str(out_dir),
        title=title,
        composer=composer,
        bpm=bpm,
        provider=provider,
    )
    job.save()

    # Kick off the pipeline starting at 'detect' (auto-advances through all steps)
    asyncio.create_task(run_step(job, 'detect'))

    return {"job_id": job.id, "piece_id": piece_id}


# ── Job listing / status ───────────────────────────────────────────────────────

@router.get("/jobs")
def get_jobs():
    return [
        {"id": j.id, "piece_id": j.piece_id, "title": j.title,
         "created": j.created, "approved": j.approved,
         "steps": {k: {"status": v.status, "pct": v.pct}
                   for k, v in j.steps.items()}}
        for j in list_jobs()[:20]
    ]


@router.get("/jobs/{job_id}")
def get_job_state(job_id: str):
    job = _require_job(job_id)
    return job.to_dict()


# ── SSE stream ─────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """Server-Sent Events stream of all job events (step transitions + log lines)."""
    job = _require_job(job_id)

    async def event_gen():
        # Replay recent log lines so a reconnecting client catches up
        for step_name, step in job.steps.items():
            if step.status != 'idle':
                snapshot = {
                    'step': step_name, 'status': step.status,
                    'pct': step.pct, 'result': step.result,
                    'issues': step.issues,
                }
                yield _sse('step', snapshot)
                for line in step.log[-30:]:
                    yield _sse('log', {'step': step_name, 'line': line})

        # Stream live events
        q = job.subscribe()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield _sse(msg['type'], msg['data'])
                    if msg['type'] == 'step' and msg['data'].get('status') == 'approved':
                        break
                except asyncio.TimeoutError:
                    yield _sse('ping', {})
        finally:
            job.unsubscribe(q)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event_type: str, data) -> str:
    payload = json.dumps(data)
    if event_type == 'message' or event_type == 'log':
        return f"data: {payload}\n\n"
    return f"event: {event_type}\ndata: {payload}\n\n"


# ── Step control ───────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/steps/{step_name}/run")
async def run_job_step(job_id: str, step_name: str):
    """Start or rerun a specific pipeline step."""
    job = _require_job(job_id)
    if step_name not in STEP_ORDER:
        raise HTTPException(400, f"Unknown step '{step_name}'. Valid: {STEP_ORDER}")

    current = job.steps[step_name]
    if current.status == 'running':
        raise HTTPException(409, f"Step '{step_name}' is already running")

    # Reset this step and all downstream steps
    idx = STEP_ORDER.index(step_name)
    for s in STEP_ORDER[idx:]:
        job.steps[s].status = 'idle'
        job.steps[s].pct = 0
        job.steps[s].result = None
        job.steps[s].issues = []

    asyncio.create_task(run_step(job, step_name))
    return {"ok": True, "step": step_name}


# ── Approve ────────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/approve")
async def approve_job(job_id: str):
    """Finalise the import: write MIDI files from current bars, mark as approved."""
    job = _require_job(job_id)
    if not job.bars:
        raise HTTPException(400, "No bars to approve — run the read step first")
    asyncio.create_task(run_approve(job))
    return {"ok": True}


# ── Bar access / editing ───────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/bars")
def get_bars(job_id: str):
    return _require_job(job_id).bars


class BarPatch(BaseModel):
    melody: Optional[str] = None
    bass: Optional[str] = None


@router.patch("/jobs/{job_id}/bars/{bar_n}")
def patch_bar(job_id: str, bar_n: int, patch: BarPatch):
    """Edit a single bar's note strings."""
    job = _require_job(job_id)
    bar = job.get_bar(bar_n)
    if bar is None:
        raise HTTPException(404, f"Bar {bar_n} not found")
    job.set_bar(bar_n, melody=patch.melody, bass=patch.bass)
    job.save()
    return job.get_bar(bar_n)


# ── Page image ─────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/page/{page_n}")
def get_page_image(job_id: str, page_n: int):
    """Return the rendered PNG for a page."""
    job = _require_job(job_id)
    pages_dir = job.out_dir / '_pages'
    img = pages_dir / f'page_{page_n:02d}.png'
    if not img.exists():
        raise HTTPException(404, f"Page {page_n} not found")
    return FileResponse(str(img), media_type='image/png')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_job(job_id: str) -> Job:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job
