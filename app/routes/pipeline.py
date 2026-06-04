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
from app.pipeline.steps import run_step, run_approve, run_feedback, run_read, run_recompile_page

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
    pages: Optional[str] = Form(None),
):
    """Upload a PDF and create a new import job. Returns job_id immediately.
    `pages` (e.g. "1-2") compiles only a subset now; the rest can be
    compiled later from the review screen."""
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
    job.pages_spec = (pages or '').strip() or None
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
                    if (msg['type'] == 'step' and
                            msg['data'].get('status') == 'approved'):
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

class StepRunRequest(BaseModel):
    pages: Optional[str] = None   # e.g. "1-2", "3,5" — read step only


@router.post("/jobs/{job_id}/steps/{step_name}/run")
async def run_job_step(job_id: str, step_name: str, req: Optional[StepRunRequest] = None):
    """Start or rerun a step. For 'read', an optional {pages} compiles only a
    subset (others left pending), and is resume-aware: already-compiled pages
    are reused, so you can compile more of the PDF over several runs."""
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

    pages = (req.pages if req else None)
    if step_name == 'read' and pages:
        async def _read_then_advance():
            await run_read(job, pages_spec=pages)
            if job.steps['read'].status == 'done':
                await run_step(job, 'pitch')
        asyncio.create_task(_read_then_advance())
    else:
        asyncio.create_task(run_step(job, step_name))
    return {"ok": True, "step": step_name, "pages": pages}


# ── Approve ────────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/approve")
async def approve_job(job_id: str):
    """Finalise the import: write MIDI files from current bars, mark as approved."""
    job = _require_job(job_id)
    if not job.bars:
        raise HTTPException(400, "No bars to approve — run the read step first")
    asyncio.create_task(run_approve(job))
    return {"ok": True}


# ── Feedback AI pass ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    feedback: str


@router.post("/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, req: FeedbackRequest):
    """Send human feedback to AI for a targeted correction pass.

    The AI receives the full bar listing plus the feedback text and returns
    specific bar rewrites. Corrections are applied in-place and a
    'bars_updated' SSE event is emitted so connected clients refresh.
    """
    job = _require_job(job_id)
    if not job.bars:
        raise HTTPException(400, "No bars yet — run the read step first")
    fb = (req.feedback or '').strip()
    if not fb:
        raise HTTPException(400, "feedback text is required")
    asyncio.create_task(run_feedback(job, fb))
    return {"ok": True, "feedback": fb}


# ── Pipeline log ───────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/log")
def get_pipeline_log(job_id: str):
    """Return the structured pipeline log for offline analysis."""
    job = _require_job(job_id)
    log_path = job.out_dir / '_job' / 'pipeline.log.json'
    if not log_path.exists():
        return job.pipeline_log
    try:
        import json as _json
        return _json.loads(log_path.read_text(encoding='utf-8'))
    except Exception:
        return job.pipeline_log


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


@router.delete("/jobs/{job_id}/bars/{bar_n}")
async def delete_bar(job_id: str, bar_n: int):
    """Delete a single bar; remaining bars renumber to stay contiguous."""
    job = _require_job(job_id)
    if not job.delete_bar(bar_n):
        raise HTTPException(404, f"Bar {bar_n} not found")
    job.save()
    await job.emit('bars_updated', {'bars': job.bars, 'pages': job.pages})
    return {"ok": True, "bars": len(job.bars)}


# ── Page-level segment operations ───────────────────────────────────────────────

@router.get("/jobs/{job_id}/pages")
def get_pages(job_id: str):
    """The page→bar map: which pages are compiled, pending, and their bar ranges."""
    return _require_job(job_id).pages


@router.delete("/jobs/{job_id}/pages/{page}")
async def delete_page(job_id: str, page: int):
    """Drop all bars from a page (it stays 'pending' so you can recompile it)."""
    job = _require_job(job_id)
    removed = job.delete_page(page)
    job.save()
    await job.emit('bars_updated', {'bars': job.bars, 'pages': job.pages})
    return {"ok": True, "removed": removed, "pages": job.pages}


@router.post("/jobs/{job_id}/pages/{page}/recompile")
async def recompile_page(job_id: str, page: int):
    """Re-transcribe a single page from scratch (clears its cache + bars),
    then re-validates downstream. Other pages are untouched."""
    job = _require_job(job_id)
    if job.steps['read'].status == 'running':
        raise HTTPException(409, "Read step is already running")
    asyncio.create_task(run_recompile_page(job, page))
    return {"ok": True, "page": page}


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


# ── Meta suggestion ────────────────────────────────────────────────────────────

class SuggestMetaRequest(BaseModel):
    filename: str


@router.post("/suggest-meta")
async def suggest_meta(req: SuggestMetaRequest):
    """Ask the AI bridge to infer title + composer from a filename.

    Returns {title, composer} or raises 503 if the bridge is unreachable.
    """
    import re as _re
    import sys as _sys
    from app.config import CORE_DIR

    core = str(CORE_DIR)
    if core not in _sys.path:
        _sys.path.insert(0, core)

    import ai_correct as ac

    loop = asyncio.get_event_loop()
    bridge_up = await loop.run_in_executor(None, ac._bridge_ping)
    if not bridge_up:
        raise HTTPException(503, "AI bridge not available — start browser-ai-bridge first")

    filename = req.filename.strip()
    prompt = f"""A user is importing a sheet music file named "{filename}".

Identify the most likely piece title and composer based on the filename.
Use canonical names (e.g. "Für Elise" not "fur_elise", "Ludwig van Beethoven" not "beethoven").
If the filename is ambiguous, make the most reasonable inference.

Respond with JSON only — no prose, no markdown:
{{"title": "...", "composer": "..."}}"""

    try:
        response = await loop.run_in_executor(
            None, lambda: ac._bridge_ask(prompt, provider='gemini')
        )
        data = json.loads(_re.search(r'\{[^{}]+\}', response, _re.DOTALL).group())
        return {
            'title':    str(data.get('title', '')).strip(),
            'composer': str(data.get('composer', '')).strip(),
        }
    except Exception as e:
        raise HTTPException(500, f"AI suggestion failed: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_job(job_id: str) -> Job:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job
