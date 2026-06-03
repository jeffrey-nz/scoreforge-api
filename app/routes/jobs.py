"""Job store and status/streaming endpoints for long-running transcription jobs."""
import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

# In-memory job store: {job_id: {"status": "running"|"done"|"error", "log": [str]}}
_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "log": []}
    return job_id


def push_log(job_id: str, line: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["log"].append(line)


def finish_job(job_id: str, success: bool) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "done" if success else "error"


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


@router.get("/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "log": job["log"]}


@router.get("/{job_id}/stream")
async def job_stream(job_id: str):
    """Server-Sent Events stream of live log output for a job."""
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_gen():
        sent = 0
        while True:
            job = _jobs.get(job_id, {})
            lines = job.get("log", [])
            while sent < len(lines):
                yield f"data: {lines[sent]}\n\n"
                sent += 1
            if job.get("status") in ("done", "error"):
                yield f"event: done\ndata: {job['status']}\n\n"
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
