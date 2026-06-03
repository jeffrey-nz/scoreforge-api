"""
scoreforge-api — Sheet music import, transcription, and validation API.

Multi-step pipeline (recommended):
  POST /pipeline/jobs                        Upload PDF → job_id (auto-runs all steps)
  GET  /pipeline/jobs/{id}/stream            SSE live events
  POST /pipeline/jobs/{id}/steps/{s}/run     Rerun a specific step
  POST /pipeline/jobs/{id}/approve           Finalise MIDI output
  PATCH /pipeline/jobs/{id}/bars/{n}         Edit a bar

Legacy single-shot endpoints:
  POST /transcribe          Upload PDF, one-shot transcription
  POST /validate            Theory-check a piece
  POST /correct             AI correction pass (SSE stream)

Run:
  uvicorn main:app --port 8001 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import transcribe, validate, correct, jobs
from app.routes.pipeline import router as pipeline_router

app = FastAPI(
    title="ScoreForge API",
    description="Sheet music import, AI transcription, and music-theory validation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router,   prefix="/pipeline",   tags=["Pipeline"])
app.include_router(transcribe.router, prefix="/transcribe", tags=["Legacy"])
app.include_router(jobs.router,       prefix="/jobs",       tags=["Legacy"])
app.include_router(validate.router,   prefix="/validate",   tags=["Validation"])
app.include_router(correct.router,    prefix="/correct",    tags=["Correction"])


@app.get("/", tags=["Health"])
def root():
    return {"service": "scoreforge-api", "status": "ok"}
