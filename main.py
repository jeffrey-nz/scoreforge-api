"""
scoreforge-api — Sheet music import, transcription, and validation API.

Endpoints:
  POST /transcribe          Upload a PDF; returns job_id for polling
  GET  /jobs/{id}           Job status + accumulated log
  GET  /jobs/{id}/stream    SSE stream of live log output
  POST /validate            Run theory-check on a piece
  POST /validate/all        Run theory-check on every piece
  POST /correct             AI correction pass on a piece (SSE stream)

Run:
  uvicorn main:app --port 8001 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import transcribe, validate, correct, jobs

app = FastAPI(
    title="ScoreForge API",
    description="Sheet music import, AI transcription, and music-theory validation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router, prefix="/transcribe", tags=["Transcription"])
app.include_router(jobs.router,       prefix="/jobs",       tags=["Jobs"])
app.include_router(validate.router,   prefix="/validate",   tags=["Validation"])
app.include_router(correct.router,    prefix="/correct",    tags=["Correction"])


@app.get("/", tags=["Health"])
def root():
    return {"service": "scoreforge-api", "status": "ok"}
