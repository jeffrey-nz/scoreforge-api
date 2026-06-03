# scoreforge-api

REST API for sheet music import, AI transcription, and music-theory validation.
Part of the [procmusic](https://github.com/jeffrey-nz/procmusic-dashboard) ecosystem.

## Overview

| Endpoint | Method | Description |
|---|---|---|
| `/transcribe` | POST | Upload a PDF; returns `job_id` |
| `/jobs/{id}` | GET | Job status + accumulated log |
| `/jobs/{id}/stream` | GET | Live SSE stream of job output |
| `/validate` | POST | Theory-check a single piece |
| `/validate/all` | POST | Theory-check all pieces |
| `/correct` | POST | AI correction pass (SSE stream) |

Interactive docs at **http://localhost:8001/docs** when running.

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit the environment file
copy .env.example .env

# 4. Start the server
uvicorn main:app --port 8001 --reload
```

## Configuration

All settings can be set via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `PROCMUSIC_PYTHON` | `python` | Python interpreter path |
| `MIDI_OUTPUT_DIR` | `../procmusic-dashboard/dashboard/showcase-midi` | Where transcribed MIDI lands |

## Related projects

- **[soundgen-api](https://github.com/jeffrey-nz/soundgen-api)** — MIDI composition and audio rendering
- **[procmusic-dashboard](https://github.com/jeffrey-nz/procmusic-dashboard)** — Web dashboard that ties everything together
