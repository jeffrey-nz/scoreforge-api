"""AI correction pass via ai_correct.py — streams progress as SSE."""
import asyncio
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.config import CORE_DIR, PYTHON

router = APIRouter()


@router.post("")
async def correct_piece(
    piece_id: str,
    provider: str = Query(default="gemini", description="AI provider: gemini or chatgpt"),
    reference_pdf: Optional[str] = Query(default=None, description="Path to a reference PDF for comparison"),
    dry_run: bool = Query(default=False),
):
    """
    Run the AI correction loop on an imported piece.
    Returns a Server-Sent Events stream of progress log lines.
    """
    args = [PYTHON, "-u", str(CORE_DIR / "ai_correct.py"), piece_id,
            "--provider", provider]
    if reference_pdf:
        args += ["--reference", reference_pdf]
    if dry_run:
        args.append("--dry-run")

    async def event_gen():
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            yield f"data: {line}\n\n"
        await proc.wait()
        status = "done" if proc.returncode == 0 else "error"
        yield f"event: done\ndata: {status}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
