"""
Job state machine for the multi-step import pipeline.

Each job tracks six discrete steps:
  detect  – mechanical: render pages, count systems/bars
  read    – AI transcription via pdf_to_midi subprocess
  pitch   – mechanical pitch check + AI bar refinement
  rhythm  – mechanical rhythm check + AI bar refinement
  theory  – rule-based music-theory validation
  review  – human sign-off (approve or edit)

State is kept in memory and snapshotted to _job/state.json for resumability.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STEP_ORDER = ['detect', 'read', 'pitch', 'rhythm', 'theory']  # 'review' is not auto-run


@dataclass
class StepState:
    status: str = 'idle'   # idle | running | done | error
    pct: int = 0
    result: Optional[Dict] = None
    issues: List[Dict] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'status': self.status,
            'pct': self.pct,
            'result': self.result,
            'issues': self.issues,
            'log': self.log[-200:],
        }


class Job:
    def __init__(self, *, piece_id: str, pdf_path: str, out_dir: str,
                 title: str = '', composer: str = '',
                 bpm: Optional[int] = None, provider: str = 'gemini'):
        self.id: str = str(uuid.uuid4())
        self.piece_id = piece_id
        self.pdf_path = str(pdf_path)
        self.out_dir = Path(out_dir)
        self.title = title
        self.composer = composer
        self.bpm = bpm
        self.provider = provider
        self.created = time.time()
        self.approved = False

        self.steps: Dict[str, StepState] = {
            s: StepState() for s in STEP_ORDER + ['review']
        }
        self.bars: List[Dict] = []   # [{n, melody, bass, issues:[]}]
        self.meta: Dict = {}         # {key, timeSig, bpm, title, composer}

        self._queues: List[asyncio.Queue] = []
        self._task: Optional[asyncio.Task] = None   # running step task

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'piece_id': self.piece_id,
            'title': self.title,
            'composer': self.composer,
            'bpm': self.bpm,
            'provider': self.provider,
            'created': self.created,
            'approved': self.approved,
            'steps': {k: v.to_dict() for k, v in self.steps.items()},
            'bars': self.bars,
            'meta': self.meta,
        }

    def save(self):
        job_dir = self.out_dir / '_job'
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'state.json').write_text(
            json.dumps(self.to_dict(), indent=2), encoding='utf-8'
        )

    # ── Event pub/sub ──────────────────────────────────────────────────────────

    async def emit(self, event_type: str, data: Any = None):
        msg = {'type': event_type, 'data': data or {}, 'ts': time.time()}
        for q in list(self._queues):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    # ── Bar helpers ────────────────────────────────────────────────────────────

    def bars_as_table(self) -> List[Dict]:
        """Return bars with per-bar issue aggregation for the review table."""
        return self.bars

    def get_bar(self, n: int) -> Optional[Dict]:
        idx = n - 1
        return self.bars[idx] if 0 <= idx < len(self.bars) else None

    def set_bar(self, n: int, melody: Optional[str] = None, bass: Optional[str] = None):
        bar = self.get_bar(n)
        if bar is None:
            return
        if melody is not None:
            bar['melody'] = melody
        if bass is not None:
            bar['bass'] = bass
        bar['edited'] = True


# ── Global store ──────────────────────────────────────────────────────────────

_JOBS: Dict[str, Job] = {}


def create_job(**kwargs) -> Job:
    job = Job(**kwargs)
    _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)


def list_jobs() -> List[Job]:
    return sorted(_JOBS.values(), key=lambda j: j.created, reverse=True)
