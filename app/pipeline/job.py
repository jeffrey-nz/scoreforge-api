"""
Job state machine for the multi-step import pipeline.

Each job tracks six discrete steps:
  detect  – mechanical: render pages, count systems/bars
  read    – AI transcription via pdf_to_midi subprocess
  pitch   – mechanical pitch check + AI bar refinement
  rhythm  – mechanical rhythm check + AI bar refinement
  theory  – rule-based music-theory validation
  review  – human sign-off (approve, edit, or feedback → AI re-pass)

State is kept in memory and snapshotted to _job/state.json for resumability.
A structured pipeline.log.json is also written on every step boundary and
feedback round so the full run can be analysed after the fact.
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
        self.pages_spec: Optional[str] = None  # initial page range for the auto-run read

        self.steps: Dict[str, StepState] = {
            s: StepState() for s in STEP_ORDER + ['review']
        }
        self.bars: List[Dict] = []   # [{n, page, melody, bass, issues:[]}]
        self.pages: List[Dict] = []  # [{page, status, startBar, endBar, bars}]
        self.meta: Dict = {}         # {key, timeSig, bpm, title, composer}

        self._queues: List[asyncio.Queue] = []
        self._task: Optional[asyncio.Task] = None

        # ── Pipeline log (written to pipeline.log.json on every boundary) ──────
        self._step_start_times: Dict[str, float] = {}
        self.pipeline_log: Dict = {
            'job_id': self.id,
            'piece_id': piece_id,
            'title': title,
            'composer': composer,
            'provider': provider,
            'started_at': time.time(),
            'approved_at': None,
            'steps': {},
            'feedback_rounds': [],
        }

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
            'pages': self.pages,
            'meta': self.meta,
        }

    def save(self):
        job_dir = self.out_dir / '_job'
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'state.json').write_text(
            json.dumps(self.to_dict(), indent=2), encoding='utf-8'
        )
        self._flush_pipeline_log()

    # ── Pipeline logging ───────────────────────────────────────────────────────

    def log_step_start(self, step_name: str):
        self._step_start_times[step_name] = time.time()
        self.pipeline_log['steps'].setdefault(step_name, {})['started_at'] = time.time()
        self.pipeline_log['steps'][step_name]['status'] = 'running'
        self._flush_pipeline_log()

    def log_step_end(self, step_name: str):
        now = time.time()
        start = self._step_start_times.get(step_name, now)
        entry = self.pipeline_log['steps'].setdefault(step_name, {})
        entry['completed_at'] = now
        entry['duration_s'] = round(now - start, 1)
        entry['status'] = self.steps[step_name].status
        entry['result'] = self.steps[step_name].result
        entry['issue_count'] = len(self.steps[step_name].issues)
        entry['log_lines'] = len(self.steps[step_name].log)
        self._flush_pipeline_log()

    def log_feedback(self, feedback: str, corrections: List[Dict], applied: int):
        self.pipeline_log['feedback_rounds'].append({
            'at': time.time(),
            'feedback': feedback,
            'corrections_requested': len(corrections),
            'corrections_applied': applied,
        })
        self._flush_pipeline_log()

    def log_approved(self):
        self.pipeline_log['approved_at'] = time.time()
        self._flush_pipeline_log()

    def _flush_pipeline_log(self):
        """Write pipeline.log.json — full log including step stdout."""
        try:
            job_dir = self.out_dir / '_job'
            job_dir.mkdir(parents=True, exist_ok=True)
            log = dict(self.pipeline_log)
            log['job_id'] = self.id
            # Embed full log lines from each step for offline analysis
            for sname, step in self.steps.items():
                if sname in log['steps']:
                    log['steps'][sname]['log'] = step.log
            (job_dir / 'pipeline.log.json').write_text(
                json.dumps(log, indent=2), encoding='utf-8'
            )
        except Exception:
            pass

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

    def renumber_bars(self):
        """Re-assign contiguous 1-based 'n' after inserts/deletes, then refresh
        the page→bar map so per-page ranges stay accurate."""
        for i, bar in enumerate(self.bars, 1):
            bar['n'] = i
        self._rebuild_page_ranges()

    def delete_bar(self, n: int) -> bool:
        idx = n - 1
        if not (0 <= idx < len(self.bars)):
            return False
        self.bars.pop(idx)
        self.renumber_bars()
        return True

    def delete_page(self, page: int) -> int:
        """Drop every bar belonging to a page. Returns the count removed.
        The page stays in the model marked 'pending' so it can be recompiled."""
        before = len(self.bars)
        self.bars = [b for b in self.bars if b.get('page') != page]
        removed = before - len(self.bars)
        for p in self.pages:
            if p.get('page') == page:
                p['status'] = 'pending'
                p['bars'] = 0
        self.renumber_bars()
        return removed

    def _rebuild_page_ranges(self):
        """Recompute each page's startBar/endBar/bars from the current bars."""
        counts: Dict[int, List[int]] = {}
        for bar in self.bars:
            counts.setdefault(bar.get('page', 0), []).append(bar['n'])
        for p in self.pages:
            ns = counts.get(p.get('page'), [])
            if ns:
                p['startBar'], p['endBar'], p['bars'] = min(ns), max(ns), len(ns)
                if p.get('status') == 'pending':
                    p['status'] = 'done'
            else:
                p['startBar'] = p['endBar'] = 0
                p['bars'] = 0


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
