"""control.py — operator/control session for the dashboard.

Lets Claude Code (driving from the user's editor) take control of the app and
have the browser watch in real time: Claude claims control, narrates what it's
doing, and every step streams to all open dashboards over SSE. The actual work
(creating jobs, writing bars, fixing meters) goes through the normal pipeline
API, so the review updates live too.
"""
import time
import asyncio
from collections import deque


class ControlHub:
    def __init__(self):
        self.controller = None          # {'who','note','since'} while held, else None
        self._activity = deque(maxlen=300)
        self._subs = set()              # set[asyncio.Queue]

    # ── snapshot ──────────────────────────────────────────────────────────────
    def state(self):
        return {'controller': self.controller,
                'activity': list(self._activity)[-40:]}

    # ── pub/sub ───────────────────────────────────────────────────────────────
    def subscribe(self):
        q = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q):
        self._subs.discard(q)

    async def _broadcast(self, event, data):
        for q in list(self._subs):
            try:
                q.put_nowait((event, data))
            except Exception:
                pass

    # ── operations ────────────────────────────────────────────────────────────
    async def claim(self, who='Claude Code', note=''):
        self.controller = {'who': who, 'note': note, 'since': time.time()}
        await self._broadcast('control', {'controller': self.controller})
        await self.activity(f'{who} took control'
                            + (f' — {note}' if note else ''), kind='control')
        return self.controller

    async def release(self, who=None):
        held = (self.controller or {}).get('who', who or 'Claude Code')
        self.controller = None
        await self._broadcast('control', {'controller': None})
        await self.activity(f'{held} released control', kind='control')

    async def activity(self, message, job=None, kind='info'):
        item = {'t': time.time(), 'message': str(message), 'job': job, 'kind': kind}
        self._activity.append(item)
        await self._broadcast('activity', item)
        return item


hub = ControlHub()
