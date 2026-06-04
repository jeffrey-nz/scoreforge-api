"""ai_engine.py — pluggable transcription engine.

Music transcription (image -> notes JSON) can run through either:

  * "bridge" — the browser-ai-bridge driving a real browser LLM
               (Gemini / ChatGPT / DeepSeek / Copilot / Grok). Default.
  * "claude" — Claude Code in the user's editor session, via a FILE QUEUE.
               The pipeline drops a request (the prompt + the image path) into
               a watch folder and waits; Claude Code reads the image and writes
               the answer back as a file; the pipeline picks it up and carries
               on. No API key or browser needed — the agent already in the
               session does the reading.

Every backend exposes the same primitives — image_ask / text_ask / audio_ask —
plus available()/ping() so callers can pre-flight and tell the user what's
going on. The transcription pipeline talks ONLY to this module.
"""
import os
import sys
import json
import time
import uuid
from pathlib import Path

ENGINE_DEFAULT = (os.environ.get('TRANSCRIBE_ENGINE', 'bridge') or 'bridge').lower()

# How long to wait for Claude to answer one request before giving up, and how
# often to re-tell the user we're still waiting.
CLAUDE_WAIT_TIMEOUT = int(os.environ.get('CLAUDE_WAIT_TIMEOUT', '1800'))
CLAUDE_POLL_SECS = float(os.environ.get('CLAUDE_POLL_SECS', '2'))
_CLAUDE_NOTE_EVERY = 15  # seconds between "still waiting" notices


def _resolve_engine(engine):
    return (engine or ENGINE_DEFAULT or 'bridge').lower()


def _ensure_core_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


# ── Claude file queue ────────────────────────────────────────────────────────────

def claude_queue_dir():
    """The folder Claude Code watches. Lives next to the rendered output so the
    images it references are nearby; override with CLAUDE_QUEUE_DIR."""
    env = os.environ.get('CLAUDE_QUEUE_DIR')
    if env:
        return Path(env)
    base = os.environ.get('MIDI_OUTPUT_DIR')
    if not base:
        try:                       # API process: resolve the same dir as the subprocess
            from app.config import MIDI_OUTPUT_DIR as _m
            base = str(_m)
        except Exception:
            base = os.getcwd()
    return Path(base) / '_claude_queue'


_INSTRUCTIONS = """# Claude transcription queue

The music-import pipeline left transcription tasks here for **Claude Code**.

## How to process (just ask Claude Code: "process the transcription queue")

For every file in `pending/`:

1. Read the request JSON. It has:
   - `image`  — absolute path to a sheet-music image (a system/strip or a
     montage). Read it with the Read tool.
   - `prompt` — exactly what to produce. It always asks for **JSON only**.
2. Look at the image and produce the JSON the prompt asks for.
3. Write that JSON (raw, no markdown fences) to `done/<id>.txt`
   (same id as the request file, e.g. `pending/ab12.json` -> `done/ab12.txt`).

The pipeline polls `done/`, consumes the answer, and continues. You can delete
the `pending/<id>.json` after answering (the pipeline also cleans up).

Tip: process them in id order; each is one system of one page.
"""


def _write_instructions(qdir):
    try:
        (qdir / 'INSTRUCTIONS.md').write_text(_INSTRUCTIONS, encoding='utf-8')
    except Exception:
        pass


def _claude_queue_ask(prompt, image_path=None, label='transcribe',
                      timeout=CLAUDE_WAIT_TIMEOUT):
    """Drop a request for Claude Code and block until it writes the answer."""
    qdir = claude_queue_dir()
    pend, done = qdir / 'pending', qdir / 'done'
    pend.mkdir(parents=True, exist_ok=True)
    done.mkdir(parents=True, exist_ok=True)
    _write_instructions(qdir)

    rid = uuid.uuid4().hex[:8]
    req = {
        'id': rid,
        'label': label,
        'image': str(Path(image_path).resolve()) if image_path else None,
        'prompt': prompt,
        'created': time.time(),
    }
    (pend / f'{rid}.json').write_text(json.dumps(req, indent=2), encoding='utf-8')

    n_pending = len(list(pend.glob('*.json')))
    print(f'[claude] waiting for Claude Code — queued request {rid} ({label}); '
          f'{n_pending} pending in {qdir}', flush=True)
    print('[claude] ask Claude Code in your editor: "process the transcription '
          'queue"', flush=True)

    ans_txt, ans_json = done / f'{rid}.txt', done / f'{rid}.json'
    deadline = time.time() + timeout
    last_note = time.time()
    while time.time() < deadline:
        resp = None
        if ans_txt.exists():
            resp = ans_txt.read_text(encoding='utf-8', errors='replace')
        elif ans_json.exists():
            try:
                d = json.loads(ans_json.read_text(encoding='utf-8'))
                resp = d.get('response') if isinstance(d, dict) and 'response' in d \
                    else json.dumps(d)
            except Exception:
                resp = ans_json.read_text(encoding='utf-8', errors='replace')
        if resp is not None:
            for f in (ans_txt, ans_json, pend / f'{rid}.json'):
                try:
                    f.unlink()
                except Exception:
                    pass
            print(f'[claude] received answer for {rid} ({len(resp)} chars)', flush=True)
            return resp
        if time.time() - last_note >= _CLAUDE_NOTE_EVERY:
            waited = int(time.time() - (deadline - timeout))
            print(f'[claude] still waiting on {rid} ({waited}s) — '
                  f'{len(list(pend.glob("*.json")))} request(s) pending', flush=True)
            last_note = time.time()
        time.sleep(CLAUDE_POLL_SECS)

    raise RuntimeError(f'Claude did not answer request {rid} within {timeout}s '
                       f'(queue: {qdir})')


# ── Availability ─────────────────────────────────────────────────────────────────

def available(engine=None):
    """(ok: bool, detail: str) — whether the engine can run right now."""
    engine = _resolve_engine(engine)
    if engine == 'claude':
        # The queue is always usable; it just needs Claude Code in the session.
        return True, f'Claude Code (file queue at {claude_queue_dir()})'
    _ensure_core_on_path()
    import ai_correct
    if ai_correct._bridge_ping():
        return True, f'browser-ai-bridge ({ai_correct.BRIDGE_URL})'
    return False, (f'browser-ai-bridge not reachable at {ai_correct.BRIDGE_URL} '
                   f'— start it (open an LLM tab) or switch to the Claude engine')


def ping(engine=None):
    return available(engine)[0]


# ── Public primitives (dispatch on engine) ──────────────────────────────────────

def image_ask(image_path, prompt, engine=None, provider='gemini',
              timeout=420, attempts=3, mode=None, label='transcribe'):
    engine = _resolve_engine(engine)
    if engine == 'claude':
        return _claude_queue_ask(prompt, image_path=str(image_path), label=label)
    _ensure_core_on_path()
    import ai_correct
    kw = {'provider': provider, 'timeout': timeout, 'attempts': attempts}
    if mode:
        kw['mode'] = mode
    return ai_correct._bridge_image_ask(image_path, prompt, **kw)


def text_ask(prompt, engine=None, provider='gemini', timeout=420, attempts=3,
             mode=None, label='ask'):
    engine = _resolve_engine(engine)
    if engine == 'claude':
        return _claude_queue_ask(prompt, image_path=None, label=label)
    _ensure_core_on_path()
    import ai_correct
    kw = {'provider': provider, 'timeout': timeout, 'attempts': attempts}
    if mode:
        kw['mode'] = mode
    return ai_correct._bridge_ask(prompt, **kw)


def audio_ask(audio_path, prompt, engine=None, provider='gemini', timeout=600, attempts=2):
    engine = _resolve_engine(engine)
    if engine == 'claude':
        # Claude reads images/text here, not audio; the audio-validate pass is
        # optional, so signal "unsupported" and let the caller skip it.
        raise RuntimeError('audio validation is not supported by the Claude engine')
    _ensure_core_on_path()
    import ai_correct
    return ai_correct._bridge_audio_ask(audio_path, prompt, provider=provider,
                                        timeout=timeout, attempts=attempts)
