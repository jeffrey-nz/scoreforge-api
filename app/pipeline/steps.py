"""
Six-step import pipeline:

  1. detect  – render pages, find systems + barlines (mechanical, ~3s)
  2. read    – AI transcription via pdf_to_midi subprocess (~1-10 min)
  3. pitch   – per-bar key-fit + octave-range check, AI refine failures (~30-60s)
  4. rhythm  – per-bar fill-ratio check, AI refine failures (~20-40s)
  5. theory  – rule-based music-theory validation (~1s)
  6. review  – human sign-off (not run automatically)

Each step function:
  - is an async coroutine
  - mutates job.steps[name] for status/pct/log/result/issues
  - calls job.emit() for SSE events consumed by the client
  - saves job state to disk on completion
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.config import CORE_DIR, MIDI_OUTPUT_DIR, PYTHON
from app.pipeline.job import Job, STEP_ORDER


# ── Helpers ────────────────────────────────────────────────────────────────────

def _core_path() -> Path:
    return CORE_DIR


def _ensure_core_on_path():
    p = str(_core_path())
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_bars_from_cache(job: Job) -> List[Dict]:
    """Reconstruct bars list from page_NN.done.json caches written by ai_transcribe."""
    pages_dir = job.out_dir / '_pages'
    pages_json = pages_dir / 'pages.json'
    if not pages_json.exists():
        return []
    try:
        manifest = json.loads(pages_json.read_text(encoding='utf-8'))
    except Exception:
        return []

    all_bars: List[Dict] = []
    for pg in manifest.get('pages', []):
        # page file name is like 'page_01.png' → page number 1
        m = re.search(r'page_(\d+)', pg.get('file', ''))
        if not m:
            continue
        pnum = int(m.group(1))
        done = pages_dir / f'page_{pnum:02d}.done.json'
        if done.exists():
            try:
                page_bars = json.loads(done.read_text(encoding='utf-8')).get('bars', [])
                all_bars.extend(b for b in page_bars if isinstance(b, dict))
            except Exception:
                pass
    return all_bars


def _load_meta_from_cache(job: Job) -> Dict:
    """Load key/timeSig/bpm from the _pages/meta.json cache."""
    meta_path = job.out_dir / '_pages' / 'meta.json'
    if not meta_path.exists():
        # Fall back to catalog.json
        cat_path = job.out_dir / 'catalog.json'
        if cat_path.exists():
            try:
                cat = json.loads(cat_path.read_text(encoding='utf-8'))
                return {
                    'key': cat.get('key', 'C major'),
                    'timeSig': cat.get('timeSig', '4/4'),
                    'bpm': cat.get('bpm', 120),
                    'title': cat.get('title', job.title),
                    'composer': cat.get('composer', job.composer),
                }
            except Exception:
                pass
        return {'key': 'C major', 'timeSig': '4/4', 'bpm': 120}
    try:
        return json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _annotate_bars(bars: List[Dict], pitch_issues: Dict[int, List], rhythm_issues: Dict[int, List]) -> List[Dict]:
    """Tag each bar with its per-step issues and a confidence level."""
    result = []
    for i, bar in enumerate(bars, 1):
        p_issues = pitch_issues.get(i, [])
        r_issues = rhythm_issues.get(i, [])
        b = dict(bar)
        b['n'] = i
        b['pitch_issues'] = p_issues
        b['rhythm_issues'] = r_issues
        b['issues'] = p_issues + r_issues
        # confidence: 0=bad 1=good
        if p_issues or r_issues:
            b['confidence'] = 0.5 if len(p_issues) + len(r_issues) <= 2 else 0.2
        else:
            b['confidence'] = 1.0
        result.append(b)
    return result


# ── Step 1: Detect ─────────────────────────────────────────────────────────────

async def run_detect(job: Job):
    _ensure_core_on_path()
    import ai_transcribe as atr

    step = job.steps['detect']
    step.status = 'running'
    step.pct = 5
    await job.emit('step', {'step': 'detect', 'status': 'running', 'pct': 5,
                            'msg': 'Rendering PDF pages…'})

    loop = asyncio.get_event_loop()
    pages_dir = job.out_dir / '_pages'
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Render PDF to PNG
    try:
        pages = await loop.run_in_executor(
            None, atr._render_pdf_pages, job.pdf_path, pages_dir
        )
    except Exception as e:
        step.status = 'error'
        step.issues = [{'severity': 'error', 'msg': f'Failed to render PDF: {e}'}]
        await job.emit('step', {'step': 'detect', 'status': 'error', 'msg': str(e)})
        return

    step.pct = 40
    await job.emit('progress', {'step': 'detect', 'pct': 40,
                                'msg': f'{len(pages)} page(s) rendered'})

    # Detect systems + barlines per page
    total_systems = 0
    total_bars_est = 0

    for i, page in enumerate(pages, 1):
        strips = await loop.run_in_executor(
            None, atr._split_page_into_systems, page, pages_dir, i
        )
        page_bars = 0
        for strip in strips:
            hint = await loop.run_in_executor(None, atr._detect_barlines, strip)
            page_bars += hint or 4
        total_systems += len(strips)
        total_bars_est += page_bars

        pct = 40 + int(55 * i / len(pages))
        step.pct = pct
        step.log.append(f'page {i}/{len(pages)}: {len(strips)} system(s), ~{page_bars} bars')
        await job.emit('progress', {
            'step': 'detect', 'pct': pct,
            'msg': f'Page {i}/{len(pages)}: {len(strips)} system(s), ~{page_bars} bars',
        })

    result = {
        'pages': len(pages),
        'systems': total_systems,
        'bars_estimate': total_bars_est,
    }
    step.result = result
    step.status = 'done'
    step.pct = 100
    job.save()
    await job.emit('step', {'step': 'detect', 'status': 'done', 'pct': 100, 'result': result})


# ── Step 2: Read (AI transcription) ────────────────────────────────────────────

def _parse_read_pct(line: str, page_current: int, page_total: int) -> Optional[int]:
    """Estimate 0-98% progress from a transcription log line."""
    if re.search(r'page\s+\d+/\d+.*system', line, re.I):
        # Starting a system — rough per-page progress
        frac = (page_current - 1) / max(page_total, 1)
        return max(5, min(90, int(5 + frac * 80)))
    if 'refine' in line.lower():
        frac = (page_current - 0.3) / max(page_total, 1)
        return max(5, min(92, int(5 + frac * 85)))
    if 'holistic' in line.lower():
        frac = (page_current - 0.1) / max(page_total, 1)
        return max(5, min(94, int(5 + frac * 88)))
    if 'done:' in line and 'bars' in line:
        return 97
    return None


async def run_read(job: Job):
    step = job.steps['read']
    step.status = 'running'
    step.pct = 2
    await job.emit('step', {'step': 'read', 'status': 'running', 'pct': 2,
                            'msg': 'Starting AI transcription…'})

    out_dir = job.out_dir
    env = {
        **os.environ,
        'MIDI_OUTPUT_DIR': str(MIDI_OUTPUT_DIR),
        'PYTHONUNBUFFERED': '1',
        'PYTHONUTF8': '1',
    }
    args = [
        PYTHON, '-u', str(CORE_DIR / 'pdf_to_midi.py'),
        job.pdf_path,
        '--id', job.piece_id,
        '--out', str(out_dir.parent),  # pdf_to_midi writes to out/<piece_id>/
        '--ai-provider', job.provider,
    ]
    if job.title:    args += ['--title', job.title]
    if job.composer: args += ['--composer', job.composer]
    if job.bpm:      args += ['--bpm', str(job.bpm)]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    page_current = 0
    page_total = 1

    async for raw in proc.stdout:
        line = raw.decode(errors='replace').rstrip()
        step.log.append(line)

        # Track page counters
        if m := re.match(r'\[transcribe\]\s+(\d+)\s+page', line):
            page_total = int(m.group(1))
        if m := re.match(r'===\s+page\s+(\d+)/(\d+)', line):
            page_current = int(m.group(1))
            page_total = int(m.group(2))

        pct = _parse_read_pct(line, page_current, page_total)
        if pct:
            step.pct = pct

        # Clean up noisy lines before sending to client
        clean = re.sub(r'\x1b\[[^m]+m', '', line).strip()
        if clean and not re.search(r'NativeCommandError|CategoryInfo|FullyQualified', clean):
            await job.emit('log', {'step': 'read', 'line': clean, 'pct': step.pct})

    await proc.wait()

    if proc.returncode != 0:
        step.status = 'error'
        step.issues = [{'severity': 'error', 'msg': 'AI transcription failed — see log above'}]
        step.pct = step.pct or 10
        job.save()
        await job.emit('step', {'step': 'read', 'status': 'error', 'pct': step.pct})
        return

    # Load results from cache files
    bars = _load_bars_from_cache(job)
    meta = _load_meta_from_cache(job)
    job.bars = [{'n': i+1, 'melody': b.get('melody',''), 'bass': b.get('bass',''),
                 'issues': [], 'confidence': 1.0} for i, b in enumerate(bars)]
    job.meta = meta

    result = {
        'bars': len(job.bars),
        'key': meta.get('key', '?'),
        'timeSig': meta.get('timeSig', '?'),
        'bpm': meta.get('bpm'),
    }
    step.result = result
    step.status = 'done'
    step.pct = 100
    job.save()
    await job.emit('step', {'step': 'read', 'status': 'done', 'pct': 100, 'result': result})


# ── Step 3: Pitch check ────────────────────────────────────────────────────────

def _check_pitch_bar(bar: Dict, key_pcs: Optional[set]) -> List[str]:
    """Return list of pitch-specific issue strings for one bar."""
    _ensure_core_on_path()
    import ai_correct as ac
    import ai_transcribe as atr

    issues = []
    for track in ('melody', 'bass'):
        parsed = ac._parse_rewrite(bar.get(track, ''))
        notes = [m for m, _t in parsed if m is not None]
        if not notes:
            continue

        # Key fit: flag if ≥3 notes are outside the key
        if key_pcs:
            oot = sum(1 for n in notes if n % 12 not in key_pcs)
            if oot >= 3:
                issues.append(f'{track}: {oot} note(s) outside key')

        # Octave range: flag notes well outside their staff band
        band = atr._TREBLE_BAND if track == 'melody' else atr._BASS_BAND
        far = [n for n in notes if n < band[0] - 12 or n > band[1] + 12]
        if far:
            issues.append(f'{track}: {len(far)} note(s) outside expected range')

    return issues


async def _refine_bars(job: Job, flagged: Dict[int, List[str]], step_name: str) -> int:
    """Run the AI bar-refinement pass on a set of flagged bars. Returns fix count."""
    _ensure_core_on_path()
    import ai_correct as ac
    import ai_transcribe as atr

    if not flagged:
        return 0
    if not ac._bridge_ping():
        job.steps[step_name].log.append('[warn] AI bridge not reachable — skipping auto-fix')
        await job.emit('log', {'step': step_name, 'line': '[warn] AI bridge not reachable — skipping auto-fix'})
        return 0

    meta = job.meta
    key = meta.get('key', 'C major')
    timesig = meta.get('timeSig', '4/4')
    title = job.title or meta.get('title', '')
    composer = job.composer or meta.get('composer', '')
    pages_dir = job.out_dir / '_pages'
    CHUNK = 8

    bars = job.bars
    applied_total = 0
    flagged_items = sorted(flagged.items())

    for chunk_start in range(0, len(flagged_items), CHUNK):
        chunk = flagged_items[chunk_start:chunk_start + CHUNK]
        flagged_map = {bn: reasons for bn, reasons in chunk}

        # Build bar crop images for this chunk
        crops = []
        for bn, _ in chunk:
            crop_path = _find_bar_crop(bn, bars, pages_dir)
            if crop_path:
                crops.append((bn, crop_path))

        if not crops:
            continue

        # Build montage
        montage = await asyncio.get_event_loop().run_in_executor(
            None, lambda: atr._build_montage([p for _, p in crops], [bn for bn, _ in crops])
        )
        if not montage:
            continue

        prompt = atr._refine_prompt(title, composer, key, timesig, flagged_map)
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: ac._bridge_image_ask(str(montage), prompt, provider=job.provider)
            )
            data = atr._parse_json(resp)
        except Exception as e:
            job.steps[step_name].log.append(f'[warn] refine call failed: {e}')
            continue

        refinements = data.get('bars', [])
        chunk_bar_nums = {bn for bn, _ in chunk}

        for r in refinements:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r['bar']) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= idx < len(bars)) or (idx + 1) not in chunk_bar_nums:
                continue
            changed = False
            for track in ('melody', 'bass'):
                val = r.get(track, '')
                if val and str(val).strip() and str(val).strip() != bars[idx].get(track, ''):
                    bars[idx][track] = str(val).strip()
                    changed = True
            if changed:
                applied_total += 1

    return applied_total


def _find_bar_crop(bar_n: int, bars: List[Dict], pages_dir: Path) -> Optional[Path]:
    """Find or create a crop image for a specific bar using system strip PNGs."""
    _ensure_core_on_path()
    import ai_transcribe as atr
    from PIL import Image

    # Find which page contains this bar via pages.json
    pages_json = pages_dir / 'pages.json'
    if not pages_json.exists():
        return None

    try:
        manifest = json.loads(pages_json.read_text(encoding='utf-8'))
    except Exception:
        return None

    page_num = None
    for pg in manifest.get('pages', []):
        if pg.get('startBar', 0) <= bar_n <= pg.get('endBar', 0):
            m = re.search(r'page_(\d+)', pg.get('file', ''))
            if m:
                page_num = int(m.group(1))
            break

    if page_num is None:
        return None

    # Look for pre-existing bar crop: page_NN_bar_MM.png
    crop_path = pages_dir / f'page_{page_num:02d}_bar_{bar_n:03d}.png'
    if crop_path.exists():
        return crop_path

    # Recreate from system strips
    strip_paths = sorted(pages_dir.glob(f'page_{page_num:02d}_sys_*.png'))
    if not strip_paths:
        return None

    # Estimate which system and position within it
    pg_start = 0
    for pg in manifest.get('pages', []):
        m = re.search(r'page_(\d+)', pg.get('file', ''))
        if m and int(m.group(1)) == page_num:
            pg_start = pg.get('startBar', 1)
            break

    bar_in_page = bar_n - pg_start  # 0-indexed within page
    bars_per_strip = max(1, 4)       # rough estimate
    strip_idx = min(bar_in_page // bars_per_strip, len(strip_paths) - 1)
    strip_path = strip_paths[strip_idx]

    # Get barline positions
    positions = atr._detect_barline_positions(strip_path)
    if not positions or len(positions) < 2:
        return strip_path  # fall back to whole strip

    # Crop the relevant bar
    bar_in_strip = bar_in_page - strip_idx * bars_per_strip
    bar_in_strip = max(0, min(bar_in_strip, len(positions) - 2))

    try:
        img = Image.open(strip_path)
        x1 = positions[bar_in_strip]
        x2 = positions[bar_in_strip + 1] if bar_in_strip + 1 < len(positions) else img.width
        crop = img.crop((max(0, x1 - 5), 0, min(img.width, x2 + 5), img.height))
        crop.save(str(crop_path))
        return crop_path
    except Exception:
        return strip_path


async def run_pitch(job: Job):
    _ensure_core_on_path()
    import ai_transcribe as atr

    step = job.steps['pitch']
    step.status = 'running'
    step.pct = 5
    await job.emit('step', {'step': 'pitch', 'status': 'running', 'pct': 5,
                            'msg': 'Checking note pitches…'})

    bars = job.bars
    key = job.meta.get('key', 'C major')
    key_pcs = atr._scale_pcs(key)

    # Mechanical pitch check
    flagged: Dict[int, List[str]] = {}
    all_issues: List[Dict] = []

    for bar in bars:
        bn = bar['n']
        reasons = _check_pitch_bar(bar, key_pcs)
        if reasons:
            flagged[bn] = reasons
            for r in reasons:
                all_issues.append({'bar': bn, 'severity': 'warn', 'check': 'pitch', 'msg': r})

    step.pct = 40
    await job.emit('progress', {
        'step': 'pitch', 'pct': 40,
        'msg': f'{len(flagged)} bar(s) flagged for pitch issues',
    })
    for iss in all_issues:
        step.log.append(f'bar {iss["bar"]}: {iss["msg"]}')

    fixed = 0
    if flagged:
        fixed = await _refine_bars(job, flagged, 'pitch')
        step.log.append(f'auto-fixed {fixed} bar(s)')
        await job.emit('progress', {
            'step': 'pitch', 'pct': 90,
            'msg': f'Auto-fixed {fixed} of {len(flagged)} flagged bar(s)',
        })

    # Re-check after fixes
    remaining_issues: List[Dict] = []
    for bar in bars:
        reasons = _check_pitch_bar(bar, key_pcs)
        for r in reasons:
            remaining_issues.append({'bar': bar['n'], 'severity': 'warn', 'check': 'pitch', 'msg': r})
        bar['pitch_issues'] = reasons

    result = {'flagged': len(flagged), 'fixed': fixed, 'remaining': len(remaining_issues)}
    step.result = result
    step.issues = remaining_issues
    step.status = 'done'
    step.pct = 100
    job.save()
    await job.emit('step', {'step': 'pitch', 'status': 'done', 'pct': 100, 'result': result})


# ── Step 4: Rhythm check ───────────────────────────────────────────────────────

def _check_rhythm_bar(bar: Dict, bar_ticks: float) -> List[str]:
    """Return rhythm issue strings for one bar."""
    _ensure_core_on_path()
    import ai_correct as ac

    issues = []
    for track in ('melody', 'bass'):
        parsed = ac._parse_rewrite(bar.get(track, ''))
        ticks = [t for _m, t in parsed]
        total = sum(ticks)
        if total <= 0:
            continue
        ratio = total / bar_ticks
        if ratio < 0.72 or ratio > 1.35:
            pct = int(ratio * 100)
            issues.append(f'{track} fills {pct}% of bar (expected ~100%)')

    return issues


async def run_rhythm(job: Job):
    step = job.steps['rhythm']
    step.status = 'running'
    step.pct = 5
    await job.emit('step', {'step': 'rhythm', 'status': 'running', 'pct': 5,
                            'msg': 'Checking note durations…'})

    bars = job.bars
    timesig = job.meta.get('timeSig', '4/4')
    DIV = 16

    try:
        num, den = map(int, str(timesig).split('/'))
        bar_ticks = (num * 4 / den) * DIV
    except Exception:
        bar_ticks = 4 * DIV

    flagged: Dict[int, List[str]] = {}
    all_issues: List[Dict] = []

    for bar in bars:
        bn = bar['n']
        reasons = _check_rhythm_bar(bar, bar_ticks)
        if reasons:
            flagged[bn] = reasons
            for r in reasons:
                all_issues.append({'bar': bn, 'severity': 'warn', 'check': 'rhythm', 'msg': r})

    step.pct = 40
    await job.emit('progress', {
        'step': 'rhythm', 'pct': 40,
        'msg': f'{len(flagged)} bar(s) flagged for rhythm issues',
    })
    for iss in all_issues:
        step.log.append(f'bar {iss["bar"]}: {iss["msg"]}')

    fixed = 0
    if flagged:
        fixed = await _refine_bars(job, flagged, 'rhythm')
        step.log.append(f'auto-fixed {fixed} bar(s)')
        await job.emit('progress', {
            'step': 'rhythm', 'pct': 90,
            'msg': f'Auto-fixed {fixed} of {len(flagged)} flagged bar(s)',
        })

    remaining_issues: List[Dict] = []
    for bar in bars:
        reasons = _check_rhythm_bar(bar, bar_ticks)
        for r in reasons:
            remaining_issues.append({'bar': bar['n'], 'severity': 'warn', 'check': 'rhythm', 'msg': r})
        bar['rhythm_issues'] = reasons
        bar['issues'] = bar.get('pitch_issues', []) + reasons

    result = {'flagged': len(flagged), 'fixed': fixed, 'remaining': len(remaining_issues)}
    step.result = result
    step.issues = remaining_issues
    step.status = 'done'
    step.pct = 100
    job.save()
    await job.emit('step', {'step': 'rhythm', 'status': 'done', 'pct': 100, 'result': result})


# ── Step 5: Theory check ───────────────────────────────────────────────────────

async def run_theory(job: Job):
    step = job.steps['theory']
    step.status = 'running'
    step.pct = 10
    await job.emit('step', {'step': 'theory', 'status': 'running', 'pct': 10,
                            'msg': 'Running music-theory validation…'})

    loop = asyncio.get_event_loop()

    # First write current bars back to MIDI (in case pitch/rhythm steps edited them)
    await _write_bars_to_midi(job)

    env = {**os.environ, 'MIDI_OUTPUT_DIR': str(MIDI_OUTPUT_DIR)}
    proc = await asyncio.create_subprocess_exec(
        PYTHON, '-u', str(CORE_DIR / 'theory_check.py'), job.piece_id, '--json',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()

    step.pct = 90
    try:
        result = json.loads(stdout.decode(errors='replace').strip())
    except Exception:
        result = {'status': 'unknown', 'score': 0, 'issues': [],
                  'raw': stdout.decode(errors='replace')[:300]}

    issues = [{'severity': i.get('severity', 'warn'), 'check': i.get('check', ''),
               'msg': i.get('message', '')}
              for i in result.get('issues', [])]

    step.result = {
        'status': result.get('status', 'unknown'),
        'score': result.get('score', 0),
        'key': result.get('key'),
        'inKeyPct': result.get('inKeyPct'),
        'noteCount': result.get('noteCount'),
    }
    step.issues = issues
    step.status = 'done'
    step.pct = 100
    job.save()
    await job.emit('step', {'step': 'theory', 'status': 'done', 'pct': 100,
                            'result': step.result, 'issues': issues})


async def _write_bars_to_midi(job: Job):
    """Write current job.bars back to MIDI files (after any edits)."""
    if not job.bars:
        return
    _ensure_core_on_path()
    import ai_transcribe as atr

    meta = job.meta
    try:
        num, den = map(int, str(meta.get('timeSig', '4/4')).split('/'))
        qL_per_bar = num * 4 / den
    except Exception:
        qL_per_bar = 4.0
    bpm = int(meta.get('bpm') or 120)

    # Convert job.bars format [{melody, bass}] to the all_bars format ai_transcribe uses
    all_bars = [{'melody': b.get('melody', ''), 'bass': b.get('bass', '')} for b in job.bars]

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: atr._write_piece(job.piece_id, job.out_dir, all_bars, {
            'timeSig': meta.get('timeSig', '4/4'),
            'bpm': bpm,
            'key': meta.get('key', 'C major'),
            'title': job.title or meta.get('title', ''),
            'composer': job.composer or meta.get('composer', ''),
            'importedFrom': meta.get('importedFrom', ''),
        })
    )


# ── Approve (finalize) ─────────────────────────────────────────────────────────

async def run_approve(job: Job):
    """Write final MIDI files from current bars and mark job done."""
    step = job.steps['review']
    await job.emit('step', {'step': 'review', 'status': 'running', 'pct': 50,
                            'msg': 'Writing MIDI files…'})
    await _write_bars_to_midi(job)
    job.approved = True
    step.status = 'approved'
    job.save()
    await job.emit('step', {'step': 'review', 'status': 'approved', 'pct': 100,
                            'msg': 'Import complete — piece added to showcase'})


# ── Step runner ────────────────────────────────────────────────────────────────

_STEP_FNS = {
    'detect': run_detect,
    'read':   run_read,
    'pitch':  run_pitch,
    'rhythm': run_rhythm,
    'theory': run_theory,
}


async def run_step(job: Job, step_name: str):
    """Run a single step, then auto-advance through remaining steps."""
    if step_name not in _STEP_FNS:
        return
    try:
        await _STEP_FNS[step_name](job)
    except Exception as e:
        job.steps[step_name].status = 'error'
        job.steps[step_name].issues = [{'severity': 'error', 'msg': str(e)}]
        job.save()
        await job.emit('step', {'step': step_name, 'status': 'error', 'msg': str(e)})
        return

    # Auto-advance: if current step succeeded, start the next one automatically
    if job.steps[step_name].status == 'done':
        idx = STEP_ORDER.index(step_name)
        if idx + 1 < len(STEP_ORDER):
            next_step = STEP_ORDER[idx + 1]
            if job.steps[next_step].status == 'idle':
                await run_step(job, next_step)
