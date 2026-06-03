"""Runtime configuration — all values can be overridden by environment variables."""
import os
from pathlib import Path

# Python interpreter to use when spawning core scripts.
# Default: the python on PATH. Override with PROCMUSIC_PYTHON.
import sys as _sys
PYTHON = os.environ.get("PROCMUSIC_PYTHON", _sys.executable)

# Directory that contains the showcase-midi/<id>/ folders.
# Defaults to ../procmusic/dashboard/showcase-midi (sibling repo layout on dev machine).
# Override with MIDI_OUTPUT_DIR env var or .env file.
_here = Path(__file__).parent.parent  # scoreforge-api/
_default_midi = _here.parent / "procmusic" / "dashboard" / "showcase-midi"
MIDI_OUTPUT_DIR = Path(os.environ.get("MIDI_OUTPUT_DIR", str(_default_midi)))

# Where the core scripts live (inside this package).
CORE_DIR = Path(__file__).parent / "core"
