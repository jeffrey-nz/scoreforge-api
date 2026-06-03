"""Runtime configuration — all values can be overridden by environment variables."""
import os
from pathlib import Path

# Python interpreter to use when spawning core scripts.
# Default: the python on PATH. Override with PROCMUSIC_PYTHON.
PYTHON = os.environ.get("PROCMUSIC_PYTHON", "python")

# Directory that contains the showcase-midi/<id>/ folders.
# The scoreforge-api writes transcribed MIDI here so the dashboard can serve it.
# Override with MIDI_OUTPUT_DIR.
_default_midi = Path(__file__).parent.parent.parent.parent / "procmusic-dashboard" / "dashboard" / "showcase-midi"
MIDI_OUTPUT_DIR = Path(os.environ.get("MIDI_OUTPUT_DIR", str(_default_midi)))

# Where the core scripts live (inside this package).
CORE_DIR = Path(__file__).parent / "core"
