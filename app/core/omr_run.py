"""Run the oemer ML-OMR engine CPU-only and return compact bars.

This environment has the GPU onnxruntime build installed, whose default provider
list (CUDA/TensorRT first) hangs at InferenceSession creation because there is no
CUDA runtime — that was the long-standing oemer "deadlock". We force every
onnxruntime session onto the CPU provider before oemer builds one, run the
end-to-end pipeline (deskew off — these scans are straight), then hand the
MusicXML to omer_import for conversion to our bar note-strings.

CLI:  python -m app.core.omr_run <image> <out_dir>     # writes <out_dir>/<name>.musicxml
"""
from __future__ import annotations
import os
from pathlib import Path


def force_cpu_onnx() -> None:
    """Make every onnxruntime.InferenceSession CPU-only (skips the CUDA/TensorRT
    provider probe that hangs without a CUDA runtime)."""
    import onnxruntime as ort
    if getattr(ort, '_omr_cpu_forced', False):
        return
    orig = ort.InferenceSession

    def cpu_session(*args, **kwargs):
        kwargs['providers'] = ['CPUExecutionProvider']
        return orig(*args, **kwargs)

    ort.InferenceSession = cpu_session
    ort._omr_cpu_forced = True


def run_oemer(image_path: str, out_dir: str) -> str:
    """Run oemer on one image; returns the written .musicxml path."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    force_cpu_onnx()
    from argparse import Namespace
    from oemer import ete
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    args = Namespace(img_path=str(image_path), output_path=str(out_dir),
                     use_tf=False, save_cache=False, without_deskew=True)
    return ete.extract(args)


if __name__ == '__main__':
    import sys
    print('musicxml ->', run_oemer(sys.argv[1], sys.argv[2]))
