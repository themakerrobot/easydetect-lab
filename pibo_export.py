# Apache-2.0
""""Pibo로 보내기": a finished run as a zip a robot runs with numpy, OpenCV and
ONNX Runtime alone.

The zip holds ``model.onnx`` (exported from ``best.pt`` at the run's input
size), ``labels.txt``, the standalone ``detect.py`` from ``pibo/`` and a
Korean ``README.txt``. Nothing in it imports easydetect or torch.
"""

from __future__ import annotations

import json
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "pibo"

#: The input size when a run did not record one.
DEFAULT_IMGSZ = 320

_lock = threading.Lock()   # two clicks at once must not export into one folder together


def export_onnx(weights: Path, out_dir: Path, imgsz: int) -> tuple[Path, list[str], int]:
    """``model.onnx`` and the class names, exported once per ``best.pt``.

    Kept in ``out_dir`` and reused while it is newer than the weights: a
    second download is instant, and a resumed run that wrote a new best.pt
    exports again.
    """
    from easydetect import Detector

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = out_dir / "model.onnx"
    info = out_dir / "model.json"
    with _lock:
        fresh = (onnx.exists() and info.exists()
                 and onnx.stat().st_mtime_ns >= weights.stat().st_mtime_ns
                 and json.loads(info.read_text(encoding="utf-8")).get("imgsz") == imgsz)
        if not fresh:
            model = Detector(str(weights), verbose=False)
            written = Path(model.export(format="onnx", imgsz=imgsz, out_dir=out_dir,
                                        verbose=False))
            written.replace(onnx)
            names = {int(k): str(v) for k, v in (model.names or {}).items()}
            queries = int(model.net.decoder.num_queries)
            # the exporter's own sidecar names the file it wrote; ours is model.json
            (out_dir / f"{written.stem}.names.json").unlink(missing_ok=True)
            info.write_text(json.dumps({
                "imgsz": imgsz, "queries": queries,
                "names": [names.get(i, str(i)) for i in range(max(names, default=-1) + 1)],
            }, ensure_ascii=False), encoding="utf-8")
        meta = json.loads(info.read_text(encoding="utf-8"))
    return onnx, meta["names"], meta["queries"]


def _text(bundle: zipfile.ZipFile, name: str, text: str) -> None:
    """A text file readable by everyone once unzipped (writestr's default is 0600)."""
    info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    bundle.writestr(info, text.encode("utf-8"))


def build(job: dict, run_dir: Path) -> Path:
    """The zip for job ``job``; returns its path inside ``run_dir``."""
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)
    imgsz = int(job.get("imgsz") or DEFAULT_IMGSZ)
    onnx, names, queries = export_onnx(weights, run_dir / "pibo", imgsz)
    folder = f"pibo-job{job['id']}"
    readme = (TEMPLATE / "README.txt").read_text(encoding="utf-8")
    for key, value in {"JOB": job["id"], "IMGSZ": imgsz, "QUERIES": queries,
                       "CLASSES": ", ".join(names)}.items():
        readme = readme.replace("{" + key + "}", str(value))
    archive = run_dir / f"{folder}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(onnx, f"{folder}/model.onnx")
        _text(bundle, f"{folder}/labels.txt", "\n".join(names) + "\n")
        _text(bundle, f"{folder}/detect.py", (TEMPLATE / "detect.py").read_text(encoding="utf-8"))
        # UTF-8 with a BOM, so that an old Windows Notepad shows the Hangul right too
        _text(bundle, f"{folder}/README.txt", "\ufeff" + readme)
    return archive
