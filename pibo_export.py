# Apache-2.0
""""ONNX 파일 하나": a finished run as one .onnx file Pibo (or any PC) runs with
numpy, OpenCV and ONNX Runtime alone.

The file is exported from ``best.pt`` at the run's input size and carries
inside it (ONNX metadata) the class names (``easydetect.names``) and the score
a box needs, picked on the validation pictures (``easydetect.conf``).
``pibo/detect.py`` runs it on its own and is what the robot's reader matches.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "pibo"

#: The input size when a run did not record one.
DEFAULT_IMGSZ = 320
#: The scores a picked threshold may take: below 0.25 a robot calls out
#: guesses, above 0.7 it misses what it was trained on.
CONF_STEPS = [round(0.25 + 0.05 * i, 2) for i in range(10)]
#: Validation pictures looked at for it, at most.
CONF_IMAGES = 200

_lock = threading.Lock()   # two clicks at once must not export into one folder together


def _pibo_detector(onnx: Path):
    """detect.py's own Detector: the threshold is picked on the boxes Pibo will see."""
    spec = importlib.util.spec_from_file_location("pibo_detect", TEMPLATE / "detect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Detector(str(onnx), labels=os.devnull, conf=0.0)


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_conf(onnx: Path, data_yaml: Path | None) -> float | None:
    """The score threshold with the best F1 on the validation pictures (a box
    right at IoU 0.5 or more, same class), among ``CONF_STEPS``; the higher
    one on a tie. None without validation boxes to count."""
    if data_yaml is None or not Path(data_yaml).exists():
        return None
    from easydetect.data.dataset import DetDataset, load_data_yaml
    from easydetect.utils.imageio import imread

    try:
        cfg = load_data_yaml(str(data_yaml))
        if cfg.get("val") in (None, cfg.get("train")):
            return None                 # validated on the training pictures: it would flatter
        val = DetDataset(str(data_yaml), "val", 320, augment=False)
    except (ValueError, FileNotFoundError, KeyError):
        return None
    det = _pibo_detector(onnx)
    scored, truths = [], 0          # (score, right) for every box found
    for path in val.files[:CONF_IMAGES]:
        img = imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        truth = [(int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                  (cy + bh / 2) * h) for c, cx, cy, bw, bh in val._load_labels(path).tolist()]
        truths += len(truth)
        boxes, scores, classes = det(img, conf=min(CONF_STEPS))
        used = set()
        for box, score, c in zip(boxes.tolist(), scores.tolist(), classes.tolist(), strict=True):
            best, which = 0.5, None
            for k, (tc, *corners) in enumerate(truth):
                if k not in used and tc == int(c) and _iou(box, corners) >= best:
                    best, which = _iou(box, corners), k
            if which is not None:
                used.add(which)
            scored.append((score, which is not None))
    if not truths:
        return None
    best_f1, best_conf = -1.0, None
    for conf in CONF_STEPS:
        tp = sum(right for score, right in scored if score >= conf)
        found = sum(score >= conf for score, _ in scored)
        f1 = 2 * tp / (found + truths) if found + truths else 0.0
        if f1 >= best_f1 - 1e-9:
            best_f1, best_conf = f1, conf
    return best_conf


def export_onnx(weights: Path, out_dir: Path, imgsz: int,
                data_yaml: Path | None = None) -> tuple[Path, dict]:
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
        # both written after this best.pt (model.json last: a failed export
        # leaves the old one older than the weights, so it is not trusted)
        fresh = (onnx.exists() and info.exists()
                 and onnx.stat().st_mtime_ns >= weights.stat().st_mtime_ns
                 and info.stat().st_mtime_ns >= weights.stat().st_mtime_ns
                 and json.loads(info.read_text(encoding="utf-8")).get("imgsz") == imgsz)
        if fresh and "conf" not in json.loads(info.read_text(encoding="utf-8")):
            fresh = False                     # written before the threshold was kept
        if not fresh:
            model = Detector(str(weights), verbose=False)
            info.unlink(missing_ok=True)      # whatever happens next, not the old one
            written = Path(model.export(format="onnx", imgsz=imgsz, out_dir=out_dir,
                                        verbose=False))
            names = {int(k): str(v) for k, v in (model.names or {}).items()}
            queries = int(model.net.decoder.num_queries)
            # the exporter's own sidecar names the file it wrote; ours is model.json
            (out_dir / f"{written.stem}.names.json").unlink(missing_ok=True)
            conf = pick_conf(written, data_yaml)
            written.replace(onnx)
            _write_atomic(info, json.dumps({
                "imgsz": imgsz, "queries": queries,
                "names": [names.get(i, str(i)) for i in range(max(names, default=-1) + 1)],
                # None: no validation pictures to pick it on, detect.py's 0.5
                "conf": conf,
            }, ensure_ascii=False).encode("utf-8"))
        meta = json.loads(info.read_text(encoding="utf-8"))
    return onnx, meta


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build(job: dict, run_dir: Path, data_yaml: Path | None = None) -> Path:
    """The one model file for job ``job``: ``job<id>.onnx`` in ``run_dir``,
    with the class names and the picked score threshold inside it.

    Written under a temporary name and moved into place: a download that
    starts while another click is still writing it gets the whole file."""
    from easydetect.exporter import embed_metadata
    from easydetect.predictor import ONNX_CONF_KEY

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)
    imgsz = int(job.get("imgsz") or DEFAULT_IMGSZ)
    onnx, meta = export_onnx(weights, run_dir / "pibo", imgsz, data_yaml)
    target = run_dir / f"job{job['id']}.onnx"
    tmp = run_dir / f".job{job['id']}.{os.getpid()}.{threading.get_ident()}.onnx"
    try:
        shutil.copyfile(onnx, tmp)
        conf = meta.get("conf")
        # the names are inside already (the exporter puts them there)
        embed_metadata(tmp, {ONNX_CONF_KEY: None if conf is None else f"{conf:.2f}"})
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
