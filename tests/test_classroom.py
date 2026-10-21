# Apache-2.0
"""The classroom path: the 수업용 (CPU) preset's settings, and "Pibo로 보내기"."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from test_app import image_bytes, upload

ROOT = Path(__file__).resolve().parents[1]


def _dataset(client):
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})


def test_queries_and_denoising_are_stored_checked_and_carried_into_a_resume(studio):
    client, module = studio
    _dataset(client)

    def stored(**extra):
        response = client.post("/api/jobs", json={"dataset_id": 1, **extra})
        assert response.status_code == 200, response.text
        job = module.db.one("SELECT * FROM jobs WHERE id = ?", (response.json()["id"],))
        return job["queries"], job["denoising"]

    assert stored() == (None, None)                       # the model's own 300 / 100
    assert stored(queries="", denoising="") == (None, None)
    assert stored(queries=100, denoising=30) == (100, 30)
    assert stored(queries="10", denoising=0) == (10, 0)   # 0 denoising: none at all
    assert stored(queries=300, denoising=100) == (300, 100)
    for bad in ({"queries": 9}, {"queries": 301}, {"denoising": -1}, {"denoising": 101},
                {"queries": "many"}, {"queries": 10.5}):
        response = client.post("/api/jobs", json={"dataset_id": 1, **bad})
        assert response.status_code == 400, bad
        assert response.json()["error"].startswith(("쿼리 수", "디노이징 쿼리")), bad

    job = client.post("/api/jobs", json={"dataset_id": 1, "queries": 100,
                                         "denoising": 30}).json()["id"]
    weights = module.RUNS / f"job{job}" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"w")
    module.db.update_job(job, status="done", run_dir=str(weights.parent))
    more = client.post(f"/api/jobs/{job}/resume", json={"add_epochs": 5}).json()["id"]
    again = module.db.one("SELECT queries, denoising FROM jobs WHERE id = ?", (more,))
    assert again == {"queries": 100, "denoising": 30}


def test_the_worker_hands_queries_and_denoising_to_train(studio, monkeypatch):
    from worker import _decoder_kwargs

    assert _decoder_kwargs({}) == {}                      # a row from before the columns
    assert _decoder_kwargs({"queries": None, "denoising": None}) == {}
    assert _decoder_kwargs({"queries": 100, "denoising": 0}) == {"queries": 100, "denoising": 0}

    client, module = studio
    _dataset(client)
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1, "model": "dfine-n",
                                         "queries": 100, "denoising": 30}).json()["id"]
    asked = {}

    class Detector:
        def __init__(self, name, verbose=True, task="detect"):
            asked["model"] = name

        def train(self, **kwargs):
            asked.update(kwargs)
            run = module.RUNS / "job1"
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "best.pt").write_bytes(b"x")
            (run / "summary.json").write_text(json.dumps({"best_map50_95": 0.5}))
            return run / "weights" / "best.pt"

        def export(self, **kwargs):
            pass

    import easydetect
    monkeypatch.setattr(easydetect, "Detector", Detector)
    module.worker._train(module.db.one("SELECT * FROM jobs WHERE id = ?", (job,)))
    assert asked["model"] == "dfine-n"
    assert asked["queries"] == 100 and asked["denoising"] == 30


def test_the_classroom_preset_is_in_the_form_and_picked_without_a_gpu(studio):
    client, _ = studio
    page = client.get("/").text
    assert 'data-preset="classroom">수업용 (CPU)<' in page
    preset = re.search(r"classroom: \{([^}]*)\}", page).group(1)
    for setting in ('model: "dfine-n"', "imgsz: 320", "batch: 8", 'freeze: "backbone"',
                    "queries: 100", "denoising: 30", 'epochs: "auto"', "patience: 25"):
        assert setting in preset, setting
    # the classroom epochs follow the dataset: about 1,000 steps, 35 to 100 epochs
    assert "const CLASSROOM_STEPS = 1000;" in page
    assert "Math.min(100, Math.max(35, Math.round(CLASSROOM_STEPS / steps)))" in page
    assert 'id="job-queries"' in page and 'id="job-denoising"' in page
    assert "<option>dfine-n</option>" in page
    # the first status without a CUDA card presses it
    assert re.search(r'if \(!state\.gpu\) document\.querySelector\(\'#presets '
                     r'button\[data-preset="classroom"\]\'\)\.click\(\)', page)
    i18n = client.get("/static/js/i18n.js").text
    assert '"수업용 (CPU)": "Classroom (CPU)"' in i18n and '"Pibo로 보내기": "Send to Pibo"' in i18n


def test_the_lab_asks_for_the_easydetect_that_has_queries():
    import run

    assert run.EASYDETECT_AT_LEAST == "0.7.7"
    assert "easydetect[train]>=0.7.7" in (ROOT / "requirements.txt").read_text()


# ---------------------------------------------------------------- Pibo


def _tiny_checkpoint(path: Path, names: dict[int, str], imgsz: int = 160) -> Path:
    """A dfine-n checkpoint as training writes one, with untrained weights:
    100 queries, 30 denoising, and class scores near 0.5 so boxes come out."""
    torch = pytest.importorskip("torch")
    from easydetect.nn import DFINENet

    torch.manual_seed(0)
    net = DFINENet("n", len(names), pretrained_backbone=False)
    net.trim(queries=100)
    with torch.no_grad():                  # the 0.01 prior would leave nothing above conf
        for name, p in net.named_parameters():
            if "score_head" in name and name.endswith("bias"):
                p.zero_()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": "n", "num_classes": len(names), "names": names,
                "model": net.state_dict(), "imgsz": imgsz, "queries": 100,
                "denoising": 30}, path)
    return path


def _picture(path: Path) -> Path:
    import cv2

    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), 90, np.uint8)
    img[60:180, 80:200] = (40, 160, 230)
    img[100:140, 220:300] = (200, 60, 60)
    img = cv2.add(img, rng.integers(0, 30, img.shape, dtype=np.uint8))
    cv2.imwrite(str(path), img)
    return path


def _load_detect_py(folder: Path):
    spec = importlib.util.spec_from_file_location("pibo_detect_under_test", folder / "detect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pibo_zip_runs_without_easydetect_and_matches_it(studio, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path / "cache"))
    client, module = studio
    _dataset(client)
    job = client.post("/api/jobs", json={"dataset_id": 1, "imgsz": 160}).json()["id"]
    run = module.RUNS / f"job{job}"
    best = _tiny_checkpoint(run / "weights" / "best.pt", {0: "cup", 1: "공"})
    module.db.update_job(job, status="done", run_dir=str(run))

    shown = client.get(f"/api/jobs/{job}").json()
    assert "weights" in shown["artifacts"]
    response = client.get(f"/api/jobs/{job}/download/pibo")
    assert response.status_code == 200, response.text
    folder = f"pibo-job{job}"
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert sorted(zf.namelist()) == [f"{folder}/{n}" for n in
                                         ("README.txt", "detect.py", "labels.txt", "model.json",
                                          "model.onnx")]
        zf.extractall(tmp_path / "pibo")
    here = tmp_path / "pibo" / folder
    assert (here / "labels.txt").read_text(encoding="utf-8") == "cup\n공\n"
    readme = (here / "README.txt").read_text(encoding="utf-8-sig")
    assert f"작업 #{job}" in readme and "160x160" in readme and "{" not in readme
    meta = json.loads((here / "model.json").read_text(encoding="utf-8"))
    assert meta["imgsz"] == 160 and meta["names"] == ["cup", "공"]
    assert meta["conf"] is None or 0.25 <= meta["conf"] <= 0.7
    # the run's zip is written whole or not at all: nothing half-done is left
    assert not [p for p in run.iterdir() if p.name.startswith(".")]

    import onnxruntime as ort

    session = ort.InferenceSession(str(here / "model.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 160, 160]       # the run's input size
    assert session.get_outputs()[1].shape[1] == 100                # the trained query count

    # detect.py against easydetect's own predictions, on the same picture
    picture = _picture(tmp_path / "scene.jpg")
    from easydetect import Detector

    ours = Detector(str(best), backend="onnxruntime", verbose=False)
    detect = _load_detect_py(here)
    for conf in (0.0, 0.5):
        theirs = ours.predict(str(picture), conf=conf, verbose=False)[0]
        expected = theirs.boxes.data
        found = detect.detect(str(picture), conf=conf)
        assert len(found) == len(expected) > 0, conf
        got = np.array([f[:4] for f in found], np.float32)
        assert np.abs(got - expected[:, :4]).max() <= 1.0
        assert np.abs(np.array([f[4] for f in found]) - expected[:, 4]).max() <= 1e-4
        assert [f[5] for f in found] == [ours.names[int(c)] for c in expected[:, 5]]
        assert all(isinstance(f[4], float) for f in found)

    # called the way openpibo calls its YoloOnnx: Detect.object_detector = Detector(...)
    import cv2

    det = detect.Detector(str(Path(here) / "model.onnx"), imgsz=320)
    img = cv2.imread(str(picture))
    boxes, scores, classes = det(img, conf=0.5, iou=0.4)
    assert boxes.shape[1] == 4 and len(boxes) == len(scores) == len(classes)
    assert classes.dtype.kind == "i"
    expected = detect.detect(str(picture), conf=0.5, iou=0.4)
    assert [det.name_of(c) for c in classes] == [f[5] for f in expected]
    items = [{"name": det.name_of(c), "score": int(s * 100), "box": tuple(int(v) for v in b)}
             for b, s, c in zip(boxes, scores, classes, strict=True)]  # openpibo's detect_object
    assert all(i["name"] for i in items)

    # the zip is made once per best.pt and then reused
    stamp = (run / "pibo" / "model.onnx").stat().st_mtime_ns
    assert client.get(f"/api/jobs/{job}/download/pibo").status_code == 200
    assert (run / "pibo" / "model.onnx").stat().st_mtime_ns == stamp

    # and it runs as a program with plain Python: no easydetect, no torch
    check = ("import sys; sys.argv[1:] = [sys.argv[1]]; import detect; "
             "rc = detect.main([sys.argv[1], '--conf', '0']); "
             "assert 'torch' not in sys.modules and 'easydetect' not in sys.modules; "
             "sys.exit(rc)")
    out = subprocess.run([sys.executable, "-c", check, str(picture)], cwd=here,
                         capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert "input 160x160, 2 classes (cup, 공)" in out.stdout


def test_pibo_needs_trained_weights(studio):
    client, module = studio
    _dataset(client)
    job = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    run = module.RUNS / f"job{job}"
    run.mkdir(parents=True)
    module.db.update_job(job, status="failed", run_dir=str(run))
    assert client.get(f"/api/jobs/{job}/download/pibo").status_code == 404


def test_the_pibo_threshold_is_picked_on_the_validation_pictures(tmp_path, monkeypatch):
    """Best F1 among 0.25-0.70: a model whose right boxes score 0.6 and wrong
    ones 0.4 gets a threshold between them."""
    import pibo_export

    data = tmp_path / "data.yaml"
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for i in range(3):
            _picture(tmp_path / "images" / split / f"{i}.jpg")         # 320 x 240
            (tmp_path / "labels" / split / f"{i}.txt").write_text("0 0.4375 0.5 0.375 0.5\n")
    data.write_text("path: .\ntrain: images/train\nval: images/val\nnames: {0: cup}\n")

    class Fake:
        def __call__(self, img, conf=None, iou=None):
            boxes = np.array([[80, 60, 200, 180], [220, 100, 300, 140]], np.float32)
            scores = np.array([0.6, 0.4], np.float32)
            keep = scores >= conf
            return boxes[keep], scores[keep], np.zeros(int(keep.sum()), np.int64)

    monkeypatch.setattr(pibo_export, "_pibo_detector", lambda onnx: Fake())
    assert 0.4 < pibo_export.pick_conf(tmp_path / "m.onnx", data) <= 0.6
    assert pibo_export.pick_conf(tmp_path / "m.onnx", None) is None
    (tmp_path / "noval.yaml").write_text("path: .\ntrain: images/train\nnames: {0: cup}\n")
    assert pibo_export.pick_conf(tmp_path / "m.onnx", tmp_path / "noval.yaml") is None
    # under 4 labelled pictures the lab validates on the training ones: no pick then
    (tmp_path / "same.yaml").write_text("path: .\ntrain: images/val\nval: images/val\n"
                                        "names: {0: cup}\n")
    assert pibo_export.pick_conf(tmp_path / "m.onnx", tmp_path / "same.yaml") is None


def test_a_failed_threshold_pick_leaves_no_stale_model_json_behind(tmp_path, monkeypatch):
    """The export after a new best.pt fails half-way: the next one starts over
    instead of shipping the old names and threshold with the new model."""
    import os

    import pibo_export

    weights = _tiny_checkpoint(tmp_path / "run" / "weights" / "best.pt", {0: "cup"})
    out = tmp_path / "run" / "pibo"
    _, meta = pibo_export.export_onnx(weights, out, 160)
    assert meta["names"] == ["cup"]
    _tiny_checkpoint(weights, {0: "cup", 1: "공"})                # trained again
    stamp = (out / "model.onnx").stat().st_mtime_ns + 10**9
    os.utime(weights, ns=(stamp, stamp))
    monkeypatch.setattr(pibo_export, "pick_conf", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        pibo_export.export_onnx(weights, out, 160)
    monkeypatch.undo()
    _, meta = pibo_export.export_onnx(weights, out, 160)
    assert meta["names"] == ["cup", "공"]


def test_detect_py_uses_the_model_s_threshold_and_a_pinned_one_wins(tmp_path):
    pytest.importorskip("onnxruntime")
    from easydetect.exporter import export_onnx
    from easydetect.nn import DFINENet

    net = DFINENet("n", 1, pretrained_backbone=False)
    onnx = Path(export_onnx(net, {0: "cup"}, imgsz=64, out_dir=tmp_path, fname="model",
                            verbose=False))
    detect = _load_detect_py(Path(__file__).resolve().parents[1] / "pibo")
    assert detect.Detector(str(onnx)).thresholds() == (0.5, 0.7)      # no model.json
    (tmp_path / "model.json").write_text('{"conf": 0.35}')
    det = detect.Detector(str(onnx))
    assert det.thresholds() == (0.35, 0.7)          # the model's own, when the call gives none
    assert det.thresholds(0.5, 0.4) == (0.5, 0.4)   # a call's own values otherwise
    det = detect.Detector(str(onnx), conf="auto")
    assert det.thresholds(0.5, 0.4) == (0.35, 0.4)  # openpibo's 0.5 does not override it
    det = detect.Detector(str(onnx), conf=0.6, iou=0.5)
    assert det.thresholds(0.5, 0.4) == (0.6, 0.5)
