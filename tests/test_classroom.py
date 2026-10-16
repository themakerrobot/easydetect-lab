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
                    "queries: 100", "denoising: 30", "epochs: 35", "patience: 10"):
        assert setting in preset, setting
    assert 'id="job-queries"' in page and 'id="job-denoising"' in page
    assert "<option>dfine-n</option>" in page
    # the first status without a CUDA card presses it
    assert re.search(r'if \(!state\.gpu\) document\.querySelector\(\'#presets '
                     r'button\[data-preset="classroom"\]\'\)\.click\(\)', page)
    i18n = client.get("/static/js/i18n.js").text
    assert '"수업용 (CPU)": "Classroom (CPU)"' in i18n and '"Pibo로 보내기": "Send to Pibo"' in i18n


def test_the_lab_asks_for_the_easydetect_that_has_queries():
    import run

    assert run.EASYDETECT_AT_LEAST == "0.7.4"
    assert "easydetect[train]>=0.7.4" in (ROOT / "requirements.txt").read_text()


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
                                         ("README.txt", "detect.py", "labels.txt", "model.onnx")]
        zf.extractall(tmp_path / "pibo")
    here = tmp_path / "pibo" / folder
    assert (here / "labels.txt").read_text(encoding="utf-8") == "cup\n공\n"
    readme = (here / "README.txt").read_text(encoding="utf-8-sig")
    assert f"작업 #{job}" in readme and "160x160" in readme and "{" not in readme

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
