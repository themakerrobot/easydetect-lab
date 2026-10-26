# Apache-2.0
"""Who may open what, requests that do not wait on each other, caches that
notice change, and jobs that clean up after themselves."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from test_app import image_bytes, upload

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def settle(root: Path, age: float = 60.0) -> None:
    """Every folder under ``root`` as if last changed a minute ago: a listing
    of a folder that changed within the last two seconds is never kept."""
    when = time.time() - age
    for here, _, _ in os.walk(root):
        os.utime(here, (when, when))


class FakeResult:
    def __init__(self, frame):
        self.frame = frame

    def plot(self):
        return self.frame

    def summary(self):
        return []


# ----------------------------------------------------------------- listening


def test_the_lab_listens_only_on_this_machine_unless_told_otherwise():
    import run

    assert run.build_parser().parse_args([]).host == "127.0.0.1"
    assert run.build_parser().parse_args(["--host", "0.0.0.0"]).host == "0.0.0.0"


# ---------------------------------------------------------------- path rule


def test_a_folder_outside_the_lab_is_refused_over_the_api(studio, tmp_path):
    import cv2

    client, module = studio
    outside = tmp_path / "shots"
    outside.mkdir()
    cv2.imwrite(str(outside / "a.jpg"), np.full((20, 20, 3), 30, np.uint8))

    refused = client.post("/api/datasets/local", json={"path": str(outside)})
    assert refused.status_code == 403 and "EASYDETECT_LAB_ALLOW_PATHS" in refused.json()["error"]
    assert not (outside / "data.yaml").exists()
    # a path that is not there is refused the same way: no probing for what exists
    assert client.post("/api/datasets/local",
                       json={"path": str(tmp_path / "nowhere")}).status_code == 403

    inside = module.DATA / "shots"
    inside.mkdir()
    cv2.imwrite(str(inside / "a.jpg"), np.full((20, 20, 3), 30, np.uint8))
    assert client.post("/api/datasets/local", json={"path": str(inside)}).status_code == 200


def test_the_server_can_allow_any_path_or_named_folders(studio, tmp_path, monkeypatch):
    import cv2

    client, _ = studio
    for name in ("shots", "other"):
        (tmp_path / name).mkdir()
        cv2.imwrite(str(tmp_path / name / "a.jpg"), np.full((20, 20, 3), 30, np.uint8))

    monkeypatch.setenv("EASYDETECT_LAB_ALLOW_PATHS", str(tmp_path / "shots"))
    assert client.post("/api/datasets/local",
                       json={"path": str(tmp_path / "shots")}).status_code == 200
    assert client.post("/api/datasets/local",
                       json={"path": str(tmp_path / "other")}).status_code == 403

    monkeypatch.setenv("EASYDETECT_LAB_ALLOW_PATHS", "1")
    assert client.post("/api/datasets/local",
                       json={"path": str(tmp_path / "other")}).status_code == 200


def test_label_mode_registers_any_folder(studio, tmp_path):
    """``run.py label --source`` is typed at the machine: the API's rule is not for it."""
    import cv2

    _, module = studio
    folder = tmp_path / "shots"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((20, 20, 3), 30, np.uint8))
    dataset = module.register_folder({"path": str(folder), "names": ["can"]})
    assert dataset["images"] == 1


def test_batch_inference_runs_only_on_folders_the_lab_knows(studio, tmp_path):
    client, module = studio
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "a.jpg").write_bytes(image_bytes())
    refused = client.post("/api/predict", data={"model": "ed-s", "path": str(outside)})
    assert refused.status_code == 403
    assert module.db.query("SELECT id FROM jobs") == []

    upload(client, {"images/a.jpg": image_bytes()})
    registered = client.get("/api/datasets/1").json()["images_dir"]
    queued = client.post("/api/predict", data={"model": "ed-s", "path": registered})
    assert queued.status_code == 200


def test_a_model_given_as_a_path_must_be_inside_the_lab(studio, tmp_path, monkeypatch):
    """A .pt is a pickle: loading one from anywhere would run anyone's code."""
    client, module = studio
    loaded = []
    monkeypatch.setattr(module, "_model", lambda name, task="detect": loaded.append(name))
    planted = tmp_path / "evil.pt"
    planted.write_bytes(b"not weights")
    frame = {"image": ("f.jpg", image_bytes(), "image/jpeg")}

    refused = client.post("/api/preview", data={"model": str(planted)}, files=frame)
    assert refused.status_code == 403 and loaded == []
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    assert client.post("/api/jobs", json={"dataset_id": 1,
                                          "model": str(planted)}).status_code == 403
    assert client.post("/api/datasets/1/autolabel",
                       json={"model": str(planted)}).status_code == 403
    assert client.post("/api/predict", data={"model": str(planted),
                                             "dataset_id": "1"}).status_code == 403

    # mirror names, registered models and files the lab holds are all fine
    assert module._resolve_model("ed-m") == "ed-m"
    kept = module.DATA / "models" / "mine" / "best.pt"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"x")
    assert module._resolve_model(str(kept)) == str(kept)
    module.db.add_model(name="theirs", path=str(planted), kind="torch", classes=[])
    assert module._resolve_model("theirs") == str(planted)   # registered on purpose


# ------------------------------------------------------- off the event loop


def test_inference_and_uploads_run_off_the_event_loop(studio):
    _, module = studio
    for endpoint in (module.predict, module.preview, module.upload_dataset,
                     module.upload_images, module.upload_video, module.batch_predict,
                     module.upload_model):
        assert not inspect.iscoroutinefunction(endpoint), endpoint.__name__


def test_a_slow_frame_does_not_hold_up_other_requests(studio, monkeypatch):
    import httpx

    _, module = studio
    started = threading.Event()

    class Slow:
        def __call__(self, frame, **kwargs):
            started.set()
            time.sleep(0.6)
            return [FakeResult(frame)]

    monkeypatch.setattr(module, "_model", lambda name, task="detect": Slow())

    async def go():
        transport = httpx.ASGITransport(app=module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            frame = client.post("/api/preview", data={"model": "ed-s"},
                                files={"image": ("f.jpg", image_bytes(), "image/jpeg")})
            pending = asyncio.ensure_future(frame)
            assert await asyncio.to_thread(started.wait, 5)
            begun = time.perf_counter()
            listed = await client.get("/api/datasets")
            took = time.perf_counter() - begun
            return took, listed.status_code, (await pending).status_code

    took, listed, framed = asyncio.run(go())
    assert listed == 200 and framed == 200
    assert took < 0.4          # answered while the frame was still being worked on


def test_uploads_are_written_to_disk_in_pieces(studio, monkeypatch):
    """Never read whole into memory: a 2 GB video must not need 2 GB of RAM."""
    client, module = studio
    reads = []
    real = module.shutil.copyfileobj

    def copy(source, target, length=0):
        reads.append(length)
        return real(source, target, length)

    monkeypatch.setattr(module.shutil, "copyfileobj", copy)
    big = image_bytes() + b"\0" * (3 << 20)          # a jpeg with 3 MB of trailing bytes
    response = client.post("/api/datasets/images", data={"name": "big"},
                           files=[("files", ("a.jpg", big, "image/jpeg"))])
    assert response.status_code == 200
    assert reads == [1 << 20]
    stored = Path(client.get("/api/datasets/1").json()["images_dir"]) / "a.jpg"
    assert stored.read_bytes() == big


def test_a_video_upload_still_becomes_frames(studio):
    import cv2

    client, module = studio
    path = module.DATA / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 32))
    for i in range(6):
        writer.write(np.full((32, 48, 3), 20 * i, np.uint8))
    writer.release()
    response = client.post("/api/datasets/video", data={"name": "clip", "every": "2"},
                           files={"video": ("clip.avi", path.read_bytes(), "video/x-msvideo")})
    assert response.status_code == 200, response.text
    assert response.json()["images"] == 3


# --------------------------------------------------------------- model cache


@pytest.fixture
def built(monkeypatch):
    """Detector replaced by a stand-in that records what was loaded."""
    import easydetect

    log = []

    class Detector:
        def __init__(self, name, verbose=True, task="detect"):
            log.append((name, task))

    monkeypatch.setattr(easydetect, "Detector", Detector)
    return log


def test_the_model_cache_keeps_only_the_last_few(studio, built):
    _, module = studio
    module._models.clear()
    names = ["ed-n", "ed-s", "ed-m", "ed-l", "ed-x"]
    for name in names:
        module._model(name)
    assert len(module._models) == module.MODEL_CACHE_SIZE == 4
    module._model("ed-x")                      # still there: no new load
    assert len(built) == 5
    module._model("ed-n")                      # the least recently used went first
    assert built[-1] == ("ed-n", "detect") and len(built) == 6
    module._models.clear()


def test_a_re_exported_model_is_loaded_again(studio, built):
    _, module = studio
    module._models.clear()
    folder = module.RUNS / "job1" / "openvino"
    folder.mkdir(parents=True)
    xml, weights = folder / "best.xml", folder / "best.bin"
    xml.write_text("<net/>")
    weights.write_bytes(b"1")
    first = module._model(str(xml))
    assert module._model(str(xml)) is first and len(built) == 1

    later = time.time() + 5                       # only the .bin changed: still a new model
    os.utime(weights, (later, later))
    second = module._model(str(xml))
    assert second is not first and len(built) == 2
    assert len(module._models) == 1               # the old one is not kept alongside
    module._models.clear()


def test_threads_asking_for_one_model_load_it_once(studio, monkeypatch):
    import easydetect

    _, module = studio
    module._models.clear()
    built = []

    class Detector:
        def __init__(self, name, verbose=True, task="detect"):
            time.sleep(0.05)
            built.append(name)

    monkeypatch.setattr(easydetect, "Detector", Detector)
    got = []
    threads = [threading.Thread(target=lambda: got.append(module._model("ed-s")))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built == ["ed-s"] and len({id(m) for m in got}) == 1
    module._models.clear()


# ---------------------------------------------------------------- image list


@pytest.fixture
def counted(studio, monkeypatch):
    """The studio, counting every full walk of a dataset's folder."""
    client, module = studio
    walks = []
    real = module._scan_images

    def scan(root):
        walks.append(root)
        return real(root)

    monkeypatch.setattr(module, "_scan_images", scan)
    return client, module, walks


def test_the_image_list_is_walked_once_not_per_request(counted):
    client, module, walks = counted
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes(60),
                    "images/c.jpg": image_bytes(90)})
    settle(Path(client.get("/api/datasets/1").json()["path"]))
    walks.clear()

    client.get("/api/datasets/1")
    one = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    for index in range(3):
        assert client.get(f"/api/datasets/1/image/{index}").status_code == 200
        client.post(f"/api/datasets/1/labels/{index}", json={"boxes": one})
        assert client.get(f"/api/datasets/1/labels/{index}").json()["boxes"]
    assert len(walks) == 1


def test_pictures_copied_in_from_outside_are_noticed(counted):
    client, module, walks = counted
    upload(client, {"images/a.jpg": image_bytes(), "images/sub/b.jpg": image_bytes(60)})
    root = Path(client.get("/api/datasets/1").json()["images_dir"])
    settle(root)
    names = [f["name"] for f in client.get("/api/datasets/1").json()["files"]]
    assert names == ["a.jpg", "sub/b.jpg"]

    (root / "sub" / "0.jpg").write_bytes(image_bytes(10))   # one level down
    settle(root, age=30)                                     # an older, but different, mtime
    names = [f["name"] for f in client.get("/api/datasets/1").json()["files"]]
    assert names == ["a.jpg", "sub/0.jpg", "sub/b.jpg"]


def test_what_the_lab_adds_or_removes_shows_at_once(counted, monkeypatch):
    """Even when the folder's mtime would not tell: the app forgets the list itself."""
    client, module, _ = counted
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes(60)})
    settle(Path(client.get("/api/datasets/1").json()["path"]))
    client.get("/api/datasets/1")
    monkeypatch.setattr(module, "_folders_unchanged", lambda stamps: True)

    client.post("/api/datasets/images", data={"name": "x", "dataset_id": "1"},
                files=[("files", ("0.jpg", image_bytes(5), "image/jpeg"))])
    assert [f["name"] for f in client.get("/api/datasets/1").json()["files"]] == \
        ["0.jpg", "a.jpg", "b.jpg"]
    assert client.request("DELETE", "/api/datasets/1/images/0").json()["images"] == 2
    assert [f["name"] for f in client.get("/api/datasets/1").json()["files"]] == ["a.jpg", "b.jpg"]

    # a deleted dataset's id can come back for the next one: never its old list
    client.request("DELETE", "/api/datasets/1")
    upload(client, {"images/z.jpg": image_bytes()})
    assert [f["name"] for f in client.get("/api/datasets/1").json()["files"]] == ["z.jpg"]


def test_a_picture_named_by_the_page_must_still_be_at_its_index(studio):
    """A list that changed under an open labelling page: refuse, never label the wrong one."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes(60)})
    one = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    saved = client.post("/api/datasets/1/labels/0?name=a.jpg", json={"boxes": one})
    assert saved.status_code == 200

    root = Path(client.get("/api/datasets/1").json()["images_dir"])
    (root / "0.jpg").write_bytes(image_bytes(5))         # now index 0 is someone else
    module._forget_images(1)
    stale = client.post("/api/datasets/1/labels/0?name=a.jpg", json={"boxes": one * 2})
    assert stale.status_code == 409
    assert client.get("/api/datasets/1/image/0?name=a.jpg").status_code == 409
    assert client.request("DELETE", "/api/datasets/1/images/0?name=a.jpg").status_code == 409
    assert (root / "0.jpg").exists()
    assert len(client.get("/api/datasets/1/labels/1?name=a.jpg").json()["boxes"]) == 1


def test_the_walk_lists_what_list_images_lists(studio, tmp_path):
    from labeling import list_images

    _, module = studio
    for name in ("b.JPG", "a/c.png", "a/d.txt", "a/b/e.webp", ".hidden/f.jpg", "g.jpeg"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    images, folders = module._scan_images(tmp_path)
    assert images == list_images(tmp_path)
    assert str(tmp_path / "a" / "b") in folders


# --------------------------------------------------------------- the stream


def test_the_progress_stream_waits_without_holding_a_thread(studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 2}).json()["id"]
    module.db.update_job(job, status="running", progress=0.5, detail="에폭 1/2")
    monkeypatch.setattr(module, "STREAM_POLL", 0.01)

    response = module.stream_job(job)
    # an async generator of its own, not a sync one Starlette walks in a thread
    assert response.body_iterator.ag_code.co_name == "events"

    polls = []
    real = module.db.one

    def one(sql, params=()):
        if sql.startswith("SELECT * FROM jobs WHERE id"):
            polls.append(1)
            if len(polls) == 3:
                module.db.update_job(job, status="done", progress=1.0, detail=None)
        return real(sql, params)

    monkeypatch.setattr(module.db, "one", one)
    body = client.get(f"/api/jobs/{job}/stream").text
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert [e.get("status") for e in events if e["type"] == "progress"] == ["running", "done"]
    assert events[-1] == {"type": "end", "status": "done"}


# -------------------------------------------------------------------- worker


def make_db(tmp_path):
    from db import Database

    return Database(tmp_path / "lab.db")


def test_the_worker_hands_memory_back_after_every_job(tmp_path, monkeypatch):
    import worker as worker_module

    db = make_db(tmp_path)
    db.add_dataset(name="d", path=str(tmp_path), images_dir=str(tmp_path),
                   labels_dir=str(tmp_path), images=0, labelled=0, classes=[])
    job = db.add_job(kind="train", dataset_id=1, model="ed-s")
    w = worker_module.Worker(db, tmp_path / "runs")

    def boom(_job):
        w.stop()
        raise RuntimeError("out of memory")

    monkeypatch.setattr(w, "_train", boom)
    freed = []
    monkeypatch.setattr(worker_module.gc, "collect", lambda: freed.append("gc"))
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: freed.append("cuda"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    w.run()                                   # one job, then the stop it asked for
    assert freed == ["gc", "cuda"]
    assert db.one("SELECT status FROM jobs WHERE id = ?", (job,))["status"] == "failed"


def test_a_job_device_reads_as_torch_takes_it():
    from worker import _torch_device

    assert _torch_device(None) is None and _torch_device("") is None
    assert _torch_device("0") == "cuda:0" and _torch_device("cpu") == "cpu"
    assert _torch_device("cuda:1") == "cuda:1"


@pytest.fixture
def scored(tmp_path, monkeypatch):
    """A trained run on a three-picture dataset, with a stand-in network and scorer."""
    import cv2
    import easydetect
    import easydetect.validator
    import torch

    root = tmp_path / "set"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for name in ("a", "b", "c"):
        cv2.imwrite(str(root / "images" / f"{name}.jpg"), np.full((32, 48, 3), 90, np.uint8))
        (root / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(root), "train": "images", "val": "images", "names": {0: "can"}}))

    class Net(torch.nn.Module):
        def forward(self, imgs):
            n = imgs.shape[0]
            logits = torch.full((n, 4, 1), -6.0)
            logits[:, 0, 0] = 6.0                       # one confident box per picture
            boxes = torch.tensor([0.5, 0.5, 0.4, 0.4]).repeat(n, 4, 1)
            return {"pred_logits": logits, "pred_boxes": boxes}

    calls = {"predict": 0}

    class Detector:
        def __init__(self, name, verbose=True, task="detect"):
            self.net, self.names = Net(), {0: "can"}

        def predict(self, source, **kwargs):
            from easydetect.results import Boxes

            calls["predict"] += 1
            return [SimpleNamespace(boxes=Boxes(np.zeros((0, 6), np.float32), (32, 48)),
                                    names=self.names)]

    def validate(net, data_yaml, imgsz=640, batch=8, conf=0.01, device=None, workers=2):
        from easydetect.data.dataset import DetDataset

        calls.update(batch=batch, device=device, workers=workers)
        if calls.get("watch_nothing"):
            return {"map50": 0.5, "map": 0.25, "per_class": {0: 0.25}}
        count = len(DetDataset(data_yaml, "val", imgsz, augment=False))
        for start in range(0, count, batch):
            net(torch.zeros(min(batch, count - start), 3, imgsz, imgsz))
        return {"map50": 0.5, "map": 0.25, "per_class": {0: 0.25}}

    monkeypatch.setattr(easydetect, "Detector", Detector)
    monkeypatch.setattr(easydetect.validator, "validate_torch", validate)

    from worker import Worker

    db = make_db(tmp_path)
    db.add_dataset(name="set", path=str(root), images_dir=str(root / "images"),
                   labels_dir=str(root / "labels"), images=3, labelled=3, classes=["can"])
    run = tmp_path / "runs" / "job1"
    (run / "weights").mkdir(parents=True)
    parent = db.add_job(kind="train", dataset_id=1, model="ed-n", imgsz=64,
                        status="done", run_dir=str(run))
    return SimpleNamespace(db=db, worker=Worker(db, tmp_path / "runs"), run=run,
                           parent=parent, calls=calls)


def test_evaluation_draws_from_its_one_scoring_pass(scored):
    from worker import EVAL_BATCH, _loader_workers

    db = scored.db
    job = db.add_job(kind="evaluate", dataset_id=1, model="ed-n", resume_of=scored.parent,
                     conf=0.5)
    scored.worker._evaluate(db.one("SELECT * FROM jobs WHERE id = ?", (job,)))

    assert scored.calls["predict"] == 0                  # no second pass per picture
    assert scored.calls["batch"] == EVAL_BATCH == 8
    assert scored.calls["device"] is None                # the run's own: CUDA when there is one
    assert scored.calls["workers"] == min(_loader_workers(), EVAL_BATCH)
    report = json.loads((scored.run / "eval" / "report.json").read_text())
    assert list(report) == ["map50", "map50_95", "conf", "per_class", "images"]
    assert report["per_class"] == [{"name": "can", "ap50_95": 0.25, "boxes": 3}]
    assert report["images"] == [{"file": f"{i:04d}_{n}.jpg", "truth": 1, "found": 1}
                                for i, n in enumerate("abc")]
    assert all((scored.run / "eval" / card["file"]).exists() for card in report["images"])
    row = db.one("SELECT status, best_map FROM jobs WHERE id = ?", (job,))
    assert row == {"status": "done", "best_map": 0.25}


def test_evaluation_scores_where_the_run_trained(scored):
    import torch

    db = scored.db
    db.update_job(scored.parent, device="0")
    job = db.add_job(kind="evaluate", dataset_id=1, model="ed-n", resume_of=scored.parent,
                     conf=0.5)
    scored.worker._evaluate(db.one("SELECT * FROM jobs WHERE id = ?", (job,)))
    assert scored.calls["device"] == ("cuda:0" if torch.cuda.is_available() else "cpu")


def test_evaluation_predicts_again_when_the_pass_cannot_be_watched(scored):
    scored.calls["watch_nothing"] = True
    db = scored.db
    job = db.add_job(kind="evaluate", dataset_id=1, model="ed-n", resume_of=scored.parent,
                     conf=0.5)
    scored.worker._evaluate(db.one("SELECT * FROM jobs WHERE id = ?", (job,)))
    assert scored.calls["predict"] == 3
    report = json.loads((scored.run / "eval" / "report.json").read_text())
    assert [card["found"] for card in report["images"]] == [0, 0, 0]


# ---------------------------------------------------------------------- INT8


class Exporting:
    """A model whose export writes what easydetect's would: best.* or best_int8.*."""

    def __init__(self, fail_int8=False):
        self.calls, self.fail_int8 = [], fail_int8

    def export(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("int8") and self.fail_int8:
            raise ImportError("No module named 'nncf'")
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        stem = "best_int8" if kwargs.get("int8") else "best"
        for suffix in (".xml", ".bin", ".onnx"):
            (out / f"{stem}{suffix}").write_bytes(b"x")
        return out / f"{stem}.xml"


def test_int8_is_exported_beside_fp32_only_when_asked(tmp_path):
    from worker import Worker

    worker = Worker(make_db(tmp_path), tmp_path / "runs")
    plain = Exporting()
    assert worker._export(plain, tmp_path / "plain", 1) is None
    assert [c.get("int8", False) for c in plain.calls] == [False]

    both = Exporting()
    data = tmp_path / "set" / "data.yaml"
    assert worker._export(both, tmp_path / "both", 1, int8=True, data=data) is None
    assert both.calls[1]["int8"] is True and both.calls[1]["data"] == str(data)
    out = tmp_path / "both" / "openvino"
    assert sorted(p.name for p in out.iterdir()) == \
        ["best.bin", "best.onnx", "best.xml", "best_int8.bin", "best_int8.xml"]

    failing = Exporting(fail_int8=True)
    note = worker._export(failing, tmp_path / "failing", 1, int8=True, data=data)
    assert note.startswith("INT8 export failed")
    assert (tmp_path / "failing" / "openvino" / "best.xml").exists()


def test_the_training_form_asks_for_int8_and_the_job_keeps_it(studio):
    client, module = studio
    page = client.get("/").text
    assert 'id="job-int8"' in page and "int8: $(\"job-int8\").checked" in page
    i18n = client.get("/static/js/i18n.js").text
    assert '"INT8도 만들기 (CPU에서 약 2배 빠름, 정확도 확인 필요)"' in i18n

    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    query = "SELECT int8 FROM jobs WHERE id = ?"
    plain = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    asked = client.post("/api/jobs", json={"dataset_id": 1, "int8": True}).json()["id"]
    assert module.db.one(query, (plain,)) == {"int8": None}
    assert module.db.one(query, (asked,)) == {"int8": 1}

    (module.RUNS / f"job{asked}" / "weights").mkdir(parents=True)
    (module.RUNS / f"job{asked}" / "weights" / "last.pt").write_bytes(b"x")
    module.db.update_job(asked, status="done")
    more = client.post(f"/api/jobs/{asked}/resume", json={"add_epochs": 2}).json()["id"]
    assert module.db.one(query, (more,)) == {"int8": 1}


def test_the_int8_ir_travels_in_the_ir_zip_but_is_never_the_default(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "int8": True}).json()["id"]
    run = module.RUNS / f"job{job}"
    Exporting().export(out_dir=run / "openvino")
    Exporting().export(out_dir=run / "openvino", int8=True)
    (run / "openvino" / "best_int8.onnx").unlink()
    (run / "openvino" / "labels.txt").write_text("can\n")
    module.db.update_job(job, status="done", run_dir=str(run))

    ir = client.get(f"/api/jobs/{job}/download/openvino")
    names = zipfile.ZipFile(io.BytesIO(ir.content)).namelist()
    assert {"best.xml", "best.bin", "best_int8.xml", "best_int8.bin", "labels.txt"} <= set(names)
    onnx = client.get(f"/api/jobs/{job}/download/onnx")
    assert "best_int8.xml" not in zipfile.ZipFile(io.BytesIO(onnx.content)).namelist()

    files = client.get(f"/api/jobs/{job}").json()["files"]
    assert files["ir"].endswith("best.xml") and files["ir_int8"].endswith("best_int8.xml")
    model = client.post("/api/models", json={"job_id": job, "name": "v1"}).json()["id"]
    path = module.db.one("SELECT path FROM models WHERE id = ?", (model,))["path"]
    assert path.endswith("best.xml")


def test_the_full_precision_ir_is_picked_whatever_the_names(studio, tmp_path):
    _, module = studio
    (tmp_path / "openvino").mkdir()
    for name in ("alpha_int8.xml", "beta.xml"):
        (tmp_path / "openvino" / name).write_text("<net/>")
    assert module._main_ir(tmp_path).name == "beta.xml"
    assert module._main_ir(tmp_path / "nothing") is None
