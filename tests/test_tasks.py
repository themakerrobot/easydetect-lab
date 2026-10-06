# Apache-2.0
"""Outlines and keypoints: label files that keep them, the page's API for
choosing what a dataset is labelled for, and training that learns them."""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest
import yaml

from labeling import dataset_task, flip_pairs, predict_boxes, read_labels, write_labels

OUTLINE = {"cls": 1, "points": [[0.1, 0.2], [0.5, 0.2], [0.3, 0.6]]}
KPTS = [[0.2, 0.3, 2], [0.0, 0.0, 0], [0.4, 0.5, 1]]


def test_an_outline_round_trips_and_its_box_is_the_outlines(tmp_path):
    path = tmp_path / "a.txt"
    write_labels(path, [OUTLINE, {"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
    lines = path.read_text().splitlines()
    assert lines[0].split()[0] == "1" and len(lines[0].split()) == 7
    first, second = read_labels(path)
    assert first["points"] == OUTLINE["points"]
    assert (first["cx"], first["cy"]) == pytest.approx((0.3, 0.4))
    assert (first["w"], first["h"]) == pytest.approx((0.4, 0.4))
    assert "points" not in second                   # a box stays a box


def test_keypoints_round_trip_and_every_line_has_all_of_them(tmp_path):
    path = tmp_path / "a.txt"
    box = {"cls": 0, "cx": 0.3, "cy": 0.4, "w": 0.2, "h": 0.3}
    write_labels(path, [dict(box, kpts=KPTS), box], kpt_shape=[3, 3])
    full, bare = path.read_text().splitlines()
    assert len(full.split()) == len(bare.split()) == 5 + 3 * 3
    rows = read_labels(path, kpt_shape=[3, 3])
    assert rows[0]["kpts"] == KPTS
    assert rows[1]["kpts"] == [[0.0, 0.0, 0]] * 3   # none placed yet
    # an unlabelled point is written as zeros, wherever it was left
    write_labels(path, [dict(box, kpts=[[0.7, 0.7, 0], *KPTS[1:]])], kpt_shape=[3, 3])
    assert read_labels(path, [3, 3])[0]["kpts"][0] == [0.0, 0.0, 0]


def test_a_two_column_keypoint_set_reads_placed_points_as_visible(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.2 0.2 0.4 0.4 0 0\n")
    assert read_labels(path, [2, 2])[0]["kpts"] == [[0.4, 0.4, 2], [0.0, 0.0, 0]]


def test_left_and_right_find_each_other():
    assert flip_pairs(["nose", "left_eye", "right_eye"]) == [0, 2, 1]
    assert flip_pairs(["l_hand", "r_hand", "head"]) == [1, 0, 2]
    assert flip_pairs(["왼손", "오른손"]) == [1, 0]
    assert flip_pairs(["tip", "base"]) == [0, 1]


def test_what_a_dataset_is_labelled_for(tmp_path):
    outline = tmp_path / "o.txt"
    outline.write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n")
    box = tmp_path / "b.txt"
    box.write_text("0 0.5 0.5 0.2 0.2\n")
    assert dataset_task({"kpt_shape": [3, 3]}, [outline]) == "pose"
    assert dataset_task({"task": "detect"}, [outline]) == "detect"
    assert dataset_task({}, [box, outline]) == "segment"
    assert dataset_task({}, [box]) == "detect"


def test_auto_labelling_keeps_outlines_and_matches_keypoints_by_name():
    class Boxes:
        cls, conf = [0.0], [0.9]
        xywhn = [(0.5, 0.5, 0.5, 0.5)]

        def __len__(self):
            return 1

    class Masks:
        xy = [np.array([[10, 10], [30, 10], [30, 30], [10, 30]], np.float32)]

    class Keypoints:
        names = ("nose", "left_eye", "right_eye")
        data = np.array([[[20, 12, 0.9], [18, 10, 0.2], [22, 10, 0.8]]], np.float32)

    class Result:
        boxes, masks, keypoints = Boxes(), Masks(), Keypoints()
        orig_shape = (40, 40)

        def name_of(self, index):
            return "person"

    class Model:
        def predict(self, *args, **kwargs):
            return [Result()]

    (row,) = predict_boxes(Model(), "x.jpg", ["person"], kpt_names=["right_eye", "nose", "tail"])
    assert len(row["points"]) == 4 and row["points"][0] == [0.25, 0.25]
    assert row["kpts"] == [[0.55, 0.25, 2], [0.5, 0.3, 2], [0.0, 0.0, 0]]


# ------------------------------------------------------------------- the API

from test_app import image_bytes, upload  # noqa: E402


def _yaml(module, dataset_id=1):
    row = module.db.one("SELECT path FROM datasets WHERE id = ?", (dataset_id,))
    from pathlib import Path

    return yaml.safe_load((Path(row["path"]) / "data.yaml").read_text())


def test_choosing_keypoints_writes_the_data_yaml_and_keeps_the_boxes(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    assert client.get("/api/datasets/1").json()["task"] == "detect"

    body = client.post("/api/datasets/1/task", json={
        "task": "pose", "skeleton": [["0", "1"]],
        "keypoints": [{"name": "left_ear"}, {"name": "right_ear"}, {"name": "tail"}]}).json()
    assert body["task"] == "pose" and body["files"] == 1
    assert body["keypoints"] == {"names": ["left_ear", "right_ear", "tail"],
                                 "skeleton": [[0, 1]], "flip": [1, 0, 2]}
    cfg = _yaml(module)
    assert cfg["kpt_shape"] == [3, 3] and cfg["flip_idx"] == [1, 0, 2]
    from easydetect.data.dataset import load_data_yaml
    from easydetect.pose import KeypointSpec

    spec = KeypointSpec.from_data(load_data_yaml(module.DATASETS / next(
        p.name for p in module.DATASETS.iterdir()) / "data.yaml"))
    assert spec.names == ("left_ear", "right_ear", "tail")

    (box,) = client.get("/api/datasets/1/labels/0").json()["boxes"]
    assert box["cx"] == 0.5 and box["kpts"] == [[0.0, 0.0, 0]] * 3
    box["kpts"][2] = [0.6, 0.6, 2]
    client.post("/api/datasets/1/labels/0", json={"boxes": [box]})
    assert client.get("/api/datasets/1/labels/0").json()["boxes"][0]["kpts"][2] == [0.6, 0.6, 2]

    stats = client.get("/api/datasets/1/stats").json()
    assert stats["task"] == "pose" and stats["boxes_without_keypoints"] == 0
    assert [k["placed"] for k in stats["per_keypoint"]] == [0, 0, 1]


def test_renaming_reordering_and_dropping_keypoints_carries_the_points(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(),
                    "labels/a.txt": b"0 .5 .5 .2 .2 .1 .1 2 .2 .2 2 .3 .3 1\n",
                    "data.yaml": yaml.safe_dump({"train": "images", "val": "images",
                                                 "names": {0: "cat"}, "kpt_shape": [3, 3],
                                                 "kpt_names": {0: ["a", "b", "c"]}}).encode()})
    plan = client.post("/api/datasets/1/task", json={
        "task": "pose", "dry_run": True,
        "keypoints": [{"name": "c", "was": 2}, {"name": "A", "was": 0}, {"name": "new"}]}).json()
    assert plan["files"] == 1 and plan["keypoints_kept"] == 2
    assert client.get("/api/datasets/1/labels/0").json()["boxes"][0]["kpts"][1] == [0.2, 0.2, 2]

    client.post("/api/datasets/1/task", json={
        "task": "pose",
        "keypoints": [{"name": "c", "was": 2}, {"name": "A", "was": 0}, {"name": "new"}]})
    (box,) = client.get("/api/datasets/1/labels/0").json()["boxes"]
    assert box["kpts"] == [[0.3, 0.3, 1], [0.1, 0.1, 2], [0.0, 0.0, 0]]
    assert _yaml(module)["kpt_names"] == {0: ["c", "A", "new"]}

    # back to boxes: the keypoints go, the box stays, and nothing reads as an outline
    client.post("/api/datasets/1/task", json={"task": "detect"})
    (box,) = client.get("/api/datasets/1/labels/0").json()["boxes"]
    assert "kpts" not in box and "points" not in box and box["w"] == 0.2
    assert "kpt_shape" not in _yaml(module)
    assert client.get("/api/datasets/1").json()["task"] == "detect"


def test_bad_keypoint_lists_are_refused(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes()})
    for keypoints in ([], [{"name": "a"}, {"name": "a"}], [{"name": ""}]):
        res = client.post("/api/datasets/1/task", json={"task": "pose", "keypoints": keypoints})
        assert res.status_code == 400
    assert client.post("/api/datasets/1/task", json={"task": "mask"}).status_code == 400


def test_an_outline_dataset_is_recognised_and_its_outlines_kept(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(),
                    "labels/a.txt": b"0 0.1 0.1 0.5 0.1 0.3 0.6\n"})
    assert client.get("/api/datasets/1").json()["task"] == "segment"
    assert [d["task"] for d in client.get("/api/datasets").json()] == ["segment"]
    (obj,) = client.get("/api/datasets/1/labels/0").json()["boxes"]
    assert obj["points"] == [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]]
    obj["points"].append([0.1, 0.5])
    client.post("/api/datasets/1/labels/0", json={"boxes": [obj, {"cls": 0, "cx": .5, "cy": .5,
                                                                   "w": .1, "h": .1}]})
    stats = client.get("/api/datasets/1/stats").json()
    assert stats["outlines"] == 1 and stats["boxes"] == 2

    # an explicit choice sticks, whatever the labels hold
    client.post("/api/datasets/1/task", json={"task": "detect"})
    assert client.get("/api/datasets/1").json()["task"] == "detect"
    assert _yaml(module)["task"] == "detect"


def test_the_outline_button_asks_mobile_sam(studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes()})       # 48 x 32

    class Segmenter:
        def __call__(self, img, xyxy):
            masks = np.zeros((len(xyxy), *img.shape[:2]), bool)
            for m, (x0, y0, x1, y1) in zip(masks, xyxy.astype(int), strict=True):
                m[y0:y1, x0:x1] = True
            return masks, np.ones(len(xyxy), np.float32)

    monkeypatch.setattr(module, "_segmenter", lambda dataset_id: Segmenter())
    res = client.post("/api/datasets/1/outline/0",
                      json={"boxes": [{"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}]}).json()
    (points,) = res["outlines"]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    assert min(xs) == pytest.approx(0.25) and min(ys) == pytest.approx(0.25)
    assert client.post("/api/datasets/1/outline/5", json={"boxes": []}).status_code == 404


def test_an_export_and_a_merge_keep_the_keypoints(studio):
    client, module = studio
    cfg = {"train": "images", "val": "images", "names": {0: "fish"}, "kpt_shape": [2, 3],
           "kpt_names": {0: ["head", "tail"]}}
    for name in ("a", "b"):
        upload(client, {"images/a.jpg": image_bytes(), "data.yaml": yaml.safe_dump(cfg).encode(),
                        "labels/a.txt": b"0 .5 .5 .2 .2 .1 .1 2 .2 .2 2\n"}, name=name)
    with zipfile.ZipFile(io.BytesIO(client.get("/api/datasets/1/export").content)) as zf:
        cfg = yaml.safe_load(zf.read("data.yaml"))
        label = zf.read("labels/a.txt").decode()
    assert cfg["kpt_shape"] == [2, 3] and cfg["kpt_names"] == {0: ["head", "tail"]}
    assert len(label.split()) == 5 + 2 * 3

    merged = client.post("/api/datasets/merge", json={"ids": [1, 2], "name": "both"}).json()
    body = client.get(f"/api/datasets/{merged['id']}").json()
    assert body["task"] == "pose" and body["keypoints"]["names"] == ["head", "tail"]
    rows = client.get(f"/api/datasets/{merged['id']}/labels/0").json()["boxes"]
    assert rows[0]["kpts"][0] == [0.1, 0.1, 2]

    client.post("/api/datasets/2/task", json={"task": "pose", "keypoints": [{"name": "x"}]})
    refused = client.post("/api/datasets/merge", json={"ids": [1, 2]})
    assert refused.status_code == 400 and "keypoints" in refused.json()["error"]


def test_training_reports_the_keypoint_and_outline_stages(studio, monkeypatch):
    """After the boxes the bar starts again, with the stage named, and the
    finished job says what the stage reached."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(),
                    "labels/a.txt": b"0 0.1 0.1 0.5 0.1 0.3 0.6\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    seen, asked = [], {}

    class Detector:
        def __init__(self, name, verbose=True, task="detect"):
            pass

        def train(self, **kwargs):
            asked.update(kwargs)
            run = module.RUNS / "job1"
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "best.pt").write_bytes(b"x")
            (run / "summary.json").write_text(json.dumps({"best_map50_95": 0.5}))
            (run / "segment").mkdir()
            (run / "segment" / "results.csv").write_text("epoch,loss,miou,seconds\n0,,0.61,0\n"
                                                         "1,0.3,0.72,1\n")
            for p in ({"phase": "masks", "stage": "encode", "step": 1, "steps": 2, "seconds": 1},
                      {"phase": "masks", "epoch": 1, "epochs": 2, "step": 3, "steps": 4,
                       "seconds": 3.0, "miou": 0.7}):
                kwargs["on_progress"](p)
                seen.append(module.db.one("SELECT detail, progress FROM jobs WHERE id = ?",
                                          (job,)))
            return run / "weights" / "best.pt"

        def export(self, **kwargs):
            pass

    import easydetect
    monkeypatch.setattr(easydetect, "Detector", Detector)
    module.worker._train(module.db.one("SELECT * FROM jobs WHERE id = ?", (job,)))
    assert asked["seg"] is True
    assert "준비" in seen[0]["detail"] and seen[0]["progress"] == 0.5
    assert "에폭 1/2" in seen[1]["detail"] and "mIoU 0.700" in seen[1]["detail"]
    assert seen[1]["progress"] == pytest.approx(0.375)
    done = module.db.one("SELECT status, detail FROM jobs WHERE id = ?", (job,))
    assert done["status"] == "done" and "윤곽 mIoU 0.720 (학습 전 0.610)" in done["detail"]


def test_a_box_dataset_with_a_stray_outline_trains_boxes_only(studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(),
                    "labels/a.txt": b"0 0.1 0.1 0.5 0.1 0.3 0.6\n"})
    client.post("/api/datasets/1/task", json={"task": "detect"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    asked = {}

    class Detector:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, **kwargs):
            asked.update(kwargs)
            raise RuntimeError("stop here")

    import easydetect
    monkeypatch.setattr(easydetect, "Detector", Detector)
    with pytest.raises(RuntimeError):
        module.worker._train(module.db.one("SELECT * FROM jobs WHERE id = ?", (job,)))
    assert asked["seg"] is False


def test_a_trained_keypoint_run_predicts_keypoints(studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    run = module.RUNS / f"job{job}"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (run / "weights" / "pose.onnx").write_bytes(b"x")
    module.db.update_job(job, status="done", run_dir=str(run), finished=time.time())
    assert module._run_task(run) == "pose"
    tasks = []

    class Result:
        def plot(self):
            return np.zeros((8, 8, 3), np.uint8)

        def summary(self):
            return []

    def model(name, task="detect"):
        tasks.append(task)
        return lambda *a, **k: [Result()]

    monkeypatch.setattr(module, "_model", model)
    res = client.post(f"/api/jobs/{job}/predict",
                      files={"image": ("a.jpg", image_bytes(), "image/jpeg")})
    assert res.status_code == 200 and tasks == ["pose"]
