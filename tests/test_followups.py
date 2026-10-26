# Apache-2.0
"""The second review's fixes: each test failed before its fix."""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from test_app import image_bytes, upload

from labeling import class_lookup, label_path

LAB = Path(__file__).resolve().parents[1]


# --------------------------------------------------------- auto-labelling a set


def _fake_detector(monkeypatch, found, during=None):
    """A Detector whose predictions are ``found[image stem]`` (class, box)."""
    import easydetect

    class Boxes:
        def __init__(self, rows):
            self.cls = [float(c) for c, _ in rows]
            self.conf = [0.9] * len(rows)
            self.xywhn = [b for _, b in rows]

        def __len__(self):
            return len(self.cls)

    class Result:
        masks = keypoints = None
        orig_shape = (32, 48)

        def __init__(self, rows):
            self.boxes = Boxes(rows)

        def name_of(self, c):
            return {0: "cup", 1: "dog"}[int(c)]

    class Detector:
        def __init__(self, *a, **kw):
            pass

        def predict(self, path, **kw):
            stem = Path(path).stem
            if during:
                during(stem)
            return [Result(found.get(stem, []))]

    monkeypatch.setattr(easydetect, "Detector", Detector)


def _autolabel(client, module):
    job_id = client.post("/api/datasets/1/autolabel", json={"model": "ed-n"}).json()["id"]
    module.worker._autolabel(module.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,)))
    return module.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))


def _images(module):
    row = module.db.one("SELECT images_dir FROM datasets WHERE id = 1")
    return {p.stem: p for p in Path(row["images_dir"]).iterdir()}


def test_auto_labelling_writes_no_empty_file_and_keeps_what_someone_saved_meanwhile(
        studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes(),
                    "images/c.jpg": image_bytes()})
    client.post("/api/datasets/1/classes", json={"names": ["컵"]})
    images = _images(module)

    def someone_saves_b(stem):
        if stem == "a":       # while a is predicted, a person saves b by hand
            label_path(images["b"]).parent.mkdir(parents=True, exist_ok=True)
            label_path(images["b"]).write_text("0 0.1 0.1 0.1 0.1\n")

    _fake_detector(monkeypatch, {"a": [(0, (0.5, 0.5, 0.2, 0.2))],
                                 "b": [(0, (0.9, 0.9, 0.1, 0.1))]}, someone_saves_b)
    job = _autolabel(client, module)
    assert job["status"] == "done" and job["progress"] == 1.0
    assert label_path(images["a"]).read_text().startswith("0 ")       # 컵 ← cup
    assert label_path(images["b"]).read_text() == "0 0.1 0.1 0.1 0.1\n"
    assert not label_path(images["c"]).exists()     # nothing found: still unlabelled
    assert job["detail"].startswith("1 of 3 images got boxes")


def test_progress_moves_on_pictures_where_nothing_was_found(studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes()})
    seen = []
    real = module.db.update_job

    def spy(job_id, **kw):
        if "progress" in kw:
            seen.append(kw["progress"])
        return real(job_id, **kw)

    monkeypatch.setattr(module.db, "update_job", spy)
    _fake_detector(monkeypatch, {})
    _autolabel(client, module)
    assert 0.5 in seen


def test_korean_class_names_take_the_model_s_english_ones():
    assert class_lookup({0: "cup", 1: "dog"}, ["강아지", "컵"]) == {"dog": 0, "cup": 1}
    assert class_lookup({0: "Cup"}, ["cup"]) == {"Cup": 0}          # any case
    assert class_lookup({0: "cup", 1: "cups"}, ["cup"]) == {"cup": 0}  # exact wins, once
    assert class_lookup(["dog"], ["cat"]) == {}


# ------------------------------------------------------- starting it twice


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(handler) -> tuple[int, http.server.HTTPServer]:
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1], server


class _Lab(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"running_job": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _Other(_Lab):
    def do_GET(self):  # noqa: N802
        self.send_response(404)
        self.end_headers()


def test_what_answers_on_the_port():
    import run

    assert run._what_answers("127.0.0.1", _free_port()) is None
    port, server = _serve(_Lab)
    assert run._what_answers("127.0.0.1", port) == "lab"
    server.shutdown()
    port, server = _serve(_Other)
    assert run._what_answers("127.0.0.1", port) == "other"
    server.shutdown()


def test_a_second_start_opens_the_running_lab_and_does_not_start_another(monkeypatch):
    import run

    port, server = _serve(_Lab)
    opened = []
    monkeypatch.setattr(run.webbrowser, "open", opened.append)
    monkeypatch.setattr(run, "_single_instance", lambda port: pytest.fail("started another"))
    assert run.main(["--port", str(port)]) == 0
    assert opened == [f"http://127.0.0.1:{port}/"]
    server.shutdown()


def test_a_port_another_program_holds_is_said_and_nothing_starts(monkeypatch, capsys):
    import run

    port, server = _serve(_Other)
    monkeypatch.setattr(run, "_single_instance", lambda port: pytest.fail("started"))
    assert run.main(["--port", str(port), "--no-browser"]) == 1
    assert "다른 프로그램" in capsys.readouterr().out
    server.shutdown()


def test_a_second_lab_on_the_same_data_folder_is_refused_and_points_at_the_first(
        tmp_path, monkeypatch, capsys):
    import run

    monkeypatch.setenv("EASYDETECT_LAB_HOME", str(tmp_path))
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.path.insert(0, sys.argv[1]); import run; "
         "assert run._single_instance(9123) is not None; print('held', flush=True); "
         "time.sleep(30)", str(LAB)],
        stdout=subprocess.PIPE, text=True, env={**__import__("os").environ})
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(run, "_LOCK", None)
        assert run._single_instance(_free_port()) is None
        opened = []
        monkeypatch.setattr(run.webbrowser, "open", opened.append)
        assert run.main(["--port", str(_free_port())]) == 0
        assert opened == ["http://127.0.0.1:9123/"]
        assert "이미 켜져" in capsys.readouterr().out
    finally:
        holder.kill()
        holder.wait()


# ------------------------------------------------- cancel against the worker


def test_a_cancel_that_loses_the_race_to_the_worker_stops_the_running_job(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    assert module.db.claim_job(job)                 # the worker took it first
    assert not module.db.claim_job(job)             # and only once
    assert client.post(f"/api/jobs/{job}/cancel").json()["status"] == "cancelling"
    assert job in module.worker.cancelled
    assert module.db.one("SELECT status FROM jobs WHERE id = ?", (job,))["status"] == "running"


def test_a_cancelled_job_is_never_claimed(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    assert client.post(f"/api/jobs/{job}/cancel").json()["status"] == "cancelled"
    assert not module.db.claim_job(job)


def test_a_dataset_whose_run_is_exporting_is_not_deleted(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    module.db.update_job(job, status="exporting")
    assert client.delete("/api/datasets/1").status_code == 400
    page = client.get("/").text
    assert "가중치 파일은 runs 폴더에 남아요." in page        # the confirm says what goes


# ------------------------------------------------- requests from elsewhere


def test_a_post_from_another_site_is_refused_and_one_from_the_lab_is_not(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes()})
    evil = client.delete("/api/datasets/1", headers={"Origin": "http://evil.example"})
    assert evil.status_code == 403
    assert client.delete("/api/datasets/1", headers={"Sec-Fetch-Site": "cross-site"}
                         ).status_code == 403
    assert client.get("/api/datasets").json()                      # still there
    ok = client.delete("/api/datasets/1", headers={"Origin": "http://testserver",
                                                   "Sec-Fetch-Site": "same-origin"})
    assert ok.status_code == 200


def test_only_addresses_and_this_machine_s_names_reach_the_lab(studio):
    client, module = studio
    assert client.get("/api/status", headers={"Host": "192.168.0.5:8080"}).status_code == 200
    assert client.get("/api/status", headers={"Host": "[::1]:8080"}).status_code == 200
    assert client.get("/api/status", headers={"Host": "localhost:8080"}).status_code == 200
    rebound = client.get("/api/status", headers={"Host": "attacker.example:8080"})
    assert rebound.status_code == 403 and "EASYDETECT_LAB_HOSTS" in rebound.json()["error"]
    assert module._host_allowed(socket.gethostname() + ":8080")


def test_an_upload_over_the_limit_is_refused_before_it_is_read(studio, monkeypatch):
    client, module = studio
    monkeypatch.setattr(module, "MAX_UPLOAD_MB", 1)
    big = upload(client, {"images/a.jpg": bytes(2 * 1024 * 1024)})
    assert big.status_code == 413 and "1 MB" in big.json()["error"]


def test_a_stopped_resumed_run_says_the_epoch_it_reached(studio):
    from worker import _stopped_note

    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 20}).json()["id"]
    for epoch in (11, 12):                         # a resume that started at epoch 11
        module.db.add_epoch(job, {"epoch": epoch, "loss": 1.0, "map50_95": 0.1, "seconds": 1})
    row = module.db.one("SELECT * FROM jobs WHERE id = ?", (job,))
    assert _stopped_note(module.db, row) == "에폭 12/20에서 중지했어요"


def test_windows_is_kept_awake_for_the_length_of_a_job(monkeypatch):
    import ctypes
    from types import SimpleNamespace

    import worker

    calls = []
    monkeypatch.setattr(worker.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        SetThreadExecutionState=calls.append)), raising=False)
    with pytest.raises(RuntimeError), worker._kept_awake():
        assert calls == [0x80000001]
        raise RuntimeError("the job failed")
    assert calls == [0x80000001, 0x80000000]        # and let go, however it ended


def test_label_mode_with_the_lab_already_running_says_so(monkeypatch, capsys, tmp_path):
    import run

    port, server = _serve(_Lab)
    monkeypatch.setattr(run, "_single_instance", lambda port: pytest.fail("started"))
    assert run.main(["label", "--source", str(tmp_path), "--port", str(port),
                     "--no-browser"]) == 0
    out = capsys.readouterr().out
    assert "이미 켜져" in out and "다른 프로그램" not in out
    server.shutdown()


def test_auto_label_never_replaces_a_file_saved_while_the_model_looked(tmp_path, monkeypatch):
    import worker

    path = tmp_path / "labels" / "a.txt"
    box = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    assert worker._write_new_labels(path, box, None)
    path.write_text("0 0.1 0.1 0.1 0.1\n")                # the person's own
    assert not worker._write_new_labels(path, box, None)
    assert path.read_text() == "0 0.1 0.1 0.1 0.1\n"
    assert sorted(p.name for p in path.parent.iterdir()) == ["a.txt"]
    monkeypatch.setattr(worker.os, "link", lambda a, b: (_ for _ in ()).throw(OSError("FAT")))
    other = tmp_path / "labels" / "b.txt"
    assert worker._write_new_labels(other, box, None) and other.read_text().startswith("0 ")
    assert not worker._write_new_labels(path, box, None)


def test_origin_null_is_refused_and_a_listed_proxy_name_is_the_lab_s_own(studio, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes()})
    assert client.delete("/api/datasets/1", headers={"Origin": "null"}).status_code == 403
    monkeypatch.setenv("EASYDETECT_LAB_HOSTS", "testserver,lab.school.kr")
    # nginx without proxy_set_header Host: the lab sees 127.0.0.1, the page its own name
    proxied = client.delete("/api/datasets/1", headers={"Host": "127.0.0.1:8080",
                                                        "Origin": "https://lab.school.kr"})
    assert proxied.status_code == 200
