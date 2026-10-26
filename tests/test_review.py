# Apache-2.0
"""What the design review changed underneath the pages: a training run is
checked before it is queued, a failure keeps its message apart from its log,
and the labelling page never puts a file name into markup unescaped."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_app import image_bytes, upload

STATIC = Path(__file__).resolve().parents[1] / "static"


# ------------------------------------------------------------ run validation


@pytest.mark.parametrize("bad, said", [
    ({"epochs": 0}, "에폭은 1에서"),
    ({"epochs": -3}, "에폭은 1에서"),
    ({"epochs": 2.5}, "에폭: 정수로"),
    ({"epochs": "many"}, "에폭: 정수로"),
    ({"batch": 0}, "배치는 1에서"),
    ({"imgsz": 100}, "32의 배수"),
    ({"imgsz": 0}, "32의 배수"),
    ({"imgsz": 4096}, "32의 배수"),
])
def test_a_run_that_cannot_work_is_refused_before_it_is_queued(studio, bad, said):
    """Epochs of -3 used to be queued and fail in the trainer a minute later."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    refused = client.post("/api/jobs", json={"dataset_id": 1, **bad})
    assert refused.status_code == 400 and said in refused.json()["error"]
    assert module.db.query("SELECT id FROM jobs") == []          # nothing was queued


def test_a_run_inside_the_limits_is_queued_as_asked(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": "3", "batch": 2, "imgsz": 64})
    assert job.status_code == 200
    row = module.db.one("SELECT epochs, batch, imgsz FROM jobs WHERE id = ?", (job.json()["id"],))
    assert row == {"epochs": 3, "batch": 2, "imgsz": 64}
    # left out, the defaults hold as before
    plain = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    row = module.db.one("SELECT epochs, imgsz FROM jobs WHERE id = ?", (plain,))
    assert row == {"epochs": 50, "imgsz": 640}


def test_train_more_needs_at_least_one_epoch(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(job, status="done")
    weights = module.RUNS / f"job{job}" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"x")
    refused = client.post(f"/api/jobs/{job}/resume", json={"add_epochs": 0})
    assert refused.status_code == 400 and "더 할 에폭" in refused.json()["error"]


# ------------------------------------------------------ failure and its log


def _worker(tmp_path, studio_module):
    import worker as worker_module

    db = studio_module.db
    return worker_module.Worker(db, tmp_path / "runs"), worker_module


def test_a_failure_keeps_its_message_apart_from_the_log(studio, tmp_path, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    w, _ = _worker(tmp_path, module)

    def boom(_job):
        w.stop()
        module.db.update_job(job, log_tail="[easydetect] epoch 1/1 loss 3.2\n"
                                           "[easydetect] out of memory")
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(w, "_train", boom)
    w.run()
    body = client.get(f"/api/jobs/{job}").json()
    assert body["status"] == "failed"
    assert body["detail"] == "RuntimeError: CUDA out of memory"          # one line, no log in it
    assert body["log_tail"].splitlines() == ["[easydetect] epoch 1/1 loss 3.2",
                                             "[easydetect] out of memory"]


def test_a_failure_without_a_log_keeps_its_last_progress_as_the_log(studio, tmp_path, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    w, _ = _worker(tmp_path, module)

    def boom(_job):
        w.stop()
        module.db.update_job(job, detail="에폭 1/1 · 배치 3/9 · 남은 시간 약 4초")
        raise ValueError("bad label")

    monkeypatch.setattr(w, "_train", boom)
    w.run()
    body = client.get(f"/api/jobs/{job}").json()
    assert body["detail"] == "ValueError: bad label"
    assert body["log_tail"] == "에폭 1/1 · 배치 3/9 · 남은 시간 약 4초"


def test_a_stopped_run_says_where_it_stopped_not_its_log(studio, tmp_path, monkeypatch):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 40}).json()["id"]
    w, worker_module = _worker(tmp_path, module)

    def stopped(_job):
        w.stop()
        for epoch in (1, 2, 3):
            module.db.add_epoch(job, {"epoch": epoch, "loss": 1.0, "map50_95": 0.1, "seconds": 1.0})
        module.db.update_job(job, log_tail="[easydetect] epoch 3/40 loss 1.0")
        raise worker_module.Cancelled()

    monkeypatch.setattr(w, "_train", stopped)
    w.run()
    body = client.get(f"/api/jobs/{job}").json()
    assert body["status"] == "cancelled"
    assert body["detail"] == "에폭 3/40에서 중지했어요"
    assert body["log_tail"] == "[easydetect] epoch 3/40 loss 1.0"


def test_a_failure_written_before_the_split_is_split_when_read(studio):
    """Rows from before log_tail carry message and log in detail, joined by " — "."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(job, status="failed",
                         detail="FileNotFoundError: best.pt — [easydetect] a / [easydetect] b")
    for body in (client.get(f"/api/jobs/{job}").json(), client.get("/api/jobs").json()[0]):
        assert body["detail"] == "FileNotFoundError: best.pt"
        assert body["log_tail"] == "[easydetect] a\n[easydetect] b"


def test_an_old_database_gets_the_log_column(tmp_path):
    import sqlite3

    from db import Database

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:      # a jobs table from before log_tail
        conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, kind TEXT, dataset_id INTEGER,"
                     " model TEXT, status TEXT, detail TEXT, created REAL)")
        conn.execute("INSERT INTO jobs VALUES (1, 'train', 1, 'm', 'failed', 'x — y', 0)")
    db = Database(path)
    row = db.one("SELECT detail, log_tail FROM jobs WHERE id = 1")
    assert row == {"detail": "x — y", "log_tail": None}                 # kept, and readable


# --------------------------------------------------- the labelling page markup


def _label_script() -> str:
    page = (STATIC / "label.html").read_text(encoding="utf-8")
    return re.findall(r"<script>(.*?)</script>", page, re.S)[0]


def test_file_and_folder_names_reach_the_labelling_page_escaped():
    """A picture named <img onerror=…>.jpg must show as text, not run."""
    script = _label_script()
    files = script[script.index("function renderFiles"):script.index("async function load(")]
    folders = script[script.index("function renderFolders"):script.index("function step(")]
    assert "${esc(label)}" in files and "${label}" not in files
    assert '${esc(d)}' in folders and 'value="${d}"' not in folders
    # every name interpolated into markup on the page goes through esc()
    for name in ("label", "name", "d"):
        for hit in re.finditer(r"innerHTML\s*=\s*`[^`]*\$\{" + name + r"\}", script):
            pytest.fail(f"unescaped ${{{name}}} in markup: {hit.group(0)[:80]}")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_labelling_page_escape_neutralises_markup():
    line = next(ln for ln in _label_script().splitlines() if ln.startswith("const esc ="))
    body = _label_script()
    start = body.index(line)
    source = body[start:body.index(";", body.index("[c]", start)) + 1]
    out = subprocess.run(
        ["node", "-e", source + ' console.log(esc(\'<img src=x onerror="alert(1)">&.jpg\'))'],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;.jpg"
