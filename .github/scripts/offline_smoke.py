# Apache-2.0
"""Drive a running offline bundle through the classroom path, with the stdlib only.

    python -X utf8 offline_smoke.py http://127.0.0.1:8080 <scratch folder> [local]

Waits for the lab, uploads ten generated pictures with labels, trains the
수업용 (CPU) preset for two epochs at a small input size, checks that the run
started from the cached COCO weights (no download happened), tries the model
on a picture, pre-labels a picture with the cached COCO IR, and downloads the
Pibo zip and runs its detect.py with plain Python. Exits non-zero on any failure.
Run it with the bundle's own python.exe: it needs numpy and OpenCV.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np

BASE = sys.argv[1].rstrip("/")
SCRATCH = Path(sys.argv[2])
#: "local" as a third argument: a machine without the cached COCO weights (a
#: development box with no route to the mirror) may start from ImageNet
LOCAL = sys.argv[3:] == ["local"]
# the lab is on this machine: never through the dead proxy the test sets
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(method: str, path: str, body: bytes | None = None, ctype: str | None = None,
         timeout: float = 600) -> bytes:
    request = urllib.request.Request(BASE + path, data=body, method=method)
    if ctype:
        request.add_header("Content-Type", ctype)
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path}: HTTP {exc.code} {exc.read()[:500]!r}") from None


def as_json(data: bytes):
    return json.loads(data.decode("utf-8"))


def post_json(path: str, payload: dict):
    return as_json(call("POST", path, json.dumps(payload).encode(), "application/json"))


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]):
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                  f"{value}\r\n".encode())
    for name, (filename, data, ctype) in files.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                  f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
        out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def picture(rng) -> tuple[np.ndarray, str]:
    h, w = 240, 320
    img = np.full((h, w, 3), int(rng.integers(60, 200)), np.uint8)
    lines = []
    for _ in range(int(rng.integers(1, 3))):
        cls = int(rng.integers(0, 2))
        bw, bh = int(rng.integers(40, 110)), int(rng.integers(40, 110))
        x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        colour = (30, 30, 220) if cls == 0 else (220, 120, 20)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, -1)
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    return img, "\n".join(lines) + "\n"


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while True:
        try:
            status = as_json(call("GET", "/api/status", timeout=5))
            break
        except (OSError, SystemExit):
            if time.time() - started > 300:
                raise SystemExit("the lab did not answer within 5 minutes") from None
            time.sleep(3)
    print(f"[smoke] lab up after {time.time() - started:.0f} s: {status}")
    page = call("GET", "/").decode("utf-8")
    assert 'data-preset="classroom"' in page, "the page has no 수업용 preset"

    rng = np.random.default_rng(0)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(10):
            img, labels = picture(rng)
            zf.writestr(f"images/p{i}.jpg", cv2.imencode(".jpg", img)[1].tobytes())
            zf.writestr(f"labels/p{i}.txt", labels)
            if i == 0:
                cv2.imwrite(str(SCRATCH / "test.jpg"), img)
        zf.writestr("data.yaml", "names: [box, ball]\n")
    body, ctype = multipart({"name": "smoke"},
                            {"archive": ("smoke.zip", archive.getvalue(), "application/zip")})
    dataset = as_json(call("POST", "/api/datasets", body, ctype))
    print(f"[smoke] dataset {dataset}")

    # 수업용 (CPU), shortened to two epochs; at its 320 px, so it starts from
    # the bundle's ed-n-320.pt
    job = post_json("/api/jobs", {
        "dataset_id": dataset["id"], "model": "ed-n", "epochs": 2, "imgsz": 320, "batch": 8,
        "freeze": "backbone", "patience": 10, "augment": False, "queries": 100,
        "denoising": 30, "val_ratio": 0.2})["id"]
    t0 = time.time()
    while True:
        info = as_json(call("GET", f"/api/jobs/{job}"))
        if info["status"] in ("done", "failed", "cancelled"):
            break
        if time.time() - t0 > 1800:
            raise SystemExit(f"training #{job} did not finish in 30 minutes: {info['detail']}")
        time.sleep(5)
    print(f"[smoke] training #{job}: {info['status']} in {time.time() - t0:.0f} s; "
          f"{info['detail']}")
    if info["status"] != "done":
        print(info.get("log_tail") or "")
        raise SystemExit("training did not finish")
    run = info["run"]
    print(f"[smoke] started from {run.get('start')}, queries {run.get('queries')}, "
          f"denoising {run.get('denoising')}, epochs {[e['seconds'] for e in info['epochs_done']]}")
    if run.get("start", {}).get("weights") != "ed-n-320":
        message = f"ed-n-320.pt did not come from the bundle's cache: {run.get('start')}"
        if not LOCAL:
            raise SystemExit(message)
        print(f"[smoke] (local run, allowed) {message}")
    assert run.get("queries") == 100 and run.get("denoising") == 30, run

    # the trained model on a picture (its OpenVINO IR, compiled here)
    body, ctype = multipart({"conf": "0.05"}, {"image": ("t.jpg", (SCRATCH / "test.jpg")
                                                         .read_bytes(), "image/jpeg")})
    assert call("POST", f"/api/jobs/{job}/predict", body, ctype)[:2] == b"\xff\xd8"
    # a COCO draft: the ed-s IR from the cache
    try:
        drafted = post_json(f"/api/datasets/{dataset['id']}/autolabel/0", {"model": "ed-s"})
        print(f"[smoke] COCO draft boxes: {len(drafted['boxes'])}")
    except SystemExit as exc:
        if not LOCAL:
            raise
        print(f"[smoke] (local run, allowed) no COCO draft: {exc}"[:300])

    data = call("GET", f"/api/jobs/{job}/download/pibo")
    here = SCRATCH / "pibo"                 # on the robot: the one file, and detect.py to try it
    here.mkdir(parents=True, exist_ok=True)
    (here / f"pibo-job{job}.onnx").write_bytes(data)
    shutil.copy(Path(__file__).resolve().parents[2] / "pibo" / "detect.py", here)
    print(f"[smoke] Pibo model {len(data) / 1e6:.1f} MB")
    out = subprocess.run([sys.executable, "-I", "detect.py", str(SCRATCH / "test.jpg"),
                          "--conf", "0"], cwd=here, capture_output=True, text=True,
                         encoding="utf-8", timeout=300)
    print(out.stdout[-1500:], out.stderr[-1500:])
    assert out.returncode == 0 and "input 320x320, 2 classes (box, ball)" in out.stdout
    print(f"[smoke] all good in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
