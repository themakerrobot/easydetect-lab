# Apache-2.0
"""Label files, and the model-assisted first pass over them.

Labels are the same ``labels/<name>.txt`` files the trainer reads, in
Ultralytics' formats, all normalised 0-1:

- a box:        ``cls cx cy w h``
- an outline:   ``cls x1 y1 x2 y2 ...`` (a segmentation dataset; its box is
  the outline's)
- keypoints:    ``cls cx cy w h`` then ``x y v`` for each of the dataset's
  ``kpt_shape[0]`` keypoints (a keypoint dataset; ``v`` 0 unlabelled, 1
  hidden, 2 visible)

In the page and the API one object is a dict: ``cls cx cy w h``, plus
``points`` (``[[x, y], ...]``) for an outline or ``kpts`` (``[[x, y, v],
...]``) for keypoints. Nothing here is clever; the value is that the detector
writes the first draft and a person only corrects it.
"""

from __future__ import annotations

from pathlib import Path

from easydetect.data.labels import label_path as _shared_label_path
from easydetect.data.labels import label_row_to_box, label_row_to_keypoints

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def label_path(image: Path) -> Path:
    """Where the trainer will read this image's boxes — the only place to write them.

    One rule, shared with the trainer (``easydetect.data.labels``): the last
    ``images`` folder becomes ``labels``, and with none the label sits beside
    the image. Two rules used to exist here, and a folder not called ``images``
    got its labels written where training never looked.
    """
    return Path(_shared_label_path(Path(image)))


def labels_beside(images_root: Path) -> Path:
    """The folder labels for images directly in ``images_root`` land in."""
    return label_path(Path(images_root) / "x.jpg").parent


def read_labels(path: Path, kpt_shape=None) -> list[dict]:
    """A label file's objects; with ``kpt_shape`` (a keypoint dataset's) their
    keypoints too. Outlines keep their points, and their box is the outline's."""
    path = Path(path)
    if not path.exists():
        return []
    objects = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if kpt_shape is not None:
            parsed = label_row_to_keypoints(values, kpt_shape)
            if parsed is not None:
                (cls, cx, cy, w, h), kpts = parsed
                objects.append({"cls": int(cls), "cx": cx, "cy": cy, "w": w, "h": h,
                                "kpts": [[round(float(x), 6), round(float(y), 6), int(v)]
                                         for x, y, v in kpts]})
                continue
        row = label_row_to_box(values, kpt_shape)   # boxes, and polygons as their boxes
        if row is None:
            continue
        cls, cx, cy, w, h = row
        item = {"cls": int(cls), "cx": cx, "cy": cy, "w": w, "h": h}
        if kpt_shape is None and len(values) >= 7 and len(values) % 2 == 1:
            item["points"] = [[float(values[i]), float(values[i + 1])]
                              for i in range(1, len(values), 2)]
        objects.append(item)
    return objects


def write_labels(path: Path, boxes: list[dict], kpt_shape=None) -> None:
    """An image with no objects is a valid label: an empty file, not a missing one.

    An object with three or more ``points`` is written as an outline; in a
    keypoint dataset (``kpt_shape``) every object carries its keypoints,
    unlabelled ones as zeros, so each line has the length the trainer expects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_label_line(b, kpt_shape) for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _label_line(b: dict, kpt_shape=None) -> str:
    cls = int(b["cls"])
    points = b.get("points") or []
    if kpt_shape is None and len(points) >= 3:
        return f"{cls} " + " ".join(f"{_clamp(x):.6f} {_clamp(y):.6f}" for x, y, *_ in points)
    line = (f"{cls} {_clamp(b['cx']):.6f} {_clamp(b['cy']):.6f} "
            f"{_clamp(b['w']):.6f} {_clamp(b['h']):.6f}")
    if kpt_shape is None:
        return line
    k, dims = int(kpt_shape[0]), int(kpt_shape[1])
    given = list(b.get("kpts") or [])[:k]
    given += [[0.0, 0.0, 0]] * (k - len(given))
    values = []
    for point in given:
        x, y = float(point[0]), float(point[1])
        v = int(point[2]) if len(point) > 2 else 2
        if v <= 0:
            x = y = 0.0
        values.append(f"{_clamp(x):.6f} {_clamp(y):.6f}" + (f" {min(v, 2)}" if dims == 3 else ""))
    return line + " " + " ".join(values)


def box_of(points: list[list[float]]) -> dict:
    """``cx cy w h`` of an outline's points."""
    xs = [_clamp(p[0]) for p in points]
    ys = [_clamp(p[1]) for p in points]
    return {"cx": (min(xs) + max(xs)) / 2, "cy": (min(ys) + max(ys)) / 2,
            "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


TASKS = ("detect", "segment", "pose")


def kpt_shape_of(cfg: dict | None) -> list[int] | None:
    """A data.yaml's ``kpt_shape`` as ``[K, dims]``; None for a box or outline dataset."""
    shape = (cfg or {}).get("kpt_shape")
    if not shape:
        return None
    return [int(shape[0]), int(shape[1]) if len(shape) > 1 else 3]


def keypoint_names(cfg: dict | None) -> list[str] | None:
    """The keypoints' names (``kpt_names``: a list, or Ultralytics' one list per class)."""
    shape = kpt_shape_of(cfg)
    if shape is None:
        return None
    names = cfg.get("kpt_names")
    if isinstance(names, dict):
        names = next(iter(names.values()), None)
    if not names or len(names) != shape[0]:
        names = [f"kp{i}" for i in range(shape[0])]
    return [str(n) for n in names]


def has_outlines(label_files, look: int = 200) -> bool:
    """Whether any of the first ``look`` existing label files holds an outline."""
    seen = 0
    for path in label_files:
        if seen >= look:
            break
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        seen += 1
        if any(len(v := line.split()) >= 7 and len(v) % 2 == 1 for line in text.splitlines()):
            return True
    return False


def dataset_task(cfg: dict | None, label_files=()) -> str:
    """What a dataset is labelled for: keypoints when its data.yaml has a
    ``kpt_shape``, then what its ``task:`` says, then outlines if any label
    holds one, else boxes."""
    cfg = cfg or {}
    if kpt_shape_of(cfg) is not None:
        return "pose"
    if cfg.get("task") in ("detect", "segment"):
        return cfg["task"]
    return "segment" if has_outlines(label_files) else "detect"


_SIDES = [("left", "right"), ("Left", "Right"), ("LEFT", "RIGHT"), ("왼쪽", "오른쪽"),
          ("왼", "오른"), ("좌", "우")]


def flip_pairs(names: list[str]) -> list[int]:
    """Each keypoint's mirror image (``flip_idx``): left_eye <-> right_eye,
    l_hand <-> r_hand, 왼손 <-> 오른손; itself when it has no other side."""
    index = {n: i for i, n in enumerate(names)}
    out = []
    for i, name in enumerate(names):
        other = None
        for a, b in _SIDES:
            for x, y in ((a, b), (b, a)):
                if x in name and name.replace(x, y, 1) in index:
                    other = index[name.replace(x, y, 1)]
                    break
            if other is not None:
                break
        if other is None:
            low = name.lower()
            for x, y in (("l_", "r_"), ("r_", "l_")):
                if low.startswith(x):
                    twin = next((n for n in names if n.lower() == y + low[2:]), None)
                    other = index[twin] if twin else None
            for x, y in (("_l", "_r"), ("_r", "_l")):
                if other is None and low.endswith(x):
                    twin = next((n for n in names if n.lower() == low[:-2] + y), None)
                    other = index[twin] if twin else None
        out.append(i if other is None else other)
    # only pairs that point at each other; a one-sided match flips onto itself
    return [j if out[j] == i else i for i, j in enumerate(out)]


def predict_boxes(model, image: Path, names: list[str], conf: float = 0.35,
                  kpt_names: list[str] | None = None) -> list[dict]:
    """Run a model over one image and keep the boxes whose class we label.

    Classes are matched by name, case-insensitively: a COCO model labelling a
    ``person, car`` project contributes its people and cars and stays quiet
    about the rest. A segmenting model's masks come back as ``points``; a
    keypoint model's keypoints as ``kpts`` for the dataset's ``kpt_names`` —
    matched by name, or by position when none match and the counts agree —
    with the ones it is unsure of left unlabelled.
    """
    result = model.predict(str(image), conf=conf, verbose=False)[0]
    lookup = {name.lower(): i for i, name in enumerate(names)}
    outlines = result.masks.xy if getattr(result, "masks", None) is not None else None
    points = getattr(result, "keypoints", None)
    order = _keypoint_order(points.names, kpt_names) if points is not None and kpt_names else None
    if outlines is not None or order is not None:
        height, width = result.orig_shape[:2]
    boxes = []
    for i in range(len(result.boxes)):
        name = result.name_of(result.boxes.cls[i]).lower()
        if name not in lookup:
            continue
        cx, cy, w, h = result.boxes.xywhn[i]
        item = {
            "cls": lookup[name],
            "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
            "conf": round(float(result.boxes.conf[i]), 3),
        }
        if outlines is not None:
            outline = simplify_outline(outlines[i], width, height)
            if len(outline) >= 3:
                item["points"] = outline
        if order is not None:
            kpts = []
            for j in order:
                if j is None or float(points.data[i, j, 2]) < KEYPOINT_CONF:
                    kpts.append([0.0, 0.0, 0])
                    continue
                x, y = float(points.data[i, j, 0]), float(points.data[i, j, 1])
                kpts.append([round(_clamp(x / width), 6), round(_clamp(y / height), 6), 2])
            item["kpts"] = kpts
        boxes.append(item)
    return boxes


KEYPOINT_CONF = 0.5  # a model's keypoint below this is left for a person to place


def _keypoint_order(model_names, wanted: list[str]) -> list[int | None] | None:
    """For each keypoint the dataset wants, the model's index of it."""
    have = {str(n).lower(): j for j, n in enumerate(model_names or ())}
    order = [have.get(str(n).lower()) for n in wanted]
    if any(j is not None for j in order):
        return order
    if len(model_names or ()) == len(wanted):
        return list(range(len(wanted)))
    return None


def simplify_outline(xy, width: int, height: int, most: int = 60) -> list[list[float]]:
    """A mask's outline (pixel points) as at most ``most`` normalised points —
    few enough to drag about, close enough to train on."""
    import cv2
    import numpy as np

    pts = np.asarray(xy, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return []
    contour = pts.reshape(-1, 1, 2)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    simple = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(simple) > most:
        epsilon *= 1.5
        simple = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(simple) < 3:
        simple = pts
    return [[round(_clamp(x / width), 6), round(_clamp(y / height), 6)] for x, y in simple]
