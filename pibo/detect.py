#!/usr/bin/env python3
# Apache-2.0
"""Run a model trained in the easydetect lab on Pibo (or any computer).

Needs only numpy, opencv (cv2) and onnxruntime: no easydetect, no torch.
model.onnx and labels.txt sit in this folder beside this file.

    python3 detect.py photo.jpg              # one picture: prints what it found
    python3 detect.py photo.jpg --save out.jpg
    python3 detect.py 0                      # camera 0, until Ctrl+C
    python3 detect.py 0 --show               # ... in a window (needs a screen)
    python3 detect.py 0 --conf 0.6 --every 5 # print every 5th frame only

From robot code, in the same folder:

    from detect import detect
    for x1, y1, x2, y2, score, name in detect(frame):   # frame: a BGR picture
        print(name, score)

The steps are exactly easydetect's (predictor.py), so the boxes and scores
match what the lab shows: BGR -> RGB, a plain resize to the model's size,
/255, NCHW float32; boxes come out as centre/size fractions of the picture,
scores per class (a sigmoid only if they are still raw logits); each
candidate keeps its best class, those under ``conf`` go, and a box that
covers a surer one by more than IoU 0.7 is dropped whatever its class.

Written for Python 3.6 and later, and old numpy / onnxruntime releases.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # said in main(), so that ``import detect`` still explains itself
    ort = None

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.onnx")
LABELS = os.path.join(HERE, "labels.txt")

#: The lab's defaults: a box needs a score of 0.5, and two boxes overlapping
#: by more than IoU 0.7 are one object.
CONF = 0.5
IOU = 0.7
MAX_DET = 300


def read_labels(path=LABELS):
    """Class names, one per line, in class order."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines()]


def preprocess(img, size):
    """BGR HWC uint8 -> (1, 3, size, size) float32 RGB 0..1, plain-resized."""
    resized = cv2.resize(img, (size, size))
    out = np.empty((1, 3, size, size), np.float32)
    np.multiply(resized[:, :, ::-1].transpose(2, 0, 1), np.float32(1 / 255), out=out[0],
                casting="unsafe")
    return out


def drop_duplicates(xyxy, iou=IOU):
    """Indices of the boxes to keep, from boxes sorted best first: a box
    overlapping a better one by more than ``iou`` goes, whatever the class."""
    x1, y1, x2, y2 = xyxy.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    alive = np.ones(len(xyxy), bool)
    for i in range(len(xyxy)):
        if not alive[i]:
            continue
        rest = np.nonzero(alive[i + 1:])[0] + i + 1
        if not len(rest):
            break
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        overlap = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        alive[rest[overlap > iou]] = False
    return np.nonzero(alive)[0]


def postprocess(boxes, scores, shape, conf=CONF, iou=IOU, max_det=MAX_DET):
    """(Q, 4) cx cy w h fractions + (Q, C) scores -> (N, 6) x1 y1 x2 y2 score class."""
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):  # still logits
        scores = 1.0 / (1.0 + np.exp(-scores))
    cls = scores.argmax(-1)
    best = scores.max(-1)
    keep = best >= conf
    boxes, best, cls = boxes[keep], best[keep], cls[keep]
    if len(best) > max_det:
        top = np.argsort(-best)[:max_det]
        boxes, best, cls = boxes[top], best[top], cls[top]
    order = np.argsort(-best)
    boxes, best, cls = boxes[order], best[order], cls[order]

    h, w = shape[:2]
    b = boxes.astype(np.float32)
    xyxy = np.empty_like(b)
    xyxy[..., 0] = b[..., 0] - b[..., 2] / 2
    xyxy[..., 1] = b[..., 1] - b[..., 3] / 2
    xyxy[..., 2] = b[..., 0] + b[..., 2] / 2
    xyxy[..., 3] = b[..., 1] + b[..., 3] / 2
    xyxy = xyxy * np.array([w, h, w, h], np.float32)
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)
    if iou is not None and len(xyxy) > 1:
        kept = drop_duplicates(xyxy, iou)
        xyxy, best, cls = xyxy[kept], best[kept], cls[kept]
    return np.concatenate(
        [xyxy, best.astype(np.float32)[:, None], cls.astype(np.float32)[:, None]], axis=1)


class Detector:
    """model.onnx on ONNX Runtime's CPU provider. ``threads``: CPU threads to use
    (default: all four cores of a Raspberry Pi 4)."""

    def __init__(self, model=MODEL, labels=LABELS, threads=None):
        if ort is None:
            raise ImportError("onnxruntime is not installed: pip3 install onnxruntime")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        try:
            self.session = ort.InferenceSession(model, options,
                                                providers=["CPUExecutionProvider"])
        except TypeError:  # onnxruntime older than 1.9 takes no providers
            self.session = ort.InferenceSession(model, options)
        except Exception as exc:
            raise RuntimeError(
                "could not load {} with onnxruntime {} ({}). The model needs opset 17 "
                "(onnxruntime 1.13 or newer): pip3 install -U onnxruntime".format(
                    model, getattr(ort, "__version__", "?"), exc)) from exc
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(max(shape[2], shape[3])) if len(shape) == 4 and \
            isinstance(shape[2], int) and isinstance(shape[3], int) else 320
        self.names = read_labels(labels)

    def raw(self, img):
        """``(N, 6)`` array of x1 y1 x2 y2 score class for a BGR picture, all boxes."""
        outputs = self.session.run(None, {self.input_name: preprocess(img, self.size)})
        boxes = next(a for a in outputs if a.shape[-1] == 4)
        scores = next(a for a in outputs if a is not boxes)
        return boxes[0], scores[0]

    def detect(self, img, conf=CONF, iou=IOU):
        """``[(x1, y1, x2, y2, score, class_name), ...]`` for a BGR picture (or a path),
        surest first, in the picture's own pixels."""
        if isinstance(img, str):
            path = img
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"cannot read the picture {path}")
        boxes, scores = self.raw(img)
        det = postprocess(boxes, scores, img.shape, conf=conf, iou=iou)
        out = []
        for x1, y1, x2, y2, score, c in det.tolist():
            c = int(c)
            name = self.names[c] if c < len(self.names) and self.names[c] else str(c)
            out.append((x1, y1, x2, y2, score, name))
        return out


_default = None


def detect(img, conf=CONF, iou=IOU):
    """``[(x1, y1, x2, y2, score, class_name), ...]`` — model.onnx beside this file,
    loaded on the first call."""
    global _default
    if _default is None:
        _default = Detector()
    return _default.detect(img, conf=conf, iou=iou)


COLOURS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (255, 148, 0)]


def draw(img, detections):
    """The boxes drawn on a copy of the picture. OpenCV cannot write Hangul, so
    a class name that is not plain ASCII is drawn as its number."""
    names = []
    out = img.copy()
    for x1, y1, x2, y2, score, name in detections:
        if name not in names:
            names.append(name)
        colour = COLOURS[names.index(name) % len(COLOURS)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, colour, 2)
        try:
            name.encode("ascii")
            text = name
        except UnicodeEncodeError:
            text = f"#{names.index(name)}"
        label = f"{text} {score:.2f}"
        cv2.putText(out, label, (p1[0], max(p1[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1, cv2.LINE_AA)
    return out


def say(detections, prefix=""):
    if not detections:
        print(prefix + "nothing found")
    for x1, y1, x2, y2, score, name in detections:
        print(f"{prefix}{name} {score:.2f}  ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")


def main(argv=None):
    # class names may be Korean: a Windows console's code page would print "?"
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Detect objects with model.onnx")
    parser.add_argument("source", help="a picture's path, or a camera number (0)")
    parser.add_argument("--conf", type=float, default=CONF, help="smallest score shown (0.5)")
    parser.add_argument("--model", default=MODEL, help="the .onnx to run (model.onnx here)")
    parser.add_argument("--labels", default=LABELS, help="class names (labels.txt here)")
    parser.add_argument("--save", help="write the picture with boxes drawn to this file")
    parser.add_argument("--show", action="store_true", help="show it in a window (needs a screen)")
    parser.add_argument("--threads", type=int, help="CPU threads (default: all)")
    parser.add_argument("--every", type=int, default=1, help="camera: print every n-th frame")
    args = parser.parse_args(argv)
    try:  # a Hangul class name must not crash a print into a pipe or a log file
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):  # Python 3.6, or a stream that cannot
        pass
    if ort is None:
        print("onnxruntime is not installed: pip3 install onnxruntime", file=sys.stderr)
        return 1
    model = Detector(args.model, args.labels, threads=args.threads)
    print("model {}: input {}x{}, {} classes ({})".format(
        os.path.basename(args.model), model.size, model.size, len(model.names),
        ", ".join(model.names)))

    if not args.source.isdigit():                       # a picture
        img = cv2.imread(args.source)
        if img is None:
            print(f"cannot read the picture {args.source}", file=sys.stderr)
            return 1
        started = time.time()
        found = model.detect(img, conf=args.conf)
        print(f"{(time.time() - started) * 1000:.0f} ms")
        say(found)
        if args.save:
            cv2.imwrite(args.save, draw(img, found))
            print(f"saved {args.save}")
        if args.show:
            cv2.imshow("detect", draw(img, found))
            cv2.waitKey(0)
        return 0

    camera = cv2.VideoCapture(int(args.source))
    if not camera.isOpened():
        print(f"cannot open camera {args.source}", file=sys.stderr)
        return 1
    frame_no = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("the camera stopped sending pictures", file=sys.stderr)
                break
            started = time.time()
            found = model.detect(frame, conf=args.conf)
            frame_no += 1
            if frame_no % max(args.every, 1) == 0:
                say(found, prefix=f"[{frame_no} · {(time.time() - started) * 1000:.0f} ms] ")
            if args.save:
                cv2.imwrite(args.save, draw(frame, found))
            if args.show:
                cv2.imshow("detect", draw(frame, found))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
