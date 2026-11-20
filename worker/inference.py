"""Model wrappers: YOLO signature detector + embedding model for similarity.

Both degrade gracefully so the whole pipeline can be tested before training
finishes:
  - no detector ONNX  -> treats the whole image as one signature (logged loudly)
  - no embedder ONNX  -> uses a simple pixel-based embedding (PLACEHOLDER; weak,
                         only for plumbing tests, replace with the real model)
"""
import logging
import os

import cv2
import numpy as np
import onnxruntime as ort

from shared import config

log = logging.getLogger(__name__)
PROVIDERS = ["CPUExecutionProvider"]


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img  # BGR


# ----------------------------------------------------------------- detector
class Detector:
    def __init__(self, path: str = config.DETECTOR_PATH, size: int = 640):
        self.size = size
        self.session = None
        if os.path.exists(path):
            self.session = ort.InferenceSession(path, providers=PROVIDERS)
            self.input_name = self.session.get_inputs()[0].name
            log.info("detector loaded: %s", path)
        else:
            log.warning("DETECTOR NOT FOUND at %s -> whole-image fallback box", path)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, r, left, top

    def detect(self, img: np.ndarray) -> list[dict]:
        h, w = img.shape[:2]
        if self.session is None:
            return [{"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h), "confidence": 0.0}]

        canvas, r, left, top = self._letterbox(img)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]

        boxes, scores = [], []
        if out.ndim == 3 and out.shape[-1] == 6:
            # end-to-end export: (1, N, 6) = x1,y1,x2,y2,conf,cls (NMS already applied)
            for x1, y1, x2, y2, conf, _ in out[0]:
                if conf >= config.DETECT_CONF:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(conf))
            keep = list(range(len(boxes)))
        else:
            # classic export: (1, 4+nc, N) = cx,cy,w,h,conf...
            pred = out[0].T  # (N, 4+nc)
            conf = pred[:, 4:].max(axis=1)
            m = conf >= config.DETECT_CONF
            pred, conf = pred[m], conf[m]
            for (cx, cy, bw, bh), c in zip(pred[:, :4], conf):
                boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
                scores.append(float(c))
            if boxes:
                xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
                keep = list(np.array(cv2.dnn.NMSBoxes(xywh, scores, config.DETECT_CONF, config.DETECT_IOU)).flatten())
            else:
                keep = []

        results = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            results.append(
                {
                    "x1": float(np.clip(x1, 0, w)),
                    "y1": float(np.clip(y1, 0, h)),
                    "x2": float(np.clip(x2, 0, w)),
                    "y2": float(np.clip(y2, 0, h)),
                    "confidence": round(scores[i], 4),
                }
            )
        return sorted(results, key=lambda d: -d["confidence"])


# ----------------------------------------------------------------- embedder
class Embedder:
    def __init__(self, path: str = config.EMBEDDER_PATH):
        self.session = None
        if os.path.exists(path):
            self.session = ort.InferenceSession(path, providers=PROVIDERS)
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            # expects NCHW; fall back to 224 if the dims are dynamic
            self.h = inp.shape[2] if isinstance(inp.shape[2], int) else 224
            self.w = inp.shape[3] if isinstance(inp.shape[3], int) else 224
            log.info("embedder loaded: %s input=%sx%s", path, self.h, self.w)
        else:
            log.warning("EMBEDDER NOT FOUND at %s -> PLACEHOLDER pixel embedding", path)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if self.session is None:
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (128, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            v = (255.0 - g).flatten()  # ink = high
            v -= v.mean()
        else:
            x = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_AREA)
            x = x[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
            std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
            v = self.session.run(None, {self.input_name: (x - mean) / std})[0].flatten()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def crop_box(img: np.ndarray, box: dict, pad: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = box["x2"] - box["x1"], box["y2"] - box["y1"]
    x1 = int(max(0, box["x1"] - pad * bw))
    y1 = int(max(0, box["y1"] - pad * bh))
    x2 = int(min(w, box["x2"] + pad * bw))
    y2 = int(min(h, box["y2"] + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are L2-normalised
