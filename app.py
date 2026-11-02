"""
EgoBlur service — face + licence-plate anonymisation for submission photos.

Runs Meta's EgoBlur Gen2 TorchScript detectors on CPU and blurs every detection with an
elliptical mask (same approach as the reference script_demo.py). Internal-only: the API reaches
it at http://egoblur:8000 inside the Compose network; it is never exposed publicly.

  POST /blur    multipart "image" → image/jpeg (X-Faces / X-Plates headers carry counts)
  POST /detect  multipart "image" → JSON boxes
  GET  /health  readiness (models loaded)
"""
import os
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  (registers torchvision ops used inside the TorchScript graphs)
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
FACE_T = float(os.getenv("FACE_THRESHOLD", "0.5"))  # Gen2 scores run lower than Gen1
LP_T = float(os.getenv("LP_THRESHOLD", "0.5"))
NMS_IOU = float(os.getenv("NMS_IOU", "0.3"))
SCALE = float(os.getenv("BOX_SCALE", "1.15"))            # grow boxes a little so blur covers edges

JPEG_Q = int(os.getenv("JPEG_QUALITY", "88"))
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

app = FastAPI(title="EgoBlur")
MODELS: dict = {}
LOCK = threading.Lock()  # one inference at a time keeps memory flat on a shared box
STATE = {"loaded": False, "error": None, "load_seconds": None}


class _Gen2(torch.nn.Module):
    """Gen2 weights are a scripted Detectron2 GeneralizedRCNN. Its forward() runs a postprocess step
    that is not scriptable, and its Instances type can't cross into Python — so call inference()
    without postprocessing and pull plain tensors out inside TorchScript."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.m.inference([{"image": img}], None, False)
        inst = out[0]
        pb = inst._pred_boxes  # ScriptedInstances stores fields as Optional private attributes
        sc = inst._scores
        if pb is None or sc is None:
            return torch.zeros((0, 4)), torch.zeros((0,))
        return pb.tensor, sc


def _load():
    t0 = time.time()
    try:
        for name in ("face", "lp"):
            path = os.path.join(MODEL_DIR, f"ego_blur_{name}_gen2.jit")
            raw = torch.jit.load(path, map_location="cpu")  # frozen graph: never call .eval() on it
            try:
                MODELS[name] = ("gen2", torch.jit.script(_Gen2(raw)))
            except Exception:
                MODELS[name] = ("gen1", raw)  # (boxes, labels, scores, dims) signature
        STATE.update(loaded=True, load_seconds=round(time.time() - t0, 1), kinds={k: v[0] for k, v in MODELS.items()})
    except Exception as e:  # surfaced on /health instead of crash-looping
        STATE.update(error=str(e))


@app.on_event("startup")
def startup():
    threading.Thread(target=_load, daemon=True).start()


SHORT_EDGE = int(os.getenv("SHORT_EDGE", "800"))   # Detectron2 test-time ResizeShortestEdge defaults
MAX_SIZE = int(os.getenv("MAX_SIZE", "1333"))


def _detect(entry, bgr: np.ndarray, threshold: float):
    kind, model = entry
    h, w = bgr.shape[:2]
    k = SHORT_EDGE / min(h, w)
    if max(h, w) * k > MAX_SIZE:
        k = MAX_SIZE / max(h, w)
    nh, nw = int(h * k + 0.5), int(w * k + 0.5)
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR) if (nh, nw) != (h, w) else bgr
    tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))  # BGR uint8 CHW (Detectron2 default)
    with torch.no_grad():
        if kind == "gen2":
            boxes, scores = model(tensor)
        else:
            out = model(tensor)
            boxes, scores = out[0], out[2]
    if boxes.numel() == 0:
        return []
    keep = torchvision.ops.nms(boxes.float(), scores.float(), NMS_IOU)
    boxes, scores = boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()
    return [(b / k).tolist() for b, s in zip(boxes, scores) if s >= threshold]  # back to original pixels


def _scale(box, w, h):
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * SCALE, (y2 - y1) * SCALE
    return max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))


def _blur(bgr: np.ndarray, boxes):
    if not boxes:
        return bgr
    h, w = bgr.shape[:2]
    fg = bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = _scale(box, w, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = fg[y1:y2, x1:x2]
        k = (max(3, (x2 - x1) // 2), max(3, (y2 - y1) // 2))
        fg[y1:y2, x1:x2] = cv2.blur(cv2.blur(roi, k), k)  # two passes: no recoverable detail
        cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
    m3 = cv2.merge([mask, mask, mask])
    return np.where(m3 == 255, fg, bgr)


def _read(image: UploadFile) -> np.ndarray:
    data = image.file.read()
    if not data:
        raise HTTPException(400, "empty image")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(415, "not a decodable image")
    return bgr


def _ready():
    if not STATE["loaded"]:
        raise HTTPException(503, STATE["error"] or "models loading")


def _run(bgr, face_t, lp_t):
    with LOCK:
        faces = _detect(MODELS["face"], bgr, face_t)
        plates = _detect(MODELS["lp"], bgr, lp_t)
    return faces, plates


@app.get("/health")
def health():
    body = {"ok": STATE["loaded"], **STATE, "threads": torch.get_num_threads(), "torch": torch.__version__}
    return JSONResponse(body, status_code=200 if STATE["loaded"] else 503)


# Plain `def`: FastAPI runs it in a worker thread, so CPU inference never blocks /health.
@app.post("/blur")
def blur(image: UploadFile = File(...), face_threshold: float = Query(FACE_T), lp_threshold: float = Query(LP_T)):
    _ready()
    bgr = _read(image)
    t0 = time.time()
    faces, plates = _run(bgr, face_threshold, lp_threshold)
    out = _blur(bgr, faces + plates)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(jpg.tobytes(), media_type="image/jpeg", headers={
        "X-Faces": str(len(faces)), "X-Plates": str(len(plates)), "X-Elapsed-Ms": str(int((time.time() - t0) * 1000))})


@app.post("/detect")
def detect(image: UploadFile = File(...), face_threshold: float = Query(FACE_T), lp_threshold: float = Query(LP_T)):
    _ready()
    bgr = _read(image)
    faces, plates = _run(bgr, face_threshold, lp_threshold)
    return {"faces": faces, "plates": plates, "width": bgr.shape[1], "height": bgr.shape[0]}
