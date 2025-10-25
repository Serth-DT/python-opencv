"""FastAPI service for Rubik cube facelet detection using OpenCV.

The service expects at least six base64-encoded images representing six cube
faces. Each image should contain a single face roughly aligned with the frame.
The detector divides an image into a 3×3 grid, samples the HSV average for each
cell, and maps the color to one of six canonical cube colors (U, R, F, D, L, B)
using nearest-neighbour matching in HSV space. Images are assumed to be
provided in order: Up, Right, Front, Down, Left, Back.

This is a heuristic implementation and may require calibration for real-world
lighting and cube color variations. Adjust ``COLOR_REFERENCES`` to better fit
your hardware.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Rubik Vision Service", version="0.2.0")
logger = logging.getLogger("rubik_vision")
if not logger.handlers:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s"
  )


class DetectRequest(BaseModel):
  images: List[str] = Field(
      ..., description="List of base64-encoded images representing cube faces")
  palette: Optional[Dict[str, List[int]]] = Field(
      None,
      description="Optional colour palette overrides in BGR order"
  )


class DetectResponse(BaseModel):
  facelets: str = Field(..., description="54-character facelets string")


BGR_REFERENCES = {
    'U': (230, 230, 230),   # White (BGR)
    'R': (30, 30, 230),     # Red
    'F': (30, 180, 40),     # Green
    'D': (30, 220, 255),    # Yellow
    'L': (0, 140, 255),     # Orange
    'B': (230, 60, 30),     # Blue
}


class CalibrateRequest(BaseModel):
  images: List[str] = Field(
      ..., description="List of base64-encoded images representing cube faces")


class CalibrateResponse(BaseModel):
  palette: Dict[str, List[int]] = Field(
      ..., description="Calibrated colour palette in BGR order")


def _bgr_to_lab(color) -> np.ndarray:
  sample = np.uint8([[color]])
  lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[0, 0]
  return lab.astype(np.float32)


COLOR_REFERENCES = {label: _bgr_to_lab(bgr) for label, bgr in BGR_REFERENCES.items()}

FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']


@app.post("/detect/cube", response_model=DetectResponse)
def detect_cube(req: DetectRequest) -> DetectResponse:
  if len(req.images) < 6:
    raise HTTPException(status_code=400, detail="At least 6 images are required")

  face_samples: List[List[np.ndarray]] = []
  canonical_colors: Dict[str, np.ndarray] = {}

  for index, encoded in enumerate(req.images[:6]):
    image = _decode_base64_image(encoded)
    if image is None:
      logger.warning("decode_failed image_index=%d", index)
      raise HTTPException(status_code=400, detail=f"Image {index + 1} invalid")

    lab_samples, _ = _extract_face_samples(image)
    if len(lab_samples) != 9:
      logger.warning("sticker_detection_failed image_index=%d stickers=%d", index, len(lab_samples))
      raise HTTPException(status_code=422, detail="Unable to detect 9 stickers")

    label = FACE_ORDER[index]
    face_samples.append(lab_samples)
    canonical_colors[label] = lab_samples[4]

  for label in FACE_ORDER:
    canonical_colors.setdefault(label, COLOR_REFERENCES[label])

  if req.palette:
    for label, override in req.palette.items():
      if label not in FACE_ORDER:
        continue
      try:
        constPalette = tuple(int(v) for v in override)
        canonical_colors[label] = _bgr_to_lab(constPalette)
      except Exception:
        logger.warning({"label": label, "value": override}, "Invalid palette override received")

  assignments = _assign_facelets(face_samples, canonical_colors)
  facelets = ''.join(assignments)

  if len(facelets) != 54:
    logger.error("assembly_failed facelets_length=%d", len(facelets))
    raise HTTPException(status_code=422, detail="Failed to assemble 54 stickers")

  logger.info("detect_success facelets=%s counts=%s", facelets, _count_colors(facelets))

  return DetectResponse(facelets=facelets)


@app.get("/healthz")
def healthz() -> dict[str, str]:
  return {"status": "ok"}


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest) -> CalibrateResponse:
  if len(req.images) < 6:
    raise HTTPException(status_code=400, detail="At least 6 images are required")

  palette: Dict[str, List[int]] = {}

  for index, encoded in enumerate(req.images[:6]):
    image = _decode_base64_image(encoded)
    if image is None:
      logger.warning("calibrate_decode_failed image_index=%d", index)
      raise HTTPException(status_code=400, detail=f"Image {index + 1} invalid")

    _, bgr_samples = _extract_face_samples(image)
    if len(bgr_samples) != 9:
      logger.warning("calibrate_sticker_failed image_index=%d stickers=%d", index, len(bgr_samples))
      raise HTTPException(status_code=422, detail="Unable to detect 9 stickers for calibration")

    label = FACE_ORDER[index]
    center_bgr = bgr_samples[4]
    palette[label] = [int(round(float(value))) for value in center_bgr.tolist()]

  logger.info("calibration_palette %s", palette)
  return CalibrateResponse(palette=palette)


def _decode_base64_image(data: str) -> np.ndarray | None:
  try:
    raw = base64.b64decode(data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
      return None
    return image
  except Exception:
    logger.exception("base64_decode_exception")
    return None


def _extract_face_samples(image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  height, width, _ = image.shape
  rows = np.array_split(np.arange(height), 3)
  cols = np.array_split(np.arange(width), 3)

  lab_samples: List[np.ndarray] = []
  bgr_samples: List[np.ndarray] = []

  for row_indices in rows:
    for col_indices in cols:
      roi_bgr = image[np.ix_(row_indices, col_indices)]
      mean_bgr = roi_bgr.mean(axis=(0, 1))
      mean_bgr_uint8 = np.clip(mean_bgr, 0, 255).astype(np.uint8)
      mean_lab = cv2.cvtColor(mean_bgr_uint8.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
      lab_samples.append(mean_lab)
      bgr_samples.append(mean_bgr.astype(np.float32))

  return lab_samples, bgr_samples


def _assign_facelets(
    face_samples: List[List[np.ndarray]],
    canonical_colors: Dict[str, np.ndarray]
) -> List[str]:
  samples = [sample for face in face_samples for sample in face]
  assignments = _initial_assignments(samples, canonical_colors)
  balanced = _rebalance_assignments(assignments, canonical_colors, target_per_color=9)
  return [entry['label'] for entry in balanced]


def _initial_assignments(
    samples: List[np.ndarray],
    references: Dict[str, np.ndarray]
) -> List[Dict[str, object]]:
  assignments: List[Dict[str, object]] = []
  for sample in samples:
    distances = _sorted_distances(sample, references)
    assignments.append({
        'label': distances[0][0],
        'distances': distances
    })
  return assignments


def _rebalance_assignments(
    assignments: List[Dict[str, object]],
    references: Dict[str, np.ndarray],
    target_per_color: int
) -> List[Dict[str, object]]:
  counts = {label: 0 for label in references}
  for entry in assignments:
    counts[entry['label']] += 1

  for _ in range(120):
    over_label = next((label for label in FACE_ORDER if counts[label] > target_per_color), None)
    under_label = next((label for label in FACE_ORDER if counts[label] < target_per_color), None)
    if over_label is None or under_label is None:
      break

    candidate_idx: int | None = None
    candidate_penalty: float | None = None

    for idx, entry in enumerate(assignments):
      if entry['label'] != over_label:
        continue

      base_dist = None
      alt_dist = None
      for label, distance in entry['distances']:
        if label == over_label and base_dist is None:
          base_dist = distance
        if label == under_label:
          alt_dist = distance
      if alt_dist is None:
        continue
      if base_dist is None:
        base_dist = entry['distances'][0][1]

      penalty = float(alt_dist - base_dist)
      if candidate_idx is None or penalty < candidate_penalty:
        candidate_idx = idx
        candidate_penalty = penalty

    if candidate_idx is None:
      break

    assignments[candidate_idx]['label'] = under_label
    counts[over_label] -= 1
    counts[under_label] += 1

  # Fallback balancing in case the greedy pass could not fully satisfy the constraint.
  for over_label in FACE_ORDER:
    while counts[over_label] > target_per_color:
      reassigned = False
      for idx, entry in enumerate(assignments):
        if entry['label'] != over_label:
          continue
        for label, _ in entry['distances']:
          if label == over_label:
            continue
          if counts[label] >= target_per_color:
            continue
          entry['label'] = label
          counts[over_label] -= 1
          counts[label] += 1
          reassigned = True
          break
        if reassigned:
          break
      if not reassigned:
        break

  return assignments


def _sorted_distances(
    sample: np.ndarray,
    references: Dict[str, np.ndarray]
) -> List[Tuple[str, float]]:
  distances = [
      (label, _color_distance(sample, reference))
      for label, reference in references.items()
  ]
  distances.sort(key=lambda item: item[1])
  return distances


def _color_distance(sample: np.ndarray, reference: np.ndarray) -> float:
  # Simple CIE76 distance in LAB space.
  diff = sample - reference
  return float(np.sqrt(np.sum(diff * diff)))


def _count_colors(facelets: str) -> Dict[str, int]:
  counts: Dict[str, int] = {}
  for ch in facelets:
    counts[ch] = counts.get(ch, 0) + 1
  return counts
