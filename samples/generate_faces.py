"""
Utility script to generate six synthetic cube face images for quick testing.

Each output file is a 300×300 PNG stored in the same directory as this script,
named `<FACE>.png` (U, R, F, D, L, B). The colours roughly match the canonical
Rubik cube scheme and add a small amount of noise so the detection heuristic
receives realistic values.

Usage:
    python3 generate_faces.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
TILE_COLORS = {
    'U': (255, 255, 255),  # white
    'R': (0, 0, 255),      # red
    'F': (0, 255, 0),      # green
    'D': (0, 255, 255),    # yellow
    'L': (0, 165, 255),    # orange
    'B': (255, 0, 0),      # blue
}

SIZE = 300
CELL = SIZE // 3
RNG = np.random.default_rng(42)


def _build_face(label: str, bgr: tuple[int, int, int]) -> np.ndarray:
    """Create a synthetic 3×3 cube face with slight colour noise."""
    face = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    base = np.asarray(bgr, dtype=np.int16)

    for row in range(3):
        for col in range(3):
            start_row, start_col = row * CELL, col * CELL
            end_row, end_col = start_row + CELL, start_col + CELL
            noise = RNG.integers(-6, 7, size=(CELL, CELL, 3), dtype=np.int16)
            patch = np.clip(base + noise, 0, 255).astype(np.uint8)
            face[start_row:end_row, start_col:end_col] = patch

    cv2.rectangle(face, (0, 0), (SIZE - 1, SIZE - 1), (20, 20, 20), thickness=4)
    return face


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, colour in TILE_COLORS.items():
        face = _build_face(label, colour)
        output_path = OUTPUT_DIR / f'{label}.png'
        if not cv2.imwrite(str(output_path), face):
            raise RuntimeError(f'Unable to write {output_path}')
        print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
