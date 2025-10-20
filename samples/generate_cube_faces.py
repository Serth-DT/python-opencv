"""
Utility to render six 3×3 Rubik faces to PNG (and optionally emit Base64 strings).

Usage:
  python3 generate_cube_faces.py \
     --output-dir ./samples/generated \
     --scheme solved \
     --emit-base64

By default the script renders the solved cube (each face one colour). You can
provide a simple scramble via `--scheme` where each face is a 9-character string
in order U, R, F, D, L, B (e.g. "UUUUUUUUURRRRRRRRR..."). Missing faces fall
back to solved colours.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PIL import Image
import numpy as np


DEFAULT_SCHEME = {
    "U": "UUUUUUUUU",
    "R": "RRRRRRRRR",
    "F": "FFFFFFFFF",
    "D": "DDDDDDDDD",
    "L": "LLLLLLLLL",
    "B": "BBBBBBBBB"
}

RGB_MAP: Dict[str, tuple[int, int, int]] = {
    "U": (255, 255, 0),    # yellow
    "R": (255, 0, 0),      # red
    "F": (0, 200, 0),      # green
    "D": (255, 255, 255),  # white
    "L": (255, 128, 0),    # orange
    "B": (0, 0, 255)       # blue
}

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
SIZE = 300
CELL = SIZE // 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render six Rubik faces to PNGs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rubik_vision_service/samples/generated",
        help="Directory where the face PNGs will be stored."
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        help=(
            "Optional 54-character string describing the cube in order "
            "U(9),R(9),F(9),D(9),L(9),B(9). Omitted faces fall back to default."
        )
    )
    parser.add_argument(
        "--emit-base64",
        action="store_true",
        help="Print Base64 strings for each generated face."
    )
    return parser.parse_args()


def split_scheme(raw: str | None) -> Dict[str, str]:
    if not raw:
        return DEFAULT_SCHEME.copy()

    raw = raw.strip().upper()
    if len(raw) != 54:
        raise ValueError("Scheme must contain exactly 54 characters.")

    faces: Dict[str, str] = {}
    index = 0
    for face in FACE_ORDER:
        faces[face] = raw[index:index + 9]
        index += 9
    return faces


def render_face(facelets: str) -> np.ndarray:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for idx, char in enumerate(facelets):
        colour = RGB_MAP.get(char, (128, 128, 128))
        row = idx // 3
        col = idx % 3
        start_r, start_c = row * CELL, col * CELL
        end_r, end_c = start_r + CELL, start_c + CELL

        noise = np.random.default_rng().integers(-3, 4, size=(CELL, CELL, 3), dtype=np.int16)
        patch = np.clip(np.array(colour, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        image[start_r:end_r, start_c:end_c] = patch

    edge = 3
    image[:edge, :] = 20
    image[-edge:, :] = 20
    image[:, :edge] = 20
    image[:, -edge:] = 20
    return image


def encode_base64(image: np.ndarray) -> str:
    pil = Image.fromarray(image, mode="RGB")
    buffer = BytesIO()
    pil.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    args = parse_arguments()
    scheme = split_scheme(args.scheme)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base64_payload: Dict[str, str] = {}

    for face in FACE_ORDER:
        facelets = scheme.get(face, DEFAULT_SCHEME[face])
        image = render_face(facelets)
        filepath = output_dir / f"{face}.png"
        Image.fromarray(image, mode="RGB").save(filepath)
        if args.emit_base64:
            base64_payload[face] = encode_base64(image)
        print(f"Generated {filepath}")

    if args.emit_base64:
        print("\nBase64 payload (useful for Postman variables):")
        for face in FACE_ORDER:
            print(f"{face}: {base64_payload[face][:60]}...")  # print prefix to keep log short


if __name__ == "__main__":
    main()
