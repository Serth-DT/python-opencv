import base64
from typing import Dict

import cv2
import numpy as np
from fastapi.testclient import TestClient

from rubik_vision_service.app import app, FACE_ORDER


client = TestClient(app)

_rng = np.random.default_rng(42)

_BGR_COLOR_MAP: Dict[str, tuple[int, int, int]] = {
    'U': (0, 255, 255),    # yellow
    'R': (0, 0, 255),      # red
    'F': (0, 255, 0),      # green
    'D': (255, 255, 255),  # white
    'L': (0, 165, 255),    # orange
    'B': (255, 0, 0),      # blue
}


def _generate_face_image(label: str, size: int = 300) -> str:
    """Create a synthetic cube face image and return it as base64."""
    if label not in _BGR_COLOR_MAP:
        raise ValueError(f'Unknown label: {label}')

    image = np.zeros((size, size, 3), dtype=np.uint8)
    tile_size = size // 3
    bgr_color = np.array(_BGR_COLOR_MAP[label], dtype=np.uint8)

    for row in range(3):
        for col in range(3):
            start_row, start_col = row * tile_size, col * tile_size
            end_row, end_col = start_row + tile_size, start_col + tile_size
            # Slight noise keeps HSV averages realistic without changing the expected color.
            noise = _rng.integers(-5, 6, size=(tile_size, tile_size, 3), dtype=np.int16)
            patch = np.clip(bgr_color + noise, 0, 255).astype(np.uint8)
            image[start_row:end_row, start_col:end_col] = patch

    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError('Failed to encode synthetic cube image')

    return base64.b64encode(buffer.tobytes()).decode('ascii')


def test_detect_cube_returns_facelets_for_synthetic_faces():
    payload = {
        'images': [_generate_face_image(label) for label in FACE_ORDER]
    }

    response = client.post('/detect/cube', json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    facelets = data['facelets']

    assert len(facelets) == 54
    for label in FACE_ORDER:
        assert facelets.count(label) == 9


def test_detect_cube_requires_six_images():
    payload = {'images': [_generate_face_image('U')]}
    response = client.post('/detect/cube', json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body['detail'] == 'At least 6 images are required'


def test_detect_cube_rejects_invalid_image_payload():
    payload = {'images': ['not_base64'] * 6}
    response = client.post('/detect/cube', json=payload)

    assert response.status_code == 400
    assert 'invalid' in response.json()['detail'].lower()
