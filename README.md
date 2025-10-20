# Rubik Vision Service (prototype)

Small FastAPI application that accepts six cube face images and responds with a
`facelets` string consumable by the existing Rubik solver. The current version
includes a lightweight OpenCV heuristic: each image is split into a 3×3 grid,
the average HSV color is computed per cell, and mapped to the six cube colors
(U/R/F/D/L/B). Images must be supplied in the order: Up, Right, Front, Down,
Left, Back. Calibrate `COLOR_REFERENCES` in `app.py` if your cube/lighting
differs significantly.

## Quickstart

```bash
cd rubik_vision_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8081
```

Call the service:

```bash
# (Base64 strings omitted for brevity)
curl -X POST http://localhost:8081/detect/cube \
  -H 'Content-Type: application/json' \
  -d '{"images": ["<base64_up>", "<base64_right>", "<base64_front>", "<base64_down>", "<base64_left>", "<base64_back>"]}'
```

You will receive a JSON payload containing `facelets`. Use that value when
calling `/rubik/solve` on the Node backend.

## Running tests

Install the optional test dependency and execute the suite with `pytest`:

```bash
pip install pytest
pytest
```

## Sample test images

If you need quick demo assets, generate six synthetic cube faces:

```bash
cd rubik_vision_service
python3 samples/generate_faces.py
```

PNG files `U.png`, `R.png`, `F.png`, `D.png`, `L.png`, `B.png` will appear in
`rubik_vision_service/samples/`. Convert each file to Base64 (or use Postman’s
binary upload & code snippet feature) when calling the vision or solve APIs.

Bạn cần cấu hình linh hoạt hơn? Dùng script:

```bash
python3 samples/generate_cube_faces.py \
  --output-dir samples/generated \
  --emit-base64
```

Script này hỗ trợ truyền trạng thái cube (54 ký tự) qua `--scheme` và in ra Base64
để dùng thẳng với Postman/CLI.

## CLI helper

If bạn muốn thử nhanh mà không cần Postman, dùng script Node ở thư mục gốc:

```bash
node scripts/solve_rubik_from_images.mjs \
  /path/to/U.jpg /path/to/R.jpg /path/to/F.jpg \
  /path/to/D.jpg /path/to/L.jpg /path/to/B.jpg \
  --token YOUR_JWT              # nếu backend yêu cầu auth
```

Tham số mặc định: backend tại `http://localhost:3000`, vision tại
`http://localhost:8081`. Script mặc định gọi vision service trước để hiển thị
`facelets` và thống kê màu, sau đó gửi tới backend; thêm `--skip-precheck` nếu
muốn bỏ bước này. Thêm `--vision` nếu chỉ muốn gọi vision service.
