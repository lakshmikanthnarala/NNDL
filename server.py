"""Serve the SignSpeak interface and local model prediction API."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from predict import DEFAULT_MODEL, load_predictor, predict_image


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MODEL = None
PREDICTION_LOCK = threading.Lock()


class SignSpeakHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok", "model": MODEL is not None})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json(404, {"error": "Endpoint not found."})
            return
        if MODEL is None:
            self._send_json(503, {"error": "The SignSpeak model is not loaded."})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Send the image as a JSON data URL."})
            return
        try:
            request_size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "A valid Content-Length header is required."})
            return
        if request_size < 1:
            self._send_json(400, {"error": "The request body is empty."})
            return
        if request_size > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "The image is too large to process."})
            return

        try:
            payload = json.loads(self.rfile.read(request_size))
            image = self._decode_image(payload["image"])
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            binascii.Error,
            OSError,
        ) as error:
            self._send_json(400, {"error": str(error) or "The image request is invalid."})
            return

        try:
            with PREDICTION_LOCK:
                letter, confidence, _ = predict_image(image, MODEL)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self._send_json(500, {"error": str(error)})
            return

        self._send_json(200, {"letter": letter, "confidence": confidence})

    @staticmethod
    def _decode_image(data_url: str) -> Image.Image:
        if not isinstance(data_url, str):
            raise ValueError("An image data URL is required.")
        header, separator, encoded = data_url.partition(",")
        if not separator or header not in ("data:image/png;base64", "data:image/jpeg;base64"):
            raise ValueError("Choose a PNG or JPEG image.")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError("The image is too large to process.")
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in ("PNG", "JPEG"):
                raise ValueError("Choose a PNG or JPEG image.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("The image dimensions are too large to process.")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SignSpeak on your computer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global MODEL
    MODEL = load_predictor(DEFAULT_MODEL)
    server = ThreadingHTTPServer((args.host, args.port), SignSpeakHandler)
    print(f"SignSpeak is ready at http://{args.host}:{args.port}")
    print("Images are processed by the model on this computer.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SignSpeak.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
