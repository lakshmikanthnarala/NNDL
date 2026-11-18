"""Tests for the local web and prediction API."""

import base64
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np
from PIL import Image

import server


class LocalServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.SignSpeakHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"
        cls.original_model = server.MODEL
        server.MODEL = object()

    @classmethod
    def tearDownClass(cls):
        server.MODEL = cls.original_model
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def _post(self, payload):
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/predict",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return urlopen(request)

    def test_serves_website_and_reports_model_status(self):
        with urlopen(f"{self.base_url}/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"SignSpeak", response.read())
        with urlopen(f"{self.base_url}/api/health") as response:
            self.assertEqual(json.load(response), {"status": "ok", "model": True})

    def test_prediction_api_returns_model_result_for_uploaded_image(self):
        pixels = np.arange(28 * 28, dtype=np.uint8).reshape((28, 28))
        image_file = BytesIO()
        Image.fromarray(pixels).save(image_file, format="PNG")
        image_data = base64.b64encode(image_file.getvalue()).decode("ascii")
        with patch.object(server, "predict_image", return_value=("A", 0.97, pixels)):
            with self._post({"image": f"data:image/png;base64,{image_data}"}) as response:
                self.assertEqual(
                    json.load(response),
                    {"letter": "A", "confidence": 0.97},
                )

    def test_prediction_api_rejects_unsupported_image_data_url(self):
        with self.assertRaises(HTTPError) as raised:
            self._post({"image": "data:image/gif;base64,AAAA"})
        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(
            json.load(raised.exception),
            {"error": "Choose a PNG or JPEG image."},
        )


if __name__ == "__main__":
    unittest.main()
