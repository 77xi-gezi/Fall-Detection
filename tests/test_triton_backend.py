"""Triton tensor protocol, letterbox geometry and remote model routing."""
import json

import numpy as np
import pytest

from src.pipeline import FallDetectionPipeline
from src.triton_backend import TritonHTTP, postprocess, preprocess


def test_letterbox_color_normalization_and_inverse_coordinates():
    image = np.zeros((200, 100, 3), np.uint8)
    image[:, :, 2] = 255
    tensor, ratio, padding = preprocess(image)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert padding == (160, 0) and ratio == 3.2
    assert tensor[0, :, 320, 320].tolist() == [1, 0, 0]
    output = np.zeros((1, 300, 57), np.float32)
    output[0, 0, :6] = [192, 64, 416, 576, 0.9, 0]
    output[0, 0, 6:] = np.tile([256, 192, 0.95], 17)
    persons = postprocess(output, image.shape, ratio, padding, 0.5)
    assert len(persons) == 1
    assert persons[0].bbox == pytest.approx((10, 20, 80, 180))
    np.testing.assert_allclose(persons[0].keypoints[:, :2], np.tile([30, 60], (17, 1)))


def test_binary_http_tensor_protocol(monkeypatch):
    expected = np.array([[1, 2, 3]], np.float32)
    output_header = json.dumps({"outputs": [{
        "name": "logits", "datatype": "FP32", "shape": [1, 3],
        "parameters": {"binary_data_size": expected.nbytes},
    }]}).encode()

    class Response:
        headers = {"Inference-Header-Content-Length": str(len(output_header))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return output_header + expected.tobytes()

    def open_url(request, timeout):
        size = int(request.get_header("Inference-header-content-length"))
        header = json.loads(request.data[:size])
        assert header["inputs"][0]["shape"] == [1, 1, 5]
        assert header["inputs"][0]["parameters"]["binary_data_size"] == 20
        assert len(request.data[size:]) == 20
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    actual = TritonHTTP().infer("fall_lstm", "features", "logits", np.zeros((1, 1, 5)))
    np.testing.assert_array_equal(actual, expected)


def test_remote_pipeline_uses_onnx_endpoints_without_local_weights(monkeypatch):
    calls = []

    def infer(self, model, input_name, output_name, data):
        calls.append((model, data.shape))
        if model == "yolo26s_pose":
            output = np.zeros((1, 300, 57), np.float32)
            output[0, 0, :6] = [100, 100, 400, 500, 0.95, 0]
            output[0, 0, 6:] = np.tile([200, 200, 0.95], 17)
            return output
        return np.array([[-2, 0, 3]], np.float32)

    monkeypatch.setattr(TritonHTTP, "infer", infer)
    monkeypatch.setattr("src.pipeline.PersonDetector", lambda **kw: pytest.fail("Loaded PT"))
    pipeline = FallDetectionPipeline(backend="triton")
    _, persons = pipeline.classify_image(np.zeros((640, 640, 3), np.uint8))
    assert persons[0]["state"] == "fallen"
    assert calls == [("yolo26s_pose", (1, 3, 640, 640)), ("fall_lstm", (1, 1, 5))]
    with pytest.raises(ValueError, match="640"):
        FallDetectionPipeline(backend="triton", imgsz=1280)
