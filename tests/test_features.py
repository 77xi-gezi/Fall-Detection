"""Tests for the original five-feature extraction."""

import math

import numpy as np
import pytest

from src.feature_extraction import NUM_FEATURES, FeatureExtractor, FeatureVector

# --- Helper to create keypoints ---

def make_keypoints(
    nose=(320, 100, 0.9),
    left_shoulder=(280, 200, 0.9),
    right_shoulder=(360, 200, 0.9),
    left_hip=(280, 350, 0.9),
    right_hip=(360, 350, 0.9),
    default_conf=0.9,
) -> np.ndarray:
    """Create COCO keypoints array with specified positions."""
    kps = np.zeros((17, 3), dtype=np.float32)
    # Fill with defaults at center
    for i in range(17):
        kps[i] = [320, 240, default_conf]
    # Override specific keypoints
    kps[0] = nose
    kps[5] = left_shoulder
    kps[6] = right_shoulder
    kps[11] = left_hip
    kps[12] = right_hip
    return kps


def make_standing_keypoints() -> np.ndarray:
    """Standing person: nose above hips, vertically aligned."""
    return make_keypoints(
        nose=(320, 50, 0.95),
        left_shoulder=(290, 130, 0.9),
        right_shoulder=(350, 130, 0.9),
        left_hip=(295, 280, 0.9),
        right_hip=(345, 280, 0.9),
    )


def make_fallen_keypoints() -> np.ndarray:
    """Fallen person: lying horizontal, nose and hips at similar Y."""
    return make_keypoints(
        nose=(150, 400, 0.9),
        left_shoulder=(250, 400, 0.9),
        right_shoulder=(350, 400, 0.9),
        left_hip=(400, 410, 0.9),
        right_hip=(500, 410, 0.9),
    )


def make_low_confidence_keypoints() -> np.ndarray:
    """All keypoints have very low confidence."""
    kps = make_standing_keypoints()
    kps[:, 2] = 0.1
    return kps


# --- FeatureVector Tests ---

class TestFeatureVector:
    def test_to_array(self):
        fv = FeatureVector(0.1, 0.5, -0.02, 0.4, 0.85)
        arr = fv.to_array()
        assert arr.shape == (NUM_FEATURES,)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, [0.1, 0.5, -0.02, 0.4, 0.85], rtol=1e-5)

    def test_from_array(self):
        arr = np.array([0.1, 0.5, -0.02, 0.4, 0.85], dtype=np.float32)
        fv = FeatureVector.from_array(arr)
        assert fv.body_angle == pytest.approx(0.1, abs=1e-5)
        assert fv.cog_height == pytest.approx(0.5, abs=1e-5)

    def test_roundtrip(self):
        original = FeatureVector(0.123, 0.456, -0.078, 1.234, 0.567)
        restored = FeatureVector.from_array(original.to_array())
        assert restored.body_angle == pytest.approx(original.body_angle, abs=1e-5)
        assert restored.cog_height == pytest.approx(original.cog_height, abs=1e-5)


# --- FeatureExtractor Tests ---

class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_standing_person(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)

        # Standing: body angle should be small (near vertical)
        assert fv.body_angle < 0.2  # < 36 degrees normalized
        # CoG (hips) at y≈280 in 480px frame
        assert 0.5 < fv.cog_height < 0.7
        # First frame, no velocity
        assert fv.vertical_velocity == 0.0
        # Standing bbox is tall/narrow (ratio < 1)
        assert fv.bbox_aspect_ratio < 0.5
        # Good confidence
        assert fv.keypoint_confidence > 0.8

    def test_fallen_person(self):
        kps = make_fallen_keypoints()
        bbox = (100.0, 380.0, 550.0, 430.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)

        # Fallen: body angle should be large (near horizontal)
        assert fv.body_angle > 0.3  # > 54 degrees normalized
        # CoG (hips) near bottom of frame
        assert fv.cog_height > 0.8
        # Fallen bbox is wide/short (ratio > 1)
        assert fv.bbox_aspect_ratio > 1.0

    def test_velocity_computation(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)

        # First frame
        fv1 = self.extractor.extract(kps, bbox, frame_height=480, prev_cog_height=None)
        assert fv1.vertical_velocity == 0.0

        # Second frame with known previous CoG
        fv2 = self.extractor.extract(kps, bbox, frame_height=480, prev_cog_height=0.3)
        # CoG should be around 0.58, velocity = 0.58 - 0.3 ≈ 0.28
        assert fv2.vertical_velocity > 0.0

    def test_missing_keypoints(self):
        kps = make_low_confidence_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)

        # Should not crash, should return valid features
        assert not math.isnan(fv.body_angle)
        assert not math.isnan(fv.cog_height)
        assert not math.isnan(fv.vertical_velocity)
        assert not math.isnan(fv.bbox_aspect_ratio)
        assert fv.keypoint_confidence < 0.2

    def test_zero_height_bbox(self):
        kps = make_standing_keypoints()
        bbox = (100.0, 200.0, 300.0, 200.0)  # height = 0
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        assert fv.bbox_aspect_ratio == 0.0

    def test_zero_frame_height(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=0)
        assert fv.cog_height == 0.5  # fallback

    def test_velocity_clamping(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        # Extreme previous CoG to test clamping
        fv = self.extractor.extract(kps, bbox, frame_height=480, prev_cog_height=-5.0)
        assert -1.0 <= fv.vertical_velocity <= 1.0

    def test_no_nan_or_inf(self):
        """Feature values should never be NaN or Inf."""
        test_cases = [
            (make_standing_keypoints(), (270, 40, 360, 450)),
            (make_fallen_keypoints(), (100, 380, 550, 430)),
            (make_low_confidence_keypoints(), (270, 40, 360, 450)),
            (np.zeros((17, 3), dtype=np.float32), (0, 0, 0, 0)),
        ]
        for kps, bbox in test_cases:
            fv = self.extractor.extract(kps, bbox, frame_height=480)
            arr = fv.to_array()
            assert not np.any(np.isnan(arr)), f"NaN found in features: {arr}"
            assert not np.any(np.isinf(arr)), f"Inf found in features: {arr}"

    def test_nan_keypoint_coordinates(self):
        """NaN coordinates with high confidence should not produce NaN features."""
        kps = make_standing_keypoints()
        kps[0, 0] = float("nan")  # Corrupt nose X
        kps[0, 1] = float("nan")  # Corrupt nose Y
        kps[0, 2] = 0.95  # High confidence — the dangerous case
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr)), f"NaN propagated to features: {arr}"
        assert not np.any(np.isinf(arr)), f"Inf propagated to features: {arr}"

    def test_nan_hip_coordinates(self):
        """NaN hip coordinates should not corrupt CoG height."""
        kps = make_standing_keypoints()
        kps[11, 0:2] = float("nan")  # Corrupt left hip
        kps[11, 2] = 0.9
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr)), f"NaN propagated from hip: {arr}"

    def test_single_hip_visible(self):
        kps = make_standing_keypoints()
        kps[12, 2] = 0.0  # Hide right hip
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        assert not math.isnan(fv.body_angle)
        assert not math.isnan(fv.cog_height)

    def test_shoulder_fallback(self):
        """When hips are not visible, should use shoulders."""
        kps = make_standing_keypoints()
        kps[11, 2] = 0.0  # Hide left hip
        kps[12, 2] = 0.0  # Hide right hip
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        # Should still compute a valid body angle using shoulders
        assert not math.isnan(fv.body_angle)
