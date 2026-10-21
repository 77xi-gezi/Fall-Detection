"""Drawing helpers, color maps, and visualization utilities."""

import cv2
import numpy as np

# COCO keypoint skeleton connections (pairs of keypoint indices)
COCO_SKELETON = [
    (0, 1), (0, 2),      # nose → eyes
    (1, 3), (2, 4),      # eyes → ears
    (5, 6),               # shoulder → shoulder
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso
    (11, 12),             # hip → hip
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# COCO keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Color definitions (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)

# Status colors for different states
STATUS_COLORS = {
    "normal": COLOR_GREEN,
    "falling": COLOR_ORANGE,
    "fallen": COLOR_RED,
}


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int] = COLOR_GREEN,
    confidence_threshold: float = 0.3,
    thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    """Draw skeleton on frame from COCO keypoints.

    Args:
        frame: BGR image (modified in place).
        keypoints: Array of shape (17, 3) with [x, y, confidence].
        color: BGR color for the skeleton.
        confidence_threshold: Minimum confidence to draw a keypoint/connection.
        thickness: Line thickness.
        point_radius: Radius of keypoint circles.

    Returns:
        The frame with skeleton drawn.
    """
    if keypoints is None or len(keypoints) == 0:
        return frame

    # Draw connections
    for i, j in COCO_SKELETON:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if keypoints[i][2] < confidence_threshold or keypoints[j][2] < confidence_threshold:
            continue
        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
        cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for kp in keypoints:
        if kp[2] < confidence_threshold:
            continue
        center = (int(kp[0]), int(kp[1]))
        cv2.circle(frame, center, point_radius, color, -1)

    return frame


def draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw bounding box with optional label.

    Args:
        frame: BGR image (modified in place).
        bbox: (x1, y1, x2, y2) coordinates.
        color: BGR color.
        thickness: Line thickness.
        label: Optional text label above the box.

    Returns:
        The frame with bbox drawn.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        # Draw label above bbox if room, otherwise below top edge
        label_above = y1 - text_h - baseline - 4 >= 0
        if label_above:
            cv2.rectangle(
                frame, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1
            )
            cv2.putText(frame, label, (x1, y1 - baseline - 2), font, font_scale, COLOR_BLACK, 1)
        else:
            cv2.rectangle(
                frame, (x1, y1), (x1 + text_w, y1 + text_h + baseline + 4), color, -1
            )
            cv2.putText(
                frame, label, (x1, y1 + text_h + 2), font, font_scale, COLOR_BLACK, 1
            )

    return frame
