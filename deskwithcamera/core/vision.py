"""Small camera helpers shared by the framework (no project-specific code)."""
import cv2
import numpy as np


def rectify(frame, H, size=(1280, 720)):
    """Camera frame -> top-down view of the projected content area (H = camera px -> 0..1)."""
    S = np.diag([size[0], size[1], 1.0]) @ H
    return cv2.warpPerspective(frame, S, tuple(size), flags=cv2.INTER_LINEAR)


def board_gray(frame, H, size=(640, 360)):
    """Gray, small top-down view (used by the hand-on-button detection)."""
    return cv2.cvtColor(rectify(frame, H, size), cv2.COLOR_BGR2GRAY)
