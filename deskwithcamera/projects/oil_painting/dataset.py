"""Saving one oil-painting capture to disk.

data/sessions/2026-11-11_153012/
    frame.png        full camera frame (to reprocess later)
    board.png        top-down view of the projected area (with the photo on it)
    reference.png    top-down view of the empty table
    overlay.png      board with the detected photo drawn in (for checking by eye)
    photo/photo.png       the photo as the camera saw it (inner part, perspective-corrected) = what was painted
    photo/photo_raw.png   the whole photo, perspective-corrected
    painting/painting.png the oil painting (always the LATEST stroke / color of this capture)
    session.json     calibration, stroke, color, sizes
All images are normal PNG files (cv2.imread gives BGR, PIL gives RGB).
"""
import json
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"


def save_capture(root, frame, board, reference, overlay, photo, photo_raw, painting, calib, stroke, color):
    """Write everything; returns the session folder."""
    name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = Path(root) / "sessions" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "photo").mkdir(exist_ok=True)
    (folder / "painting").mkdir(exist_ok=True)
    for fname, img in (("frame.png", frame), ("board.png", board), ("reference.png", reference),
                       ("overlay.png", overlay), ("photo/photo.png", photo), ("photo/photo_raw.png", photo_raw),
                       ("painting/painting.png", painting)):
        if img is not None:
            cv2.imwrite(str(folder / fname), img)
    write_info(folder, name, calib, board, photo, painting, stroke, color)
    return folder


def update_painting(folder, painting, stroke, color, calib=None, board=None, photo=None):
    """STROKE / COLOR painted the same photo again: replace painting/painting.png and the settings in session.json."""
    folder = Path(folder)
    (folder / "painting").mkdir(exist_ok=True)
    cv2.imwrite(str(folder / "painting" / "painting.png"), painting)
    info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    info["stroke"], info["color"] = stroke, color
    info["painting_size"] = [painting.shape[1], painting.shape[0]]
    (folder / "session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_info(folder, name, calib, board, photo, painting, stroke, color):
    info = {"session": name,
            "board_size": [board.shape[1], board.shape[0]],
            "photo_size": [photo.shape[1], photo.shape[0]],
            "painting_size": [painting.shape[1], painting.shape[0]],
            "stroke": stroke, "color": color,
            "calibration": {"H_cam_to_norm": calib.H.tolist(), "cam_size": list(calib.cam_size), "rms": calib.rms}}
    (Path(folder) / "session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
