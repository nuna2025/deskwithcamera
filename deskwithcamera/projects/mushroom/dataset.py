"""Saving one capture session to disk.

data/sessions/2026-11-02_153012/
    frame.png        full camera frame (to reprocess later)
    board.png        top-down view of the projected area
    reference.png    top-down view of the empty table
    overlay.png      board with the detected cards drawn in (for checking by eye)
    cards/poisonous/c001.png ...   256x256, inner part of the card, aspect ratio kept (white padding)
    cards/safe/c005.png ...
    raw/<label>/cNNN.png           whole card, perspective-corrected, original aspect ratio
    manifest.csv     one row per card (see COLUMNS)
    session.json     calibration + settings used
All images are normal PNG files (open them with any viewer; cv2.imread gives BGR, PIL gives RGB).
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2

from projects.mushroom import settings as cfg
from projects.mushroom.cards import crop_inner, crop_raw, letterbox, white_reference

COLUMNS = ["card_id", "session", "side", "label", "x_norm", "y_norm", "angle", "long_px", "short_px",
           "aspect", "area_px", "white_ref_r", "white_ref_g", "white_ref_b", "file", "raw_file", "timestamp"]


def save_session(root, frame, board, reference, overlay, cards, calib, labels):
    """Write everything; returns (folder, rows). `labels` maps side -> class name ("+" -> "poisonous")."""
    now = datetime.now()
    name = now.strftime("%Y-%m-%d_%H%M%S")
    folder = Path(root) / "sessions" / name
    for lab in labels.values():
        (folder / "cards" / lab).mkdir(parents=True, exist_ok=True)
        (folder / "raw" / lab).mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(folder / "frame.png"), frame)
    cv2.imwrite(str(folder / "board.png"), board)
    cv2.imwrite(str(folder / "reference.png"), reference)
    cv2.imwrite(str(folder / "overlay.png"), overlay)

    bh, bw = board.shape[:2]
    ordered = sorted(cards, key=lambda c: (c.side, c.center[1], c.center[0]))   # reading order per side
    rows = []
    for i, c in enumerate(ordered, 1):
        cid, label = f"c{i:03d}", labels[c.side]
        raw = crop_raw(board, c)
        card_img = letterbox(crop_inner(raw))
        f_card, f_raw = f"cards/{label}/{cid}.png", f"raw/{label}/{cid}.png"
        cv2.imwrite(str(folder / f_card), card_img)
        cv2.imwrite(str(folder / f_raw), raw)
        wr = white_reference(raw) or ("", "", "")
        rows.append({
            "card_id": cid, "session": name, "side": c.side, "label": label,
            "x_norm": round(c.center[0] / bw, 4), "y_norm": round(c.center[1] / bh, 4),
            "angle": round(c.angle, 1), "long_px": round(c.size[0], 1), "short_px": round(c.size[1], 1),
            "aspect": round(c.size[0] / c.size[1], 3), "area_px": round(c.area),
            "white_ref_r": wr[0], "white_ref_g": wr[1], "white_ref_b": wr[2],
            "file": f_card, "raw_file": f_raw, "timestamp": now.isoformat(timespec="seconds"),
        })

    with open(folder / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    info = {"session": name, "board_size": list(cfg.BOARD_SIZE),
            "calibration": {"H_cam_to_norm": calib.H.tolist(), "cam_size": list(calib.cam_size), "rms": calib.rms},
            "cards": {lab: sum(1 for r in rows if r["label"] == lab) for lab in labels.values()}}
    (folder / "session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return folder, rows



def load_cards(data_dir=None):
    """Load every saved card image of every session.

    Returns (images, labels, names): BGR arrays, "poisonous"/"safe", and "<session>/<label>/<file>".
    Sessions are read in alphabetical (= chronological) order.
    """
    root = Path(data_dir) if data_dir else Path(cfg.DATA_DIR)
    sessions = root / "sessions" if (root / "sessions").is_dir() else root
    images, labels, names = [], [], []
    for sdir in sorted(p for p in sessions.iterdir() if p.is_dir()) if sessions.is_dir() else []:
        for label in ("safe", "poisonous"):
            for f in sorted((sdir / "cards" / label).glob("*.png")):
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    names.append(f"{sdir.name}/{label}/{f.name}")
    return images, labels, names