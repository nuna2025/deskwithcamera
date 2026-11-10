"""Calibration module: shows ArUco markers in the four corners of the frame; the camera
finds them and the module computes the camera pixel -> frame coordinate transform
(a homography).

 ┌──────────────────────────────────┐
 │ [#0]                        [#1] │   marker ids 0..3 (one per corner)
 │          Calibrated OK           │
 │ [#3]                        [#2] │
 └──────────────────────────────────┘
Key: R = calibrate again
"""
import numpy as np
import pygame

import config
from core.module import Module
from core.shapes import draw_text

try:
    import cv2
except ImportError:
    cv2 = None

MARKER_IDS = (0, 1, 2, 3)        # top left, top right, bottom right, bottom left
NEEDED_FRAMES = 12               # compute once all four markers were seen this many frames in a row
PREVIEW_WINDOW = "Camera view (Calibration)"


def marker_layout(w, h):
    """Marker positions inside the content area (w x h). id -> (x, y, side, pad)."""
    side = round(h * 0.22)
    pad = round(side * 0.18)                 # white margin around each marker (so it is detected on a black background)
    m = round(h * 0.03)
    x0, y0 = m + pad, m + pad
    x1, y1 = w - m - pad - side, h - m - pad - side
    return {0: (x0, y0, side, pad), 1: (x1, y0, side, pad),
            2: (x1, y1, side, pad), 3: (x0, y1, side, pad)}


class CalibrationModule(Module):
    title = "Calibration"
    show_home = False            # no HOME button here: it would cover marker 0

    def __init__(self):
        self.state = "search"            # "search" | "done"
        self.found = 0                   # number of markers currently visible
        self._samples = []
        self._cache = {}
        self._size = (0, 0)
        self._last_corners = None
        self._last_ids = None
        self._preview_on = config.CAMERA_PREVIEW
        self._dict = self._detector = None
        if cv2 is not None:
            aruco = cv2.aruco
            self._dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            if hasattr(aruco, "ArucoDetector"):
                self._detector = aruco.ArucoDetector(self._dict, aruco.DetectorParameters())

    # --- Lifecycle -----------------------------------------------------------
    def on_enter(self, app):
        self.state = "done" if app.calib.ok else "search"
        self._samples.clear()

    def on_exit(self, app):
        if cv2 is not None and self._preview_on:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    def handle_event(self, event, app):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.state = "search"
            self._samples.clear()

    # --- Marker images -------------------------------------------------------
    def _marker_surface(self, marker_id, side):
        key = (marker_id, side)
        if key not in self._cache:
            aruco = cv2.aruco
            if hasattr(aruco, "generateImageMarker"):
                g = aruco.generateImageMarker(self._dict, marker_id, side, borderBits=1)
            else:
                g = aruco.drawMarker(self._dict, marker_id, side, borderBits=1)
            rgb = np.stack([g, g, g], axis=-1).swapaxes(0, 1)
            self._cache[key] = pygame.surfarray.make_surface(np.ascontiguousarray(rgb))
        return self._cache[key]

    # --- Detection + computation ---------------------------------------------
    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict)
        self._last_corners, self._last_ids = corners, ids
        found = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in MARKER_IDS:
                    found[int(i)] = c.reshape(4, 2)
        return found

    def update(self, dt, app):
        if cv2 is None:
            return
        frame = app.camera.latest()
        if frame is None:
            self.found = 0
            return
        if self.state == "search":
            found = self._detect(frame)
            self.found = len(found)
            if len(found) == len(MARKER_IDS):
                self._samples.append(np.concatenate([found[i] for i in MARKER_IDS]))
                if len(self._samples) >= NEEDED_FRAMES:
                    self._finish(frame, app)
            else:
                self._samples.clear()
        if self._preview_on:
            self._show_preview(frame, app)

    def _finish(self, frame, app):
        cam_pts = np.median(np.array(self._samples), axis=0).astype(np.float32)   # 16 x 2
        w, h = self._size
        if w == 0:
            return
        norm_pts = []
        layout = marker_layout(w, h)
        for i in MARKER_IDS:
            x, y, s, _ = layout[i]
            for cx, cy in ((x, y), (x + s, y), (x + s, y + s), (x, y + s)):
                norm_pts.append((cx / w, cy / h))
        norm_pts = np.array(norm_pts, dtype=np.float32)

        H, _ = cv2.findHomography(cam_pts, norm_pts, 0)      # least squares (16 points)
        if H is None:
            self._samples.clear()
            return
        proj = cv2.perspectiveTransform(cam_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = (proj - norm_pts) * np.array([w, h])
        rms = float(np.sqrt((err ** 2).sum(axis=1).mean()))

        app.calib.set(H, (frame.shape[1], frame.shape[0]), rms)
        app.calib.save()
        self.state = "done"
        self._samples.clear()
        print(f"[calibration] OK  rms = {rms:.2f} px  (saved to: {app.calib.path})")

    # --- Preview (on the laptop screen) --------------------------------------
    def preview_image(self, frame, app):
        img = frame.copy()
        if self._last_ids is not None and len(self._last_ids):
            cv2.aruco.drawDetectedMarkers(img, self._last_corners, self._last_ids)
        if app.calib.ok:       # draw the computed frame interior back as a green rectangle to verify it
            quad = [app.calib.norm_to_cam(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
            cv2.polylines(img, [np.array(quad, dtype=np.int32)], True, (0, 255, 0), 3)
        txt = f"markers {self.found}/4" if self.state == "search" else f"calibrated  rms {app.calib.rms:.1f}px"
        cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return img

    def _show_preview(self, frame, app):
        try:
            cv2.imshow(PREVIEW_WINDOW, self.preview_image(frame, app))
            cv2.waitKey(1)
        except cv2.error:
            self._preview_on = False       # e.g. opencv-python-headless: continue without preview
            print("[calibration] Preview window not available (install opencv-contrib-python)")

    # --- Drawing -------------------------------------------------------------
    def draw(self, surf, app):
        w, h = surf.get_size()
        self._size = (w, h)
        if cv2 is None:
            draw_text(surf, app.board.font(h * 0.06), "OpenCV missing: pip install opencv-contrib-python", (w // 2, h // 2))
            return
        for i, (x, y, side, pad) in marker_layout(w, h).items():
            pygame.draw.rect(surf, (255, 255, 255), (x - pad, y - pad, side + 2 * pad, side + 2 * pad))
            surf.blit(self._marker_surface(i, side), (x, y))

        big, small = app.board.font(h * 0.08), app.board.font(h * 0.045)
        if self.state == "done":
            draw_text(surf, big, "Calibrated OK", (w // 2, h // 2 - h * 0.05), config.CAM_ON_COLOR)
            draw_text(surf, small, f"RMS error: {app.calib.rms:.1f} px", (w // 2, h // 2 + h * 0.05))
            draw_text(surf, small, "R = redo", (w // 2, h // 2 + h * 0.11))
        elif not app.camera.is_on:
            draw_text(surf, big, "Cam OFF", (w // 2, h // 2 - h * 0.03), config.CAM_OFF_COLOR)
            draw_text(surf, small, "Camera not connected", (w // 2, h // 2 + h * 0.06))
        else:
            draw_text(surf, big, f"Searching markers {self.found}/4", (w // 2, h // 2 - h * 0.03))
            draw_text(surf, small, "Camera must see the whole frame", (w // 2, h // 2 + h * 0.06))
