"""Camera access + live ON/OFF status.

A background thread keeps reading frames, and is_on is derived from whether
a frame arrived recently. Without OpenCV or without a camera it simply
stays OFF.
"""
import sys
import threading
import time

import config

try:
    import cv2
except ImportError:          # the program must also run without a camera
    cv2 = None


class Camera:
    def __init__(self, index=None):
        self.index = config.CAMERA_INDEX if index is None else index
        self._cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._last_ok = 0.0

    @property
    def is_on(self):
        return self._running and (time.time() - self._last_ok) < config.CAMERA_STALE_SECONDS

    @property
    def running(self):
        return self._running

    def latest(self):
        """Most recent frame (BGR numpy array) or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def start(self):
        if self._running:
            return
        if cv2 is None:
            print("[camera] OpenCV is missing: pip install opencv-contrib-python")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._frame = None

    def toggle(self):
        self.stop() if self._running else self.start()

    @staticmethod
    def _backends():
        """Which OpenCV backends to try (Windows virtual webcams often need DSHOW or MSMF)."""
        if sys.platform != "win32":
            return [cv2.CAP_ANY]
        table = {"dshow": [cv2.CAP_DSHOW], "msmf": [cv2.CAP_MSMF],
                 "auto": [cv2.CAP_DSHOW, cv2.CAP_MSMF]}
        return table.get(str(config.CAMERA_BACKEND).lower(), table["auto"])

    def _open(self):
        for backend in self._backends():
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                w, h = config.CAMERA_RESOLUTION
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                return cap
            cap.release()
        return None

    def _loop(self):
        while self._running:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    time.sleep(0.5)          # look for the camera again
                    continue
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                self._last_ok = time.time()
            else:
                self._cap.release()
                self._cap = None
                time.sleep(0.2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None