"""List screens and cameras, so you know which numbers to use.

Usage (from the project root):  python scripts/list_devices.py

- Screens: the number is the monitor index (config.py -> PROFILES -> "monitor").
- Cameras: the number is the camera index (python main.py --camera N, or config.CAMERA_INDEX).
  On Windows every camera is tried with two different backends (DSHOW and MSMF),
  because some virtual webcams (DroidCam, Iriun, Camo ...) only work with one of them.
  For every working camera a snapshot camera_N_BACKEND.jpg is saved in the current
  folder, so you can open it and see which camera is the phone and which the laptop webcam.
"""
import sys
import time

MAX_CAMERAS = 6
WARMUP_SECONDS = 3.0      # virtual webcams (DroidCam ...) need a moment before real images arrive


def list_screens():
    try:
        import pygame
    except ImportError:
        print("pygame is missing: pip install pygame")
        return
    pygame.init()
    print("Screens:")
    for i, (w, h) in enumerate(pygame.display.get_desktop_sizes()):
        print(f"  monitor {i}: {w} x {h}")
    pygame.quit()


def backends(cv2):
    if sys.platform == "win32":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    return [("default", cv2.CAP_ANY)]


def looks_blank(frame):
    """True for (almost) uniform images, e.g. the plain green frame some virtual webcams send at start."""
    # spread of pixel values within each color channel (a plain green frame has none)
    return float(frame.reshape(-1, frame.shape[-1]).std(axis=0).max()) < 4.0


def grab(cv2, idx, backend):
    """Open the camera and read for a few seconds until a real (non-blank) image arrives.

    Returns (frame, opened, blank): frame is the best image or None,
    blank is True if only uniform images (e.g. all green) were received.
    """
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap.release()
        return None, False, False
    last = None
    good = None
    start = time.time()
    while time.time() - start < WARMUP_SECONDS:
        ok, f = cap.read()
        if ok:
            last = f
            if not looks_blank(f):
                good = f
                if time.time() - start > 1.0:     # real image and settled -> done
                    break
        else:
            time.sleep(0.05)
    cap.release()
    if good is not None:
        return good, True, False
    return last, True, last is not None


def list_cameras():
    try:
        import cv2
    except ImportError:
        print("OpenCV is missing: pip install opencv-contrib-python")
        return
    print("Cameras:")
    found = 0
    for idx in range(MAX_CAMERAS):
        for name, backend in backends(cv2):
            frame, opened, blank = grab(cv2, idx, backend)
            if not opened:
                continue
            if blank:
                h, w = frame.shape[:2]
                cv2.imwrite(f"camera_{idx}_{name}_BLANK.jpg", frame)
                print(f"  camera {idx} [{name}]: {w} x {h}   only a blank/uniform image "
                      f"(camera still starting, or used by another program - close main.py / OBS and retry)")
                continue
            if frame is None:
                print(f"  camera {idx} [{name}]: opens, but delivers no image "
                      f"(is another program, e.g. OBS, using it?)")
                continue
            h, w = frame.shape[:2]
            fname = f"camera_{idx}_{name}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  camera {idx} [{name}]: {w} x {h}   (snapshot: {fname})")
            found += 1
    if not found:
        print("  none found. Checklist:")
        print("   1. Close OBS and every other program that might use the camera.")
        print("   2. Is the DroidCam *Windows client* (virtual webcam) running and connected?")
        print("   3. Does the Windows 'Camera' app show the phone as a camera?")
        print("   4. Windows Settings > Privacy > Camera: allow desktop apps to access the camera.")


if __name__ == "__main__":
    list_screens()
    print()
    list_cameras()