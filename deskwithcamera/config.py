"""Global settings shared by the whole framework (every module uses these)."""

# --- Which project starts by default (folder name inside projects/) ----------
# Override at launch: python main.py --project cartoon_style
DEFAULT_PROJECT = "mushroom"

# --- Projection mode ---------------------------------------------------------
# You can also override this at launch: python main.py --profile projector
DISPLAY_PROFILE = "window"

PROFILES = {
    # Development / testing: normal window
    "window":      {"label": "Test window",             "monitor": 0, "fullscreen": False, "win_mode": None},
    # Extend 1: school projector (own screen + projector)
    "projector":   {"label": "Extend 1 / Projector",    "monitor": 1, "fullscreen": True,  "win_mode": "extend"},
    # Extend 2: laptop 2 at home (own screen + laptop 2)
    "laptop2":     {"label": "Extend 2 / Laptop 2",     "monitor": 1, "fullscreen": True,  "win_mode": "extend"},
    # Second screen only: own screen is off, the remaining screen becomes monitor 0
    "second_only": {"label": "Second screen only",      "monitor": 0, "fullscreen": True,  "win_mode": "external"},
}

# If True, the program switches the Windows Win+P mode itself
# (Windows only, uses DisplaySwitch.exe)
AUTO_WINDOWS_MODE = False

WINDOW_SIZE = (1280, 720)       # window size for the "window" profile

# --- Base frame --------------------------------------------------------------
ASPECT = (16, 9)                # frame aspect ratio (like a laptop screen)
LINE_WIDTH = 6                  # frame line thickness (pixels)
MARGIN = 0.03                   # margin to the screen edge (fraction)
HEADER_FONT_RATIO = 0.045       # header text size (relative to screen height)
HEADER_GAP = 10                 # gap between header and frame (pixels)

# --- Colors ------------------------------------------------------------------
BG_COLOR = (0, 0, 0)
LINE_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
CAM_ON_COLOR = (90, 255, 130)
CAM_OFF_COLOR = (255, 90, 90)

# --- Camera ------------------------------------------------------------------
CAMERA_ENABLED = True
CAMERA_INDEX = 0                # DroidCam (MSMF Nummer 0)
CAMERA_BACKEND = "msmf"         # "auto" | "dshow" | "msmf"
CAMERA_STALE_SECONDS = 1.0      # no new frame for this long -> shown as OFF
CAMERA_RESOLUTION = (1280, 720) # resolution requested from the camera (matters for calibration accuracy)
CAMERA_PREVIEW = True           # show a camera preview window on the laptop during calibration


