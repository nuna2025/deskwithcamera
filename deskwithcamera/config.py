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
LINE_WIDTH = 8                  # frame line thickness (pixels)
OUTER_LINE_WIDTH = 5            # extra blue line wrapped around the OUTSIDE of the frame (pixels)
CORNER_RADIUS = 50              # rounded frame corners (pixels, 0 = sharp)
MARGIN = 0.03                   # margin to the screen edge (fraction)
HEADER_FONT_RATIO = 0.045       # header text size (relative to screen height)
HEADER_GAP = 10                 # gap between header and frame (pixels)
SHOW_CAM_STATUS = True          # show "Cam: ON/OFF" at the top left (False = hide it)

# --- Fonts -------------------------------------------------------------------
# Files live in assets/fonts/Hahmlet-<Weight>.ttf
# Weights: Thin, ExtraLight, Light, Regular, Medium, SemiBold, Bold, ExtraBold, Black
FONT_DIR = "assets/fonts"
FONT_FAMILY = "Hahmlet"
TEXT_WEIGHT = "Medium"          # default weight of all projected text
HEADER_WEIGHT = "ExtraBold"     # "Cam: ON/OFF" and the module title (try "Bold" as well)

# --- Colors (RGB) ------------------------------------------------------------
BG_COLOR = (0x07, 0x19, 0x34)           # #071934  background (instead of black)
FRAME_COLOR = (0xFF, 0xF5, 0xA0)        # #fff5a0  frame line
FRAME_OUTER_COLOR = (0x00, 0x00, 0xFF)  # #0000ff  thin line around the outside of the frame
LINE_COLOR = (0xFF, 0xF5, 0xA0)         # #fff5a0  all other lines/shapes (center line, + / - ...)
TEXT_COLOR = (0xFF, 0xF5, 0xA0)         # #fff5a0  all text
CAM_LABEL_COLOR = (0xFF, 0x00, 0x00)    # #ff0000  the word "Cam:"
CAM_ON_COLOR = (0xCD, 0xDC, 0x39)       # #CDDC39  "ON", card counts, "Calibrated OK"
CAM_OFF_COLOR = (255, 90, 90)

# --- Camera ------------------------------------------------------------------
CAMERA_ENABLED = True
CAMERA_INDEX = 0                # DroidCam (MSMF Nummer 0)
CAMERA_BACKEND = "msmf"         # "auto" | "dshow" | "msmf"
CAMERA_STALE_SECONDS = 1.0      # no new frame for this long -> shown as OFF
CAMERA_RESOLUTION = (1280, 720) # resolution requested from the camera (matters for calibration accuracy)
CAMERA_PREVIEW = True           # show a camera preview window on the laptop during calibration


