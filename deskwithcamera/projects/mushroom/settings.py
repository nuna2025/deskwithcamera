"""Settings of the mushroom project (project-local; the shared settings are in the root config.py)."""
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"             # sessions are saved to data/sessions/<timestamp>/

# --- Top-down view of the projected area --------------------------------------
BOARD_SIZE = (1280, 720)             # working resolution (w, h) of the rectified view
HUD_FRAC = 0.15                      # top 15% = symbols, counts, status text; cards go BELOW this line

# --- Card detection -----------------------------------------------------------
MIN_SHORT_FRAC = 0.05                # card short side >= 5% of the board width, otherwise it is noise
MAX_LONG_FRAC = 0.30                 # card long side  <= 30% of the board width, bigger = cards touching
MIN_FILL = 0.75                      # blob area / bounding-rectangle area (low = not a rectangle)
OPEN_FRAC = 0.015                    # thin structures narrower than 1.5% of the board width are erased (frame / center line)
EDGE_MARGIN_FRAC = 0.012             # ignore a strip along the board edge (where the projected frame line is)
OTSU_SCALE = 0.4                     # threshold = this fraction of Otsu (lower = dark artwork is kept, more noise)
MIN_DIFF = 35                        # smallest brightness difference to the empty table that counts
REFERENCE_FRAMES = 10                # frames averaged for the empty-table picture
STABLE_FRAMES = 8                    # cards must stay still this many frames in a row
STABLE_TOL_PX = 6                    # ... moving less than this (in board pixels)

# --- Saved card images ----------------------------------------------------------
CROP_INNER = 0.92                    # keep the inner 92% of a card (no card edge / table)
CARD_SIZE = 256                      # saved card images are CARD_SIZE x CARD_SIZE (aspect ratio kept, white padding)

# --- Classification model -------------------------------------------------------
MODEL_ID = 1                         # 1 = HSV + Linear SVM, 2 = MobileNetV2 + LogReg, 3 = HOG + Linear SVM