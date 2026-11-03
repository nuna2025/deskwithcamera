"""Project 1: mushroom classification (+ = poisonous, - = safe)."""

TITLE = "Mushroom classification"
DESCRIPTION = "Sort cartoon mushroom cards into + (poisonous) and - (safe), learn a simple model, classify a new card."
INCLUDE_TOOLS = True      # also offer the shared tools (test pattern, calibration, text)


def modules():
    # import inside the function so that listing projects stays fast
    from projects.mushroom.card_capture import CardCapture
    from projects.mushroom.plus_minus_split import PlusMinusSplit
    return [CardCapture(), PlusMinusSplit()]