"""Project 1: mushroom classification (poisonous = one side, safe = the other side)."""

TITLE = "Mushroom classification"
DESCRIPTION = "Sort cartoon mushroom cards into O / X, learn a simple model, classify a new card."
INCLUDE_TOOLS = True      # also offer the shared tools (test pattern, calibration, text)


def modules():
    # import inside the function so that listing projects stays fast
    from projects.mushroom.ox_split import OXSplit
    return [OXSplit()]
