"""Finger counting"""

TITLE = "Finger counting"
DESCRIPTION = "Show one or two hands over the table: the number of stretched fingers is recognized and shown."
INCLUDE_TOOLS = True      # also offer the shared tools (test pattern, calibration, text)


def modules():
    # import inside the function so that listing projects stays fast
    from projects.finger_counting.main_module import FingerCounting
    return [FingerCounting()]
