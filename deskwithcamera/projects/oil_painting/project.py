"""Oil painting"""

TITLE = "Oil painting"
DESCRIPTION = "Put a printed photo on the table: its oil-painted version is projected next to it."
INCLUDE_TOOLS = True      # also offer the shared tools (test pattern, calibration, text)


def modules():
    # import inside the function so that listing projects stays fast
    from projects.oil_painting.main_module import OilPainting
    return [OilPainting()]
