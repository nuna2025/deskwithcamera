"""Photo to cartoon"""

TITLE = "Photo to cartoon"
DESCRIPTION = "TODO: one-line description of this project."
INCLUDE_TOOLS = True      # also offer the shared tools (test pattern, calibration, text)


def modules():
    # import inside the function so that listing projects stays fast
    from projects.cartoon_style.main_module import MainModule
    return [MainModule()]
