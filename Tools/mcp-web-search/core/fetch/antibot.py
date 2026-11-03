# Copyright NEXTGGTECH. Elastic License 2.0.

# Weak anti-bot phrases: at least two distinct ones must appear in the page head to count as a wall.
_ANTIBOT_MARKERS = (
    "antibot", "challenge", "captcha", "cf-browser-verification",
    "ray id", "just a moment", "checking your browser", "please wait",
    "enable javascript", "ddos-guard", "robot or human",
)

# Strong phrases: a single one in the page head is enough to treat the page as a JavaScript wall.
_ANTIBOT_SINGLE = (
    "your browser does not support javascript",
    "javascript is required",
    "please enable javascript",
    "this site requires javascript",
    "you need to enable javascript",
)


# Return True when the response body looks like an anti-bot wall. Only the first 2000
# characters are inspected, since challenge pages announce themselves at the top.
def is_antibot(text: str) -> bool:
    t = text[:2_000].lower()

    # One strong phrase is conclusive; otherwise require two weak markers to avoid false positives.
    if any(m in t for m in _ANTIBOT_SINGLE):
        return True
    return sum(1 for m in _ANTIBOT_MARKERS if m in t) >= 2
