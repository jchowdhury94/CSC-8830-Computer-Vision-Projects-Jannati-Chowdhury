"""Per-wall viewpoint count (2 or 3 stereo viewpoints) -- rules and wording (NO Streamlit import).

* Two viewpoints is the default and the original, validated workflow of BOTH walls.
* Three viewpoints exists for Wall B (room length) only.  It follows the offline three-viewpoint reconstruction that was validated for Wall B.
* Wall A also needs the room HEIGHT.  The Wall B reconstruction measures a horizontal distance between two floor corners; it contains no height geometry, there is no annotation scheme for a three-viewpoint
  Wall A, and nothing has been validated on real data.  So Wall A stays at two viewpoints: the option is shown, disabled and labelled "Not yet validated" (and the service refuses it too).
"""
from __future__ import annotations

import room_videos as RV

DEFAULT_COUNT = 2
VALID_COUNTS = (2, 3)
THREE_VIEW_WALLS = ("B",)                       # walls that may use three viewpoints
NOT_VALIDATED = "Not yet validated"
OPTION_LABEL = {2: "2 viewpoints (4 videos)", 3: "3 viewpoints (6 videos)"}
WALL_A_THREE_MISSING = ("Three viewpoints for Wall A is not available. Wall A also measures the room height, and the three-viewpoint method has only been validated for the length of Wall B. "
                        "Missing for Wall A: a three-viewpoint height reconstruction, an annotation scheme for it, and validation on real recordings.")
WALL_B_THREE_HELP = ("Three viewpoints (left end, middle, right end of the wall) can measure a longer wall than two. It is an experimental method that was validated on one room; "
                     "its result is always shown as approximate, with its quality checks.")
CHANGE_WARNING = "Changing the number of viewpoints discards this wall's uploaded videos, frame selections, boundaries and result. The other wall and the calibration are kept."
ROLE = {1: "left end", 2: "middle", 3: "right end"}


def allowed(wall: str, n: int) -> bool:
    return n == 2 or (n == 3 and wall in THREE_VIEW_WALLS)


def refusal(wall: str, n: int) -> str | None:
    """None when ``n`` viewpoints may be used on ``wall``; otherwise the visitor-safe reason."""
    if n not in VALID_COUNTS:
        return "A wall uses 2 or 3 stereo viewpoints."
    if n == 3 and wall not in THREE_VIEW_WALLS:
        return WALL_A_THREE_MISSING
    return None


def n_videos(n: int) -> int:
    return 2 * n


def count_word(n: int) -> str:
    return RV.COUNT_WORDS[n_videos(n)]
