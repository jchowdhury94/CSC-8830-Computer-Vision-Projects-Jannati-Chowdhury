"""F8.31D-3B: the visitor-facing instruction text of the six guided calibration steps (plain language; no developer terms)."""
from __future__ import annotations

INSTRUCTIONS = {
    1: (
        "- **Two cameras with overlapping views** are required. Both must see the same scene at the same time.\n"
        "- **Keep LEFT and RIGHT consistent.** The camera on your left is always LEFT and the one on your right is always RIGHT, in every video you record and upload.\n"
        "- **Mount the cameras side by side on a stable rig.**\n"
        "- **Keep their relative positions fixed** during calibration and also later during room measurement. If you move one camera, calibrate again.\n"
        "- **Use the same resolution, orientation, zoom and focus settings** on both cameras, and do not change them afterwards.\n"
        "- **Measure the distance between the two lens centres** with a ruler if you can. This distance is called the **stereo baseline**. Enter it below (leave 0 if you did not measure it)."
    ),
    2: (
        "- Use a **flat, rigid, high-contrast checkerboard** (printed on a stiff board, not on loose paper).\n"
        "- **Count the inner corners, not the printed squares.** Inner corners are the points where four squares meet.\n"
        "- **Measure the side of one square in millimetres.**\n"
        "- The board must be **fully visible in both cameras**.\n"
        "- **Avoid glare and reflections** on the board.\n\n"
        "*Example: a board with 9 × 7 squares has 8 × 6 inner corners.* Enter the corner counts and the square size below."
    ),
    3: (
        "- **Record the LEFT and RIGHT videos at the same time.**\n"
        "- **Keep the cameras fixed** while recording.\n"
        "- **Move or tilt the checkerboard** through many different positions and angles.\n"
        "- Show the board **near the centre and near the edges** of the picture.\n"
        "- Include **different distances** from the cameras.\n"
        "- **Keep the whole board visible in both recordings.**\n"
        "- **Avoid strong motion blur and glare.**"
    ),
    4: (
        "- Upload the **LEFT camera video** in the left box and the **RIGHT camera video** in the right box.\n"
        "- **The box decides which camera a video belongs to, never the file name.** Check that you did not swap them.\n"
        "- Each upload is checked as it arrives. This step is complete only when **both** videos have been received and verified.\n"
        "- If an upload fails or is no longer available, upload that video again."
    ),
    5: (
        "- Check the summary of the settings you entered in **Step 1** (stereo baseline) and **Step 2** (checkerboard). To change a value, go back to that step; the summary updates at once, "
        "and you will need to confirm that step again.\n"
        "- The **stereo baseline** is the distance between your two lens centres. It is **not** the distance from the cameras to the object or wall you will measure later. "
        "If it was not measured, the distance estimated by the calibration is used.\n"
        "- Tick the confirmation and press **Start calibration**. It can take a long time. A private recovery code is shown so you can find the result again after a page refresh."
    ),
    6: (
        "- **Quality rating.** *Good* means the calibration passed all checks. *Questionable but usable* means it works but has concerns you should read. *Failed* means it cannot be used.\n"
        "- **Stereo reprojection error** is how far, in pixels, the calibration's predictions miss the real corner positions. Smaller is better; roughly under 1 px is good.\n"
        "- **Baseline** is the distance between your cameras that will be used to give measurements their real-world scale.\n"
        "- **Accept** a calibration that is rated Good, or one you have read and are comfortable with. **Retry** (record again, with more varied board positions) if it failed or if the error is large.\n"
        "- After you accept, the calibration is kept for this session and you can continue to the room measurement. Starting another calibration replaces it."
    ),
}

ROOM_LIMITATION_NOTE = "Limitation: Accurate wall measurement depends on clear wall boundaries, visible floor texture, and consistent camera calibration. Results may be unreliable if these conditions are not met."

ROOM_INSTRUCTIONS = {
    1: (
        "- **Use the same LEFT and RIGHT cameras you used for calibration.**\n"
        "- **Keep the same camera settings** (resolution, orientation, frame rate, zoom, focus, stabilisation) and the **same stereo rig geometry** as in calibration.\n"
        "- **Keep LEFT and RIGHT consistent.** The camera on your left is always LEFT and the one on your right is always RIGHT, in every video.\n"
        "- **Choose a wall whose corner boundaries are visible**, with a floor that has visible texture where possible.\n"
        "- The measurement needs **usable floor features** (not a single flat colour or a mirror-like surface) and **visible wall boundaries**.\n"
        "- **Do not move the two cameras relative to each other** on the rig. If you do, calibrate again.\n"
        "- **Keep the room still** while you record: no people or objects moving."
    ),
    2: (
        "- Record **two different viewpoints** of the wall: **Viewpoint 1** and **Viewpoint 2**.\n"
        "- **Each viewpoint needs a LEFT and a RIGHT recording**, so you finish with **exactly four videos**.\n"
        "- **Keep the camera rig unchanged** at both viewpoints. Only move the whole rig, never one camera on it.\n"
        "- The two viewpoints should **see overlapping floor texture**, so they can be matched and joined.\n"
        "- **Both cameras must see the end edge of the wall**, and the **right camera must see where the wall meets the ceiling**.\n"
        "- **The four videos do not have to be synchronised.** You will choose the frames for each video separately in a later step."
    ),
    3: (
        "Each box takes one recording. **The box decides which camera a video belongs to, never the file name.**\n\n"
        "- **Viewpoint 1 — LEFT camera:** the recording from your LEFT camera at Viewpoint 1.\n"
        "- **Viewpoint 1 — RIGHT camera:** the recording from your RIGHT camera at Viewpoint 1.\n"
        "- **Viewpoint 2 — LEFT camera:** the recording from your LEFT camera at Viewpoint 2.\n"
        "- **Viewpoint 2 — RIGHT camera:** the recording from your RIGHT camera at Viewpoint 2.\n\n"
        "Each upload is checked as it arrives, and only a video that passes the checks counts as accepted. "
        "If a video is rejected or expires, upload that video again. **Continue** becomes available when all four are accepted."
    ),
}
