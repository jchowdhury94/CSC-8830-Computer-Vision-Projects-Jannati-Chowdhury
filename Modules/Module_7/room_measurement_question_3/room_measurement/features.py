"""Stereo feature detection and matching (SIFT, ratio test + mutual consistency), and cross-view matches."""
import cv2
import numpy as np

SIFT_NFEATURES = 12000
RATIO_STEREO = 0.75    # Air<->Pro within a view
RATIO_CROSS = 0.85     # Air<->Air across the two views

def match_set(frames, ratio=RATIO_STEREO, ratio_cross=RATIO_CROSS, nfeatures=SIFT_NFEATURES, contrast=0.04, seed=0):
    """frames: dict with grayscale uint8 images AL, PL, AR, PR (distorted, as decoded).
    Returns dict A1,B1 (view-1 Air/Pro pixels), M1 (keypoint index pairs), A2,B2,M2 (view 2), MX (AL<->AR index pairs)."""
    cv2.setRNGSeed(seed); sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast); kp = {}; ds = {}
    for k in ("AL", "PL", "AR", "PR"):
        kk, d = sift.detectAndCompute(frames[k], None); kp[k] = np.array([p.pt for p in kk], np.float32); ds[k] = d
    bf = cv2.BFMatcher()
    def match(a, b, rt):
        m = bf.knnMatch(ds[a], ds[b], k=2); good = [x[0] for x in m if len(x) == 2 and x[0].distance < rt * x[1].distance]
        back = {x[0].queryIdx: x[0].trainIdx for x in bf.knnMatch(ds[b], ds[a], k=1)}
        return np.array([(x.queryIdx, x.trainIdx) for x in good if back.get(x.trainIdx) == x.queryIdx])
    M1, M2, MX = match("AL", "PL", ratio), match("AR", "PR", ratio), match("AL", "AR", ratio_cross)
    return dict(A1=kp["AL"][M1[:, 0]], B1=kp["PL"][M1[:, 1]], M1=M1, A2=kp["AR"][M2[:, 0]], B2=kp["PR"][M2[:, 1]], M2=M2, MX=MX)

def build_tracks(F):
    """Cross-view four-view tracks: a left-camera keypoint matched to the Pro in view 1, to the Air in view 2 and on to
    the Pro in view 2.  Returns (N,4,2) pixels ordered AL, PL, AR, PR."""
    d1 = {int(q): i for i, q in enumerate(F["M1"][:, 0])}; d2 = {int(q): i for i, q in enumerate(F["M2"][:, 0])}; out = []
    for q, v in F["MX"]:
        if int(q) in d1 and int(v) in d2:
            i, j = d1[int(q)], d2[int(v)]; out.append([F["A1"][i], F["B1"][i], F["A2"][j], F["B2"][j]])
    return np.array(out, np.float64).reshape(-1, 4, 2)
