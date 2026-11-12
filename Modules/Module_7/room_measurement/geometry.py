"""Pure geometry: coordinates, stereo triangulation, robust weights, planes, rays, rigid registration.

Coordinate systems (all lengths in mm):
  * pixel (x right, y down) in the 2160x3840 video frame, lens-distorted unless stated otherwise;
  * normalised camera coordinates (x/z, y/z) after removing distortion with the calibration;
  * LEFT-camera frame: origin at the left (Air) optical centre, x right, y down, z forward; this is the frame of every
    triangulated point of a view.  The right camera sits at X_right = R @ X_left + T (|T| = physical baseline).
  * View 1 (V1) and view 2 (V2) are two rig placements; ``register`` maps V1 left-camera coordinates onto V2's.
"""
import cv2
import numpy as np

F_PX = 3000.0   # scale turning normalised-coordinate residuals into approximate pixels

# ---------------------------------------------------------------- pose parameterisation
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def pose_from_params(p5, baseline_mm):
    """p5 = (Rodrigues rx, ry, rz, theta, phi); translation = baseline * unit(theta, phi).  Returns R (3x3), T (3,)."""
    R = cv2.Rodrigues(np.asarray(p5[:3], np.float64))[0]
    T = baseline_mm * np.array([np.sin(p5[3]) * np.cos(p5[4]), np.sin(p5[3]) * np.sin(p5[4]), np.cos(p5[3])])
    return R, T

def params_from_pose(R, T):
    t = np.asarray(T, np.float64).reshape(3); t = t / np.linalg.norm(t)
    return np.r_[cv2.Rodrigues(R)[0].ravel(), np.arccos(t[2]), np.arctan2(t[1], t[0])]

def yaw_deg(p5):
    """Rodrigues y component of the stereo rotation, degrees (the weakly observable 'relative yaw')."""
    return float(np.degrees(p5[1]))

def rotate_yaw(p5, delta_deg, baseline_mm):
    """Apply a rigid yaw offset Ry(delta) to the whole pose (used only by sensitivity analyses)."""
    Rz = cv2.Rodrigues(np.array([0, np.radians(delta_deg), 0]))[0]; R, T = pose_from_params(p5, baseline_mm); return params_from_pose(Rz @ R, Rz @ T)

# ---------------------------------------------------------------- camera model
def undistort_normalized(px, K, D):
    return cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2)

def undistort_pixels(px, K, D):
    return cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

def triangulate(cal, R, T, a_px, b_px):
    """Linear (DLT) triangulation of matched distorted pixels a (left) / b (right) into the left-camera frame."""
    n1 = undistort_normalized(a_px, cal.K_left, cal.D_left).T; n2 = undistort_normalized(b_px, cal.K_right, cal.D_right).T
    X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, np.asarray(T).reshape(3, 1)]), n1, n2)
    return (X[:3] / X[3]).T

def reprojection_errors(cal, R, T, X, a_px, b_px):
    p1 = cv2.projectPoints(X, np.zeros(3), np.zeros(3), cal.K_left, cal.D_left)[0].reshape(-1, 2)
    p2 = cv2.projectPoints(X, cv2.Rodrigues(R)[0], np.asarray(T).reshape(3), cal.K_right, cal.D_right)[0].reshape(-1, 2)
    return np.linalg.norm(p1 - a_px, axis=1), np.linalg.norm(p2 - b_px, axis=1)

def epipolar_distances(cal, R, T, a_px, b_px):
    """Point-to-epipolar-line distance in the right image (undistorted pixels)."""
    ua, ub = undistort_pixels(a_px, cal.K_left, cal.D_left), undistort_pixels(b_px, cal.K_right, cal.D_right)
    E = skew(np.asarray(T).reshape(3)) @ R; F = np.linalg.inv(cal.K_right).T @ E @ np.linalg.inv(cal.K_left)
    l = np.c_[ua, np.ones(len(ua))] @ F.T
    return np.abs((l * np.c_[ub, np.ones(len(ub))]).sum(1)) / np.hypot(l[:, 0], l[:, 1])

def sampson_residual(p5, n1, n2, baseline_mm):
    """Signed Sampson epipolar distance (approx. px) for normalised correspondences n1 (left) / n2 (right)."""
    R, T = pose_from_params(p5, baseline_mm); E = skew(T / baseline_mm) @ R
    x1 = np.c_[n1, np.ones(len(n1))]; x2 = np.c_[n2, np.ones(len(n2))]
    Ex1 = x1 @ E.T; Etx2 = x2 @ E; e = (Ex1 * x2).sum(1)
    den = np.sqrt(Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2)
    return F_PX * e / np.maximum(den, 1e-12)

# ---------------------------------------------------------------- robust loss (Tukey biweight)
TUKEY_C = 4.685
def tukey_weight(r, s, c=TUKEY_C):
    z = np.abs(r) / (c * s); return np.where(z < 1, (1 - z ** 2) ** 2, 0.0)
def tukey_rho(r, s, c=TUKEY_C):
    z = np.abs(r) / (c * s); return (s * c) ** 2 / 6 * np.where(z < 1, 1 - (1 - z ** 2) ** 3, 1.0)
def mad_scale(r, floor):
    return max(1.4826 * float(np.median(np.abs(r))), floor)

# ---------------------------------------------------------------- planes, lines, rays
def fit_plane_irls(P, iters=30):
    """Deterministic robust plane n.x + d = 0 (|n|=1, d>=0): PCA start, Tukey IRLS with MAD scale (floor 0.5 mm)."""
    P = np.asarray(P, np.float64)
    if len(P) < 3: raise ValueError(f"plane fit needs at least 3 points, got {len(P)}")
    c = P.mean(0); n = np.linalg.svd(P - c)[2][2]; d = -n @ c
    for _ in range(iters):
        r = P @ n + d; s = max(1.4826 * np.median(np.abs(r)), 0.5); w = tukey_weight(r, s, TUKEY_C)
        c = (P * w[:, None]).sum(0) / w.sum(); C = ((P - c) * w[:, None]).T @ (P - c); n = np.linalg.eigh(C)[1][:, 0]; d = -n @ c
    if d < 0: n, d = -n, -d
    return n, d

def fit_floor_plane(X, left_px, y_min=2600):
    """Floor = robust plane through triangulated points whose LEFT-image row is below y_min (lower part of the frame)."""
    return fit_plane_irls(np.asarray(X)[np.asarray(left_px)[:, 1] > y_min])

def line_to_h(line):
    """Homogeneous image-line coefficients (l . [x y 1] = 0) from a fitted-line dict (kind 'v': x=a*y+b; 'h': y=a*x+b)."""
    return np.array([-1, line["a"], line["b"]]) if line["kind"] == "v" else np.array([line["a"], -1, line["b"]])

def line_intersection(l1, l2):
    p = np.cross(line_to_h(l1), line_to_h(l2)); return p[:2] / p[2]

def camera_ray(cal, R, T, cam, px_undist):
    """(origin, unit direction) in the left-camera frame of the ray through an UNDISTORTED pixel of camera 'L' or 'R'."""
    K = cal.K_left if cam == "L" else cal.K_right; dv = np.linalg.inv(K) @ np.array([px_undist[0], px_undist[1], 1.0])
    if cam == "L": return np.zeros(3), dv / np.linalg.norm(dv)
    T = np.asarray(T).reshape(3); return -R.T @ T, R.T @ dv / np.linalg.norm(dv)

def ray_plane_intersection(origin, direction, n, d):
    lam = -(n @ origin + d) / (n @ direction); return origin + lam * direction

def corner_line_on_floor(cal, R, T, line_left, line_right, floor):
    """Bottom corner: intersect the two back-projected vertical-edge planes (one per camera) with the floor plane.
    line_* are homogeneous undistorted-image lines.  Returns (corner point B, unit direction u of the 3D edge line)."""
    nL = cal.K_left.T @ line_left; nR = R.T @ (cal.K_right.T @ line_right); C2 = -R.T @ np.asarray(T).reshape(3); n, d = floor
    B = np.linalg.solve(np.array([nL, nR, n]), np.array([0, nR @ C2, -d])); u = np.cross(nL, nR); u /= np.linalg.norm(u); return B, u

def top_point_on_vertical(cal, R, T, px_top, B, up):
    """Point B + t*up on the vertical through B closest to the Pro (right-camera) ray through px_top.
    Returns (t = height above the floor along 'up', skew distance between ray and vertical)."""
    c, dv = camera_ray(cal, R, T, "R", px_top); w0 = c - B; a = dv @ dv; b = dv @ up; cc = up @ up; d_ = dv @ w0; e = up @ w0
    den = a * cc - b * b; s = (b * e - cc * d_) / den; t = (a * e - b * d_) / den
    return t, float(np.linalg.norm((c + s * dv) - (B + t * up)))

def umeyama(P, Q):
    """Rigid (no scale) least-squares fit Q ~ R P + t."""
    mp, mq = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp)); dd = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, dd]) @ Vt; return R, mq - R @ mp

def umeyama_irls(P, Q, iters=40):
    """Deterministic robust rigid fit (Tukey IRLS on 3D residual, MAD scale floor 5 mm)."""
    w = np.ones(len(P))
    for _ in range(iters):
        mp, mq = (P * w[:, None]).sum(0) / w.sum(), (Q * w[:, None]).sum(0) / w.sum()
        U, S, Vt = np.linalg.svd(((Q - mq) * w[:, None]).T @ (P - mp)); d = np.sign(np.linalg.det(U @ Vt)); R = U @ np.diag([1, 1, d]) @ Vt; t = mq - R @ mp
        e = np.linalg.norm(Q - (P @ R.T + t), axis=1); wn = tukey_weight(e, max(1.4826 * np.median(e), 5.0))
        if np.abs(wn - w).max() < 1e-6: w = wn; break
        w = wn
    return R, t

def rigid_ransac(P, Q, rng, iters=1500, thr=40.0):
    """Seeded 3-point RANSAC rigid registration + inlier refit.  Returns R, t, inlier rms (mm), n inliers, n pairs."""
    best = None
    for _ in range(iters):
        s = rng.choice(len(P), 3, replace=False); R, t = umeyama(P[s], Q[s]); e = np.linalg.norm(Q - (P @ R.T + t), axis=1); n = (e < thr).sum()
        if best is None or n > best[0]: best = (n, e < thr)
    inl = best[1]; R, t = umeyama(P[inl], Q[inl]); e = np.linalg.norm(Q[inl] - (P[inl] @ R.T + t), axis=1)
    return R, t, float(np.sqrt((e ** 2).mean())), int(inl.sum()), len(P)
