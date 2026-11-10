"""Module 6 Part B visitor SfM backend tests (synthetic data only; no Streamlit, no browser)."""
import csv
import hashlib
import io
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from Modules.Module_6.Part_B import visitor_reconstruction as vr

ROOT = Path(__file__).resolve().parents[1]
PART_B = ROOT / "Modules" / "Module_6" / "Part_B"


# ----------------------------------------------------------------- synthetic data

def texture(size, seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for scale in (4, 8, 16, 32):
        noise = rng.random((size // scale + 1, size // scale + 1, 3)).astype(np.float32)
        img += cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC) * (scale ** .5)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    for _ in range(60):
        center = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.circle(img, center, int(rng.integers(3, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return (np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))


CUBE = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float)
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
TEXTURES = None


def cube_view(yaw, pitch, size=(900, 700), scale=230, distance=40.):
    """Textured cube seen by a distant (nearly orthographic) camera; painter's algorithm."""
    global TEXTURES
    if TEXTURES is None:
        TEXTURES = [texture(256, i) for i in range(6)]
    w, h = size
    canvas = np.full((h, w, 3), 90, np.uint8)
    cam = CUBE @ rotation(yaw, pitch).T
    for f in sorted(range(6), key=lambda i: -cam[list(FACES[i]), 2].mean()):
        q = cam[list(FACES[f])]
        if np.cross(q[1] - q[0], q[3] - q[0])[2] > 0:
            continue
        k = distance / (distance + q[:, 2])
        dst = np.float32([[w / 2 + scale * p[0] * kk, h / 2 + scale * p[1] * kk] for p, kk in zip(q, k)])
        s = TEXTURES[f].shape[0]
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]]), dst)
        warped = cv2.warpPerspective(TEXTURES[f], H, (w, h))
        mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), H, (w, h)) > 127
        canvas[mask] = warped[mask]
    return canvas


def png_bytes(rgb):
    ok, data = cv2.imencode(".png", rgb[:, :, ::-1])
    assert ok
    return data.tobytes()


def cube_files(count=4, step=0.17):
    return [(png_bytes(cube_view(-0.5 + step * i, 0.45)), f"view{i + 1}.png") for i in range(count)]


def orthographic_tracks(points, views=4, noise=0.0, seed=0):
    """Exact orthographic projections of 3D points with known camera rotations."""
    rng = np.random.default_rng(seed)
    tracks = []
    for v in range(views):
        R = rotation(0.4 * v - 0.6, 0.15 * v + 0.2)
        xy = (points @ R.T)[:, :2] * 100 + [500, 400] + [30 * v, -20 * v]
        tracks.append(xy + rng.normal(0, noise, xy.shape))
    return np.array(tracks)


def procrustes_error(estimate, truth):
    """Relative RMS error after the best similarity transform; reflection allowed (SfM is mirror-ambiguous)."""
    a, b = estimate - estimate.mean(0), truth - truth.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    aligned = a @ (u @ vt) * (s.sum() / np.sum(a ** 2))
    return float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(b ** 2, axis=1))))


def random_points(count=40, seed=3):
    return np.random.default_rng(seed).uniform(-1, 1, (count, 3))


def jpeg_with_orientation(size, orientation):
    image = Image.fromarray(texture(max(size), 1)[:size[1], :size[0]])
    exif = Image.Exif()
    exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


# ----------------------------------------------------------------- tests

class ImageDecodingTests(unittest.TestCase):
    def test_png_decodes_to_rgb(self):
        image = vr.decode_image(png_bytes(texture(400, 0)), "a.png")
        self.assertEqual(image.rgb.shape, (400, 400, 3))
        self.assertEqual(image.rgb.dtype, np.uint8)
        self.assertEqual(image.scale, 1.0)

    def test_exif_orientation_is_applied(self):
        # Stored 600 x 400 (landscape) with orientation 6 (rotate 90 deg) -> upright 400 x 600 portrait.
        image = vr.decode_image(jpeg_with_orientation((600, 400), 6), "a.jpg")
        self.assertEqual(image.orientation, 6)
        self.assertEqual(image.size, (400, 600))
        self.assertEqual(image.original_size, (400, 600))
        plain = vr.decode_image(jpeg_with_orientation((600, 400), 1), "b.jpg")
        self.assertEqual(plain.size, (600, 400))

    def test_large_images_are_downscaled(self):
        image = vr.decode_image(png_bytes(np.zeros((1000, 3000, 3), np.uint8)), "wide.png", long_side=1500)
        self.assertEqual(image.size, (1500, 500))
        self.assertEqual(image.original_size, (3000, 1000))
        self.assertAlmostEqual(image.scale, 0.5)

    def test_invalid_and_unsupported_files(self):
        with self.assertRaises(vr.ImageError):
            vr.decode_image(b"not an image at all", "x.jpg")
        gif = io.BytesIO()
        Image.new("RGB", (400, 400)).save(gif, "GIF")
        with self.assertRaises(vr.ImageError):
            vr.decode_image(gif.getvalue(), "x.jpg")
        with self.assertRaises(vr.ImageError):
            vr.decode_image(png_bytes(np.zeros((100, 100, 3), np.uint8)), "tiny.png")

    def test_pixel_budget(self):
        old = vr.MAX_PIXELS
        vr.MAX_PIXELS = 10_000
        try:
            with self.assertRaises(vr.ImageError):
                vr.decode_image(png_bytes(np.zeros((400, 400, 3), np.uint8)), "big.png")
        finally:
            vr.MAX_PIXELS = old

    @unittest.skipUnless(vr.HEIC_AVAILABLE, "pillow-heif not installed")
    def test_heic_round_trip(self):
        import pillow_heif
        source = texture(400, 5)
        buffer = io.BytesIO()
        pillow_heif.from_pillow(Image.fromarray(source)).save(buffer, format="HEIF", quality=90)
        image = vr.decode_image(buffer.getvalue(), "photo.HEIC")
        self.assertEqual(image.rgb.shape, source.shape)
        self.assertLess(float(np.abs(image.rgb.astype(int) - source).mean()), 12.0)

    def test_check_uploads(self):
        ok = [(1000, f"{n}.jpg") for n in ("a", "b", "c")]
        vr.check_uploads(ok)
        for bad in (ok[:2], ok * 3, ok[:2] + [(1000, "c.gif")], ok[:2] + [(0, "c.jpg")],
                    ok[:2] + [(vr.MAX_FILE_BYTES + 1, "c.jpg")]):
            with self.assertRaises(vr.ImageError):
                vr.check_uploads(bad)
        vr.check_uploads(ok[:2] + [(1000, "c.PNG")])


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.base = texture(500, 11)
        matrix = cv2.getRotationMatrix2D((250, 250), 8, 1.0)
        matrix[:, 2] += (14, -9)
        self.matrix = matrix
        self.moved = cv2.warpAffine(self.base, matrix, (500, 500), borderMode=cv2.BORDER_REFLECT)
        self.fa, self.fb = vr.detect_features(self.base), vr.detect_features(self.moved)

    def test_features_detected(self):
        self.assertGreater(len(self.fa.points), 100)
        self.assertEqual(self.fa.descriptors.shape[1], 128)

    def test_matches_follow_the_known_transform(self):
        pairs = vr.match_descriptors(self.fa.descriptors, self.fb.descriptors)
        self.assertGreater(len(pairs), 50)
        predicted = self.fa.points[pairs[:, 0]] @ self.matrix[:, :2].T + self.matrix[:, 2]
        error = np.linalg.norm(predicted - self.fb.points[pairs[:, 1]], axis=1)
        self.assertGreater(np.mean(error < 3), 0.8)
        self.assertEqual(len(set(pairs[:, 1])), len(pairs))  # one-to-one

    def test_ransac_rejects_injected_outliers(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(50, 450, (120, 2))
        b = a @ self.matrix[:, :2].T + self.matrix[:, 2]
        outlier = np.zeros(120, bool)
        outlier[:30] = True
        b[outlier] = rng.uniform(50, 450, (30, 2))
        mask = vr.ransac_inliers(a.astype(np.float32), b.astype(np.float32))
        self.assertGreater(mask[~outlier].mean(), 0.95)
        self.assertLess(mask[outlier].mean(), 0.25)
        np.testing.assert_array_equal(mask, vr.ransac_inliers(a.astype(np.float32), b.astype(np.float32)))

    def test_ransac_too_few_points(self):
        self.assertIsNone(vr.ransac_inliers(np.zeros((5, 2), np.float32), np.zeros((5, 2), np.float32)))

    def test_match_pair_statistics(self):
        inliers, stats = vr.match_pair(self.fa, self.fb, 0, 1)
        self.assertGreaterEqual(stats.inliers, vr.MIN_PAIR_INLIERS)
        self.assertLessEqual(stats.inliers, stats.ratio_matches)
        self.assertEqual(len(inliers), stats.inliers)
        self.assertGreater(stats.median_displacement_px, 5)

    def test_unrelated_images_are_insufficient(self):
        other = vr.detect_features(texture(500, 99))
        with self.assertRaises(vr.InsufficientMatchesError):
            vr.match_pair(self.fa, other, 0, 1)

    def test_identical_images_are_degenerate(self):
        with self.assertRaises(vr.DegenerateError):
            vr.match_pair(self.fa, vr.detect_features(self.base), 0, 1)

    def test_featureless_image(self):
        with self.assertRaises(vr.InsufficientMatchesError):
            vr.detect_features(np.full((400, 400, 3), 128, np.uint8))


class TrackConsistencyTests(unittest.TestCase):
    def test_chain_keeps_only_complete_consistent_tracks(self):
        pairs = [np.array([[0, 10], [1, 11], [2, 12], [3, 13]]),
                 np.array([[10, 20], [11, 21], [13, 23], [99, 98]]),
                 np.array([[20, 30], [23, 33], [21, 31]])]
        tracks = vr.chain_tracks(pairs)
        self.assertEqual(sorted(map(tuple, tracks)), [(0, 10, 20, 30), (1, 11, 21, 31), (3, 13, 23, 33)])

    def test_chain_empty(self):
        self.assertEqual(vr.chain_tracks([np.array([[0, 1]]), np.array([[5, 6]])]).shape, (0, 3))

    def test_trim_removes_inconsistent_track(self):
        tracks = orthographic_tracks(random_points(60), noise=0.2)
        tracks[2, 7] += (80, -60)  # one wrongly matched point in one view
        keep = vr.trim_inconsistent(tracks)
        self.assertNotIn(7, keep)
        self.assertGreaterEqual(len(keep), 55)


class FactorizationTests(unittest.TestCase):
    def test_measurement_matrix_layout_and_centering(self):
        tracks = orthographic_tracks(random_points(10), views=4)
        raw, centered, centroids = vr.measurement_matrix(tracks)
        self.assertEqual(raw.shape, (8, 10))
        np.testing.assert_array_equal(raw[0], tracks[0, :, 0])  # V1_x
        np.testing.assert_array_equal(raw[3], tracks[1, :, 1])  # V2_y
        np.testing.assert_allclose(centered.mean(axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(centroids[2], tracks[2].mean(axis=0))
        with self.assertRaises(ValueError):
            vr.measurement_matrix(np.zeros((4, 10)))

    def test_exact_orthographic_data_is_rank_three(self):
        _, centered, _ = vr.measurement_matrix(orthographic_tracks(random_points()))
        sv = np.linalg.svd(centered, compute_uv=False)
        self.assertLess(sv[3] / sv[0], 1e-12)

    def test_recovers_known_structure_up_to_similarity(self):
        truth = random_points()
        result = vr.reconstruct_from_tracks(orthographic_tracks(truth), [(1000, 800)] * 4, trim_outliers=False)
        self.assertEqual(result.status, "ok")
        self.assertLess(procrustes_error(result.points_metric, truth), 1e-6)
        self.assertLess(procrustes_error(result.points, truth), 1e-6)
        self.assertLess(result.diagnostics["reprojection_rms_px"], 1e-8)
        self.assertEqual(result.quality, "good")

    def test_noisy_reconstruction_is_close_and_deterministic(self):
        truth = random_points(80)
        tracks = orthographic_tracks(truth, noise=0.3)
        a = vr.reconstruct_from_tracks(tracks, [(1000, 800)] * 4)
        b = vr.reconstruct_from_tracks(tracks, [(1000, 800)] * 4)
        np.testing.assert_array_equal(a.points, b.points)
        self.assertLess(procrustes_error(a.points, truth), 0.05)
        self.assertLess(a.diagnostics["reprojection_rms_px"], 1.0)

    def test_metric_constraint_rows_are_orthonormal(self):
        result = vr.reconstruct_from_tracks(orthographic_tracks(random_points()), [(1000, 800)] * 4)
        for norm_i, norm_j, dot in result.diagnostics["camera_rows"]:
            self.assertAlmostEqual(norm_i, 1, places=6)
            self.assertAlmostEqual(norm_j, 1, places=6)
            self.assertAlmostEqual(dot, 0, places=6)

    def test_principal_axes_frame_is_rigid(self):
        points = random_points() * [3, 2, 0.5]
        aligned, spreads = vr.principal_axes_frame(points)
        np.testing.assert_allclose(aligned.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.sort(np.linalg.norm(aligned[:, None] - aligned[None], axis=2).ravel()),
                                   np.sort(np.linalg.norm(points[:, None] - points[None], axis=2).ravel()), atol=1e-9)
        self.assertTrue(spreads[0] >= spreads[1] >= spreads[2])

    def test_reproduces_original_book_experiment(self):
        out = PART_B / "output"
        with (out / "part_b_correspondences.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        tracks = np.array([[[float(r[f"view{v}_x"]), float(r[f"view{v}_y"])] for r in rows] for v in range(1, 5)])
        result = vr.reconstruct_from_tracks(tracks, [(4284, 5712)] * 4, min_points=8, trim_outliers=False)
        with (out / "part_b_reconstructed_points_unscaled.csv").open(newline="") as handle:
            saved = np.array([[float(r[a]) for a in "XYZ"] for r in csv.DictReader(handle)])
        with (out / "part_b_singular_values.csv").open(newline="") as handle:
            saved_sv = [float(r["singular_value"]) for r in csv.DictReader(handle)]
        np.testing.assert_allclose(result.points_metric, saved, rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(result.diagnostics["singular_values"], saved_sv, rtol=1e-9, atol=1e-6)
        self.assertAlmostEqual(result.diagnostics["first_three_energy"], 0.99918174594472, places=9)
        self.assertIn("near_planar", result.warnings)  # the cover is flat; the depth is model error


class DegenerateCaseTests(unittest.TestCase):
    def test_too_few_views_or_points(self):
        with self.assertRaises(vr.InsufficientMatchesError):
            vr.reconstruct_from_tracks(orthographic_tracks(random_points(), views=2), [(1000, 800)] * 2)
        with self.assertRaises(vr.InsufficientMatchesError):
            vr.reconstruct_from_tracks(orthographic_tracks(random_points(8)), [(1000, 800)] * 4)

    def test_nonfinite_tracks_rejected(self):
        tracks = orthographic_tracks(random_points())
        tracks[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            vr.reconstruct_from_tracks(tracks, [(1000, 800)] * 4)

    def test_exactly_planar_scene_is_degenerate(self):
        points = random_points()
        points[:, 2] = 0
        # Planar scene + orthographic cameras gives rank 2: honest failure, not a fake 3D cloud.
        with self.assertRaises(vr.DegenerateError):
            vr.reconstruct_from_tracks(orthographic_tracks(points), [(1000, 800)] * 4, trim_outliers=False)

    def test_noisy_planar_scene_warns(self):
        points = random_points(80)
        points[:, 2] = 0
        result = vr.reconstruct_from_tracks(orthographic_tracks(points, noise=0.5), [(1000, 800)] * 4)
        if result.status == "ok":
            self.assertIn("near_planar", result.warnings)
            self.assertNotEqual(result.quality, "good")

    def test_collinear_points_are_degenerate(self):
        line = np.outer(np.linspace(-1, 1, 30), [1, 0.5, 0.2])
        with self.assertRaises(vr.DegenerateError):
            vr.reconstruct_from_tracks(orthographic_tracks(line), [(1000, 800)] * 4, trim_outliers=False)

    def test_non_positive_definite_metric_is_reported_not_corrected(self):
        motion = np.random.default_rng(2).normal(size=(8, 3))  # seed chosen: L has a negative eigenvalue
        metric = vr.solve_metric(motion)
        self.assertFalse(metric["positive_definite"])
        self.assertLess(metric["eigenvalues"][0], 0)

    def test_unstructured_tracks_never_look_good(self):
        tracks = np.random.default_rng(1).uniform(0, 500, (4, 40, 2))
        result = vr.reconstruct_from_tracks(tracks, [(500, 500)] * 4, trim_outliers=False)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.quality, "poor")

    def test_failed_metric_upgrade_result(self):
        original = vr.solve_metric
        bad = dict(original(np.random.default_rng(2).normal(size=(8, 3))))
        vr.solve_metric = lambda motion: bad
        try:
            result = vr.reconstruct_from_tracks(orthographic_tracks(random_points()), [(1000, 800)] * 4)
        finally:
            vr.solve_metric = original
        self.assertEqual(result.status, "metric_upgrade_failed")
        self.assertIsNone(result.points)
        self.assertEqual(result.quality, "poor")
        self.assertIn("No correction was applied", result.message)
        with self.assertRaises(ValueError):
            vr.plotly_payload(result)
        with self.assertRaises(ValueError):
            vr.apply_reference_distance(result, 0, 1, 5.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = cube_files()
        cls.result = vr.reconstruct_from_images(cls.files)

    def test_output_shapes_and_finiteness(self):
        r = self.result
        self.assertEqual(r.status, "ok")
        count = len(r.points)
        self.assertGreater(count, 100)
        self.assertEqual(r.points.shape, (count, 3))
        self.assertEqual(r.points_metric.shape, (count, 3))
        self.assertEqual(r.tracks.shape, (4, count, 2))
        self.assertEqual(r.colors.shape, (count, 3))
        self.assertEqual(r.point_errors.shape, (count,))
        for array in (r.points, r.points_metric, r.tracks, r.point_errors):
            self.assertTrue(np.isfinite(array).all())

    def test_quality_and_statistics(self):
        r = self.result
        self.assertEqual(r.quality, "good")
        self.assertLess(r.diagnostics["reprojection_rms_px"], 2.0)
        self.assertGreater(r.diagnostics["planarity_ratio"], vr.PLANAR_RATIO)
        self.assertEqual(len(r.pair_stats), 3)
        for stats in r.pair_stats:
            self.assertGreaterEqual(stats.inliers, vr.MIN_PAIR_INLIERS)
            self.assertLessEqual(stats.inliers, stats.ratio_matches)
        self.assertLessEqual(r.diagnostics["n_points_used"], r.diagnostics["n_points_initial"])

    def test_reconstruction_is_three_dimensional_cube(self):
        spreads = self.result.diagnostics["principal_spreads"]
        self.assertGreater(spreads[2] / spreads[0], 0.4)  # a cube is not flat
        self.assertIn("affine", self.result.model.lower())
        self.assertEqual(self.result.units, vr.ARBITRARY_UNITS)

    def test_deterministic(self):
        again = vr.reconstruct_from_images(self.files)
        np.testing.assert_array_equal(again.points, self.result.points)

    def test_plotly_payload(self):
        payload = vr.plotly_payload(self.result)
        count = len(self.result.points)
        for key in ("x", "y", "z", "colors", "hover"):
            self.assertEqual(len(payload[key]), count)
        self.assertEqual(payload["valid_point_count"], count)
        self.assertEqual(payload["axis_titles"]["x"], f"X ({vr.ARBITRARY_UNITS})")
        self.assertEqual(payload["aspectmode"], "data")
        self.assertEqual(len(payload["match_statistics"]), 3)
        self.assertIn("reprojection_rms_px", payload["diagnostics"])
        self.assertTrue(all(c.startswith("rgb(") for c in payload["colors"]))

    def test_reference_distance_scaling(self):
        scaled = vr.apply_reference_distance(self.result, 0, 1, 10.0, "cm")
        self.assertAlmostEqual(float(np.linalg.norm(scaled.points[0] - scaled.points[1])), 10.0)
        self.assertEqual(scaled.units, "cm")
        self.assertEqual(vr.plotly_payload(scaled)["axis_titles"]["z"], "Z (cm)")
        self.assertAlmostEqual(procrustes_error(scaled.points, self.result.points), 0, places=9)  # shape unchanged
        self.assertEqual(self.result.units, vr.ARBITRARY_UNITS)  # original untouched
        for args in ((0, 0, 5.0), (0, 10 ** 6, 5.0), (0, 1, 0.0), (0, 1, float("nan")), (0, 1, -2.0)):
            with self.assertRaises(ValueError):
                vr.apply_reference_distance(self.result, *args)

    def test_unrelated_photo_gives_actionable_error(self):
        files = self.files[:3] + [(png_bytes(texture(700, 77)), "other.png")]
        with self.assertRaises(vr.InsufficientMatchesError) as caught:
            vr.reconstruct_from_images(files)
        self.assertIn("Photos 3 and 4", str(caught.exception))

    def test_bad_upload_sets(self):
        with self.assertRaises(vr.ImageError):
            vr.reconstruct_from_images(self.files[:2])
        with self.assertRaises(vr.ImageError):
            vr.reconstruct_from_images(self.files[:3] + [(b"garbage", "bad.jpg")])

    def test_busy_server(self):
        held = [vr._JOB_SLOTS.acquire(blocking=False) for _ in range(vr.MAX_CONCURRENT_JOBS)]
        try:
            with self.assertRaises(vr.BusyError):
                vr.reconstruct_from_images(self.files)
        finally:
            for ok in held:
                if ok:
                    vr._JOB_SLOTS.release()
        self.assertTrue(vr._JOB_SLOTS.acquire(blocking=False))  # slots were released after the failure too
        vr._JOB_SLOTS.release()

    def test_jobs_release_slots_after_errors(self):
        for _ in range(vr.MAX_CONCURRENT_JOBS + 1):
            with self.assertRaises(vr.ReconstructionError):
                vr.reconstruct_from_images(self.files[:3] + [(png_bytes(texture(700, 5)), "x.png")])

    def test_progress_callback(self):
        stages = []
        vr.reconstruct_from_images(self.files, progress=lambda stage, fraction: stages.append(stage))
        self.assertEqual(stages[0], "Reading photos")
        self.assertIn("Matching photos", stages)
        self.assertEqual(stages[-1], "Factorizing")


class StagedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = cube_files()
        cls.match = vr.find_matches(cls.files)

    def test_image_summary(self):
        info = vr.image_summary(*self.files[0])
        self.assertEqual((info["width"], info["height"]), (900, 700))
        self.assertEqual(info["size_bytes"], len(self.files[0][0]))
        thumb = vr.decode_image(info["thumbnail"], "t.jpg")
        self.assertLessEqual(max(thumb.size), vr.THUMB_LONG_SIDE)
        with self.assertRaises(vr.ImageError):
            vr.image_summary(b"junk", "x.jpg")
        with self.assertRaises(vr.ImageError):
            vr.image_summary(b"junk", "x.gif")

    def test_find_matches_contents(self):
        m = self.match
        self.assertEqual(m.tracks.shape, (4, m.n_tracks, 2))
        self.assertLessEqual(m.n_tracks, m.n_tracks_found)
        self.assertEqual(len(m.overlays), 3)
        self.assertEqual(len(m.pair_stats), 3)
        self.assertEqual(m.filenames, [name for _, name in self.files])
        for blob in m.overlays + [m.marker_image]:
            self.assertEqual(vr.decode_image(blob, "o.jpg").rgb.ndim, 3)
        self.assertLessEqual(vr.decode_image(m.overlays[0], "o.jpg").size[0], vr.OVERLAY_MAX_WIDTH)
        self.assertLessEqual(len(m.marker_candidates), vr.MAX_MARKERS)
        self.assertEqual(m.marker_candidates, sorted(set(m.marker_candidates)))
        self.assertTrue(all(0 <= i < m.n_tracks for i in m.marker_candidates))

    def test_matches_are_not_invented(self):
        # Every kept track must be a real SIFT keypoint position from the photos (sub-pixel coordinates).
        image = vr.decode_image(self.files[0][0], "a.png")
        keypoints = vr.detect_features(image.rgb).points
        distance = np.min(np.linalg.norm(self.match.tracks[0][:, None] - keypoints[None], axis=2), axis=1)
        self.assertLess(distance.max(), 1e-3)

    def test_staged_equals_one_shot(self):
        staged = vr.reconstruct_matches(self.match)
        direct = vr.reconstruct_from_images(self.files)
        np.testing.assert_allclose(staged.points, direct.points, atol=1e-9)
        self.assertEqual(staged.diagnostics["n_points_initial"], self.match.n_tracks_found)
        self.assertEqual(staged.diagnostics["n_points_used"], self.match.n_tracks)

    def test_scale_applied_in_staged_workflow(self):
        scaled = vr.reconstruct_matches(self.match, dict(a=0, b=5, distance=12.5, unit="mm"))
        self.assertEqual(scaled.units, "mm")
        self.assertAlmostEqual(float(np.linalg.norm(scaled.points[0] - scaled.points[5])), 12.5)
        with self.assertRaises(vr.ReconstructionError):
            vr.reconstruct_matches(self.match, dict(a=2, b=2, distance=1.0, unit="cm"))

    def test_find_matches_failures_and_slots(self):
        with self.assertRaises(vr.InsufficientMatchesError):
            vr.find_matches(self.files[:3] + [(png_bytes(texture(700, 77)), "other.png")])
        held = [vr._JOB_SLOTS.acquire(blocking=False) for _ in range(vr.MAX_CONCURRENT_JOBS)]
        try:
            with self.assertRaises(vr.BusyError):
                vr.find_matches(self.files)
        finally:
            for ok in held:
                if ok:
                    vr._JOB_SLOTS.release()

    def test_exports(self):
        result = vr.reconstruct_matches(self.match)
        rows = list(csv.reader(io.StringIO(vr.points_csv(result))))
        self.assertEqual(len(rows), len(result.points) + 1)
        self.assertEqual(rows[0][1], f"X ({vr.ARBITRARY_UNITS})")
        self.assertEqual(float(rows[1][1]), float(f"{result.points[0, 0]:.6f}"))
        lines = vr.points_ply(result).splitlines()
        self.assertEqual(lines[0], "ply")
        self.assertIn(f"element vertex {len(result.points)}", lines)
        self.assertIn(f"comment coordinate units: {vr.ARBITRARY_UNITS}", lines)
        body = lines[lines.index("end_header") + 1:]
        self.assertEqual(len(body), len(result.points))
        self.assertEqual(len(body[0].split()), 6)
        report = vr.diagnostic_report(result)
        for text in ("arbitrary reconstruction units", "Singular values", "RANSAC inliers", "not a calibrated"):
            self.assertIn(text, report)
        scaled = vr.reconstruct_matches(self.match, dict(a=0, b=5, distance=3.0, unit="cm"))
        self.assertIn("Z (cm)", vr.points_csv(scaled))
        self.assertIn("coordinate units: cm", vr.points_ply(scaled))
        self.assertIn("points 1 and 6 set to 3 cm", vr.diagnostic_report(scaled))

    def test_failed_result_has_report_but_no_points(self):
        bad = dict(vr.solve_metric(np.random.default_rng(2).normal(size=(8, 3))))
        original = vr.solve_metric
        vr.solve_metric = lambda motion: bad
        try:
            result = vr.reconstruct_matches(self.match)
        finally:
            vr.solve_metric = original
        self.assertEqual(result.status, "metric_upgrade_failed")
        self.assertIn("metric upgrade failed", vr.diagnostic_report(result).lower())
        with self.assertRaises(ValueError):
            vr.points_csv(result)
        with self.assertRaises(ValueError):
            vr.points_ply(result)


class ProtectionTests(unittest.TestCase):
    def test_backend_does_not_touch_experiment_files(self):
        def digest():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PART_B.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts and p.name != vr.__file__.split("/")[-1]}
        before = digest()
        vr.reconstruct_from_images(cube_files(3, 0.2))
        self.assertEqual(digest(), before)


class MarkerRenderingTests(unittest.TestCase):
    POINTS = np.array([[40.0, 40.0], [160.0, 40.0], [100.0, 120.0], [40.0, 150.0]])
    INDICES = [0, 1, 2, 3]

    def render(self, background, selected=None):
        rgb = np.full((200, 200, 3), background, np.uint8)
        data = vr.render_point_markers(rgb, self.POINTS, self.INDICES, selected=selected)
        return np.array(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)

    @staticmethod
    def near(image, color, center, radius=14):
        x, y = center
        patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1].reshape(-1, 3)
        return int((np.abs(patch - np.array(color)).max(axis=1) < 60).sum())

    def test_ordinary_points_are_cyan_with_dark_outline_on_light_and_dark(self):
        for background in (255, 0):
            image = self.render(background)
            for x, y in self.POINTS.astype(int):
                self.assertGreater(self.near(image, vr.MARKER_CYAN, (x, y)), 10)
                if background == 255:  # on black the outline is indistinguishable from the photo
                    self.assertGreater(self.near(image, vr.MARKER_OUTLINE, (x, y)), 4)
                self.assertEqual(self.near(image, vr.MARKER_A_COLOR, (x, y)), 0)

    def test_selected_points_have_distinct_larger_highlights(self):
        plain = self.render(128)
        image = self.render(128, selected=(0, 2))
        a, b, other = (40, 40), (100, 120), (160, 40)
        self.assertGreater(self.near(image, vr.MARKER_A_COLOR, a), 20)
        self.assertGreater(self.near(image, vr.MARKER_B_COLOR, b), 20)
        self.assertEqual(self.near(image, vr.MARKER_B_COLOR, a), 0)
        self.assertEqual(self.near(image, vr.MARKER_A_COLOR, b), 0)
        self.assertGreater(self.near(image, vr.MARKER_CYAN, other), 10)  # others stay cyan
        self.assertEqual(self.near(image, vr.MARKER_A_COLOR, other), 0)
        self.assertGreater(self.near(image, vr.MARKER_A_COLOR, a), self.near(plain, vr.MARKER_CYAN, a))

    def test_connecting_line_follows_selected_coordinates(self):
        image = self.render(0, selected=(0, 1))  # horizontal line y=40 between x=40 and x=160
        middle = image[38:43, 100].reshape(-1, 3)
        self.assertTrue((middle.min(axis=1) > 180).any())
        off_line = image[100:105, 100].reshape(-1, 3)
        self.assertTrue((off_line.max(axis=1) < 60).all())
        diagonal = self.render(0, selected=(0, 2))
        self.assertFalse((diagonal[38:43, 100].reshape(-1, 3).min(axis=1) > 180).any())

    def test_changing_selection_changes_image_and_inputs_are_untouched(self):
        rgb = np.full((200, 200, 3), 90, np.uint8)
        points = self.POINTS.copy()
        first = vr.render_point_markers(rgb, points, self.INDICES, selected=(0, 1))
        second = vr.render_point_markers(rgb, points, self.INDICES, selected=(1, 2))
        self.assertNotEqual(first, second)
        self.assertEqual(first, vr.render_point_markers(rgb, points, self.INDICES, selected=(0, 1)))
        np.testing.assert_array_equal(points, self.POINTS)

    def test_base_rerender_matches_direct_render_and_numbering(self):
        rgb = np.full((300, 3000, 3), 90, np.uint8)  # wide enough to be downscaled
        points = np.array([[300.0, 100.0], [2400.0, 200.0]])
        base = vr.marker_base(rgb)
        self.assertLess(base[1], 1.0)
        direct = vr.render_point_markers(rgb, points, [0, 1], selected=(0, 1))
        again = vr.render_point_markers(None, points, [0, 1], selected=(0, 1), base=base)
        self.assertEqual(direct, again)


if __name__ == "__main__":
    unittest.main()
