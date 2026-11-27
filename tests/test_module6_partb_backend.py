"""Module 6 Part B shared backend tests: images, correspondence estimation, eight-point reconstruction.

Synthetic data plus the saved book experiment; no Streamlit, no browser.
"""
import csv
import hashlib
import io
import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from Modules.Module_6.Part_B import sfm_pipeline as sfm
from Modules.Module_6.Part_B import visitor_reconstruction as vr

ROOT = Path(__file__).resolve().parents[1]
PART_B = ROOT / "Modules" / "Module_6" / "Part_B"
SAVED = PART_B / "output" / "revised"
PHOTOS = PART_B / "submission_images"
BOOK_NAMES = ("IMG_7596.jpeg", "IMG_7597.jpeg", "IMG_7598.jpeg", "IMG_7603.jpeg")


# ----------------------------------------------------------------- synthetic data

def texture(size, seed, width=None):
    rng = np.random.default_rng(seed)
    width = width or size
    img = np.zeros((size, width, 3), np.float32)
    for scale in (4, 8, 16, 32):
        noise = rng.random((size // scale + 1, width // scale + 1, 3)).astype(np.float32)
        img += cv2.resize(noise, (width, size), interpolation=cv2.INTER_CUBIC) * (scale ** .5)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    for _ in range(80):
        center = (int(rng.integers(0, width)), int(rng.integers(0, size)))
        cv2.circle(img, center, int(rng.integers(3, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def png_bytes(rgb):
    ok, data = cv2.imencode(".png", rgb[:, :, ::-1])
    assert ok
    return data.tobytes()


def jpeg_with_orientation(size, orientation):
    image = Image.fromarray(texture(max(size), 1)[:size[1], :size[0]])
    exif = Image.Exif()
    exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


OBJECT_W, OBJECT_H = 420, 600
# P1-P4 corners and four interior landmarks, in object (texture) pixel coordinates
OBJECT_POINTS = np.array([[0, 0], [OBJECT_W - 1, 0], [OBJECT_W - 1, OBJECT_H - 1], [0, OBJECT_H - 1],
                          [120, 150], [300, 110], [330, 470], [90, 500]], float)
# Where the four object corners land in each 1000 x 900 photo (view 4 pushes P2 outside the photo).
VIEW_CORNERS = (((300, 150), (700, 150), (700, 720), (300, 720)),
                ((260, 210), (690, 120), (740, 700), (240, 760)),
                ((350, 130), (760, 220), (680, 780), (290, 690)),
                ((60, 780), (-25, 200), (600, 140), (680, 760)))  # rotated 90 degrees
OBJECT_TEXTURE = None


def planar_views():
    """Four photos of one textured flat rectangle; returns (photo bytes, true points per view (4, 8, 2))."""
    global OBJECT_TEXTURE
    if OBJECT_TEXTURE is None:
        OBJECT_TEXTURE = texture(OBJECT_H, 7, OBJECT_W)
    source = np.float32(OBJECT_POINTS[:4])
    files, truth = [], []
    for v, corners in enumerate(VIEW_CORNERS):
        H = cv2.getPerspectiveTransform(source, np.float32(corners))
        canvas = np.full((900, 1000, 3), 205, np.uint8)
        warped = cv2.warpPerspective(OBJECT_TEXTURE, H, (1000, 900), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((OBJECT_H, OBJECT_W), 255, np.uint8), H, (1000, 900)) > 127
        canvas[mask] = warped[mask]
        files.append((png_bytes(canvas), f"view{v + 1}.png"))
        truth.append(cv2.perspectiveTransform(OBJECT_POINTS.reshape(-1, 1, 2), H).reshape(-1, 2))
    return files, np.array(truth)


def rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])


# A 10 x 20 rectangle (P1-P4) with four raised landmarks, so the orthographic data is exactly rank 3.
SHAPE = np.array([[-5, -10, 0], [5, -10, 0], [5, 10, 0], [-5, 10, 0],
                  [-2, -4, 1.5], [3, -6, -1.0], [2, 5, 2.0], [-3, 6, -1.5]], float)


def orthographic_tracks(points=SHAPE, scale=40.0):
    tracks = []
    for v in range(4):
        R = rotation(0.35 * v - 0.5, 0.25 * v - 0.3)
        tracks.append((points @ R.T)[:, :2] * scale + [600 + 25 * v, 500 - 15 * v])
    return np.array(tracks)


def saved_rows(name):
    with (SAVED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def book_tracks():
    """The saved final (4, 8, 2) coordinates of the book experiment."""
    rows = saved_rows("part_b_correspondences.csv")
    return np.array([[[float(r[f"view{v}_x"]), float(r[f"view{v}_y"])] for r in rows] for v in range(1, 5)])


def book_selection():
    return json.loads((PART_B / "correspondence_selection.json").read_text(encoding="utf-8"))


def book_images():
    return [vr.decode_image((PHOTOS / name).read_bytes(), name) for name in BOOK_NAMES]


def estimate(images, points):
    """Homography estimates in photos 2-4 of points marked in photo 1 (the matching step of find_correspondences)."""
    return sfm._match_views(images, np.asarray(points, float))[1]


# ----------------------------------------------------------------- images

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

    def test_file_type_and_size_limits(self):
        vr.check_file(1000, "a.jpg")
        for size, name in ((1000, "d.gif"), (0, "d.jpg"), (vr.MAX_FILE_BYTES + 1, "d.jpg")):
            with self.assertRaises(vr.ImageError):
                vr.check_file(size, name)

    def test_featureless_image(self):
        with self.assertRaises(vr.InsufficientMatchesError):
            vr.detect_features(np.full((400, 400, 3), 128, np.uint8))

    def test_feature_mask_restricts_detection(self):
        rgb = texture(400, 3)
        mask = np.zeros((400, 400), np.uint8)
        mask[100:300, 100:300] = 255
        features = vr.detect_features(rgb, mask=mask)
        self.assertTrue(((features.points >= 95) & (features.points <= 305)).all())


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        # 4284 x 5712 upright original decoded to a 1200 x 1600 working image, like the book photos.
        self.image = vr.DecodedImage(np.zeros((1600, 1200, 3), np.uint8), "x.jpg", (4284, 5712), 1200 / 4284, 6)

    def test_working_and_original_round_trip(self):
        points = np.array([[0, 0], [4283, 5711], [883, 1198], [2141.5, 2855.5]], float)
        np.testing.assert_allclose(sfm.to_original(sfm.to_working(points, self.image), self.image), points, atol=1e-9)
        # pixel-center convention: the image center maps to the center
        np.testing.assert_allclose(sfm.to_working([[2141.5, 2855.5]], self.image), [[599.5, 799.5]], atol=1e-9)

    def test_display_clicks_map_to_original_pixels(self):
        # a 400 px wide display of the whole photo: each displayed pixel covers 4284 / 400 original pixels
        center = sfm.display_to_original(199.5, 266.5, 400, 533, self.image)
        np.testing.assert_allclose(center, [2141.5, 2855.5], atol=6)
        corner = sfm.display_to_original(-5, 10_000, 400, 533, self.image)
        np.testing.assert_allclose(corner, [0, 5711])  # clipped inside the photo
        with self.assertRaises(ValueError):
            sfm.display_to_original(1, 1, 0, 533, self.image)

    def test_zoom_clicks_map_through_the_crop(self):
        origin, crop = (500, 700), (100, 100)  # working pixels, shown 4x larger
        point = sfm.display_to_original(201.5, 41.5, 400, 400, self.image, origin, crop)
        np.testing.assert_allclose(sfm.to_working(point, self.image), [550.0, 710.0], atol=1e-9)

    def test_zoom_window_stays_inside(self):
        self.assertEqual(vr.zoom_window((5, 5), 50, (1200, 1600)), (0, 0, 100, 100))
        self.assertEqual(vr.zoom_window((1199, 1599), 50, (1200, 1600)), (1100, 1500, 100, 100))
        self.assertEqual(vr.zoom_window((600, 800), 50, (1200, 1600)), (550, 750, 100, 100))


# ----------------------------------------------------------------- correspondence estimation

class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files, cls.truth = planar_views()
        cls.images = [vr.decode_image(data, name) for data, name in files]
        cls.predictions = estimate(cls.images, cls.truth[0])

    def test_predictions_follow_the_known_homographies(self):
        self.assertEqual([p.view for p in self.predictions], [2, 3, 4])
        for prediction in self.predictions:
            self.assertTrue(prediction.reliable, prediction.message)
            self.assertGreaterEqual(prediction.inliers, sfm.MIN_HOMOGRAPHY_INLIERS)
            self.assertGreaterEqual(prediction.coverage, sfm.MIN_INLIER_COVERAGE)
            found = np.isfinite(prediction.points).all(axis=1)
            error = np.linalg.norm(prediction.points[found] - self.truth[prediction.view - 1][found], axis=1)
            self.assertLess(error.max(), 2.0, prediction.view)

    def test_point_identities_are_kept(self):
        # each predicted point is nearest to its own true position, never to another point's
        for prediction in self.predictions:
            truth = self.truth[prediction.view - 1]
            for i, point in enumerate(prediction.points):
                if np.isfinite(point).all():
                    self.assertEqual(int(np.argmin(np.linalg.norm(truth - point, axis=1))), i)

    def test_point_outside_the_photo_is_reported_missing(self):
        view4 = self.predictions[2]
        self.assertEqual(view4.status[1], sfm.OUTSIDE)
        self.assertTrue(np.isnan(view4.points[1]).all())
        self.assertEqual(view4.status.count(sfm.PREDICTED), 7)
        self.assertIn("outside the photo", view4.message)

    def test_deterministic(self):
        again = estimate(self.images, self.truth[0])
        for a, b in zip(self.predictions, again):
            np.testing.assert_array_equal(a.points, b.points)

    def test_unrelated_photo_gives_no_prediction(self):
        unrelated = vr.decode_image(png_bytes(texture(900, 42, 1000)), "other.png")
        predictions = estimate([*self.images[:3], unrelated], self.truth[0])
        self.assertTrue(predictions[0].reliable)
        self.assertFalse(predictions[2].reliable)
        self.assertEqual(predictions[2].status, (sfm.UNAVAILABLE,) * 8)
        self.assertTrue(np.isnan(predictions[2].points).all())
        self.assertIn("Select its points manually", predictions[2].message)

    def test_featureless_object_gives_no_prediction(self):
        blank = vr.DecodedImage(np.full((900, 1000, 3), 128, np.uint8), "blank.png", (1000, 900), 1.0, 1)
        predictions = estimate([blank, *self.images[1:]], self.truth[0])
        self.assertTrue(all(not p.reliable for p in predictions))

    def test_invalid_reference_points(self):
        size = self.images[0].original_size
        truth = self.truth[0]
        for bad in (truth[:7], np.where(np.arange(8)[:, None] == 2, np.nan, truth),
                    truth[[0, 2, 1, 3, 4, 5, 6, 7]],  # corners out of order: crossing outline
                    np.vstack([truth[:7], truth[:1]]),  # duplicate position
                    truth + [2000, 0]):  # outside the photo
            with self.assertRaises(sfm.PointError):
                sfm.check_reference_points(bad, size)
        tiny = truth.copy()
        tiny[:4] = [[10, 10], [20, 10], [20, 20], [10, 20]]
        with self.assertRaises(sfm.PointError):
            sfm.check_reference_points(tiny, size)

    def test_working_resolution_predictions_are_in_original_pixels(self):
        # The same photos upscaled 2x decode to the same working image size; predictions scale back up.
        files, truth = planar_views()
        big = []
        for data, name in files:
            rgb = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            big.append(vr.decode_image(png_bytes(cv2.resize(rgb, None, fx=2, fy=2)), name, long_side=1000))
        self.assertAlmostEqual(big[0].scale, 0.5)
        predictions = estimate(big, truth[0] * 2 + 0.5)
        found = np.isfinite(predictions[0].points).all(axis=1)
        error = np.linalg.norm(predictions[0].points[found] - (truth[1][found] * 2 + 0.5), axis=1)
        self.assertLess(error.max(), 4.0)


class AutomaticLandmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files, cls.truth = planar_views()
        cls.images = [vr.decode_image(data, name) for data, name in files]
        cls.found = sfm.find_correspondences(cls.images, cls.truth[0][:4])

    def test_four_distinct_landmarks_inside_the_boundary_in_all_photos(self):
        self.assertEqual(self.found.landmarks_found, 4)
        points = self.found.points
        np.testing.assert_array_equal(points[0, :4], self.truth[0][:4])
        self.assertTrue(np.isfinite(points[:, 4:]).all())
        to_object = cv2.getPerspectiveTransform(np.float32(self.truth[0][:4]),
                                                np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
        inside = cv2.perspectiveTransform(points[0, 4:].reshape(-1, 1, 2), to_object).reshape(-1, 2)
        self.assertTrue(((inside >= 0.045) & (inside <= 0.955)).all())
        gaps = np.linalg.norm(inside[:, None] - inside[None], axis=2)[np.triu_indices(4, 1)]
        self.assertGreaterEqual(gaps.min(), sfm.LANDMARK_SEPARATION - 0.01)
        self.assertEqual([tuple((p >= 0.5).astype(int)) for p in inside], list(sfm.LANDMARK_QUADRANTS))

    def test_landmarks_are_the_same_physical_points(self):
        for v in range(1, 4):
            H = cv2.findHomography(self.truth[0], self.truth[v])[0]
            expected = cv2.perspectiveTransform(self.found.points[0, 4:].reshape(-1, 1, 2), H).reshape(-1, 2)
            self.assertLess(np.linalg.norm(self.found.points[v, 4:] - expected, axis=1).max(), 3.0)

    def test_corners_match_the_homography_estimates(self):
        eight = np.vstack([self.truth[0][:4], self.truth[0][4:]])
        for old, new in zip(estimate(self.images, eight), self.found.views):
            np.testing.assert_array_equal(new.points[:4], old.points[:4])
        self.assertTrue(np.isnan(self.found.points[3, 1]).all())

    def test_deterministic(self):
        again = sfm.find_correspondences(self.images, self.truth[0][:4])
        np.testing.assert_array_equal(again.points, self.found.points)

    def test_unreliable_photo_leaves_landmarks_missing(self):
        unrelated = vr.decode_image(png_bytes(texture(900, 42, 1000)), "other.png")
        found = sfm.find_correspondences([*self.images[:3], unrelated], self.truth[0][:4])
        self.assertEqual(found.landmarks_found, 0)
        self.assertTrue(np.isnan(found.points[:, 4:]).all())
        self.assertTrue(np.isnan(found.points[3]).all())
        self.assertTrue(np.isfinite(found.points[1, :4]).all())

    def test_selection_never_invents_or_clusters_landmarks(self):
        clustered = np.array([[0.3, 0.3], [0.31, 0.3], [0.7, 0.3], [0.01, 0.9]])
        chosen = sfm.select_landmarks(clustered, [1, 2, 3, 4])
        self.assertEqual(sorted(i for i in chosen if i is not None), [1, 2])
        self.assertEqual(sfm.select_landmarks(np.empty((0, 2)), []), [None] * 4)

    def test_invalid_corners(self):
        with self.assertRaises(sfm.PointError):
            sfm.find_correspondences(self.images, self.truth[0][[0, 2, 1, 3]])


class SavedExperimentTests(unittest.TestCase):
    def test_saved_outputs_reproduce_from_the_shared_functions(self):
        final = book_tracks()
        corners = np.array([p["views"] for p in book_selection()["points"]], float).transpose(1, 0, 2)
        np.testing.assert_array_equal(final[:, :4], corners)  # verified corners
        found = sfm.find_correspondences(book_images(), corners[0])
        self.assertEqual(found.landmarks_found, 4)
        np.testing.assert_allclose(final[:, 4:], found.points[:, 4:], atol=0.01)
        script = (PART_B / "automatic_landmark_experiment.py").read_text()
        self.assertIn("sfm.find_correspondences(", script)
        self.assertIn("sfm.reconstruct(", script)
        for duplicate in ("np.linalg.svd(", "def reconstruct", "def solve_metric", "def align"):
            self.assertNotIn(duplicate, script)


# ----------------------------------------------------------------- reconstruction

class ReconstructionTests(unittest.TestCase):
    def test_reproduces_the_saved_book_experiment(self):
        dimensions = book_selection()["book_dimensions_cm"]
        self.assertEqual((dimensions["width"], dimensions["height"]), (14.5, 21.7))
        result = sfm.reconstruct(book_tracks(), 14.5, 21.7)
        summary = json.loads((SAVED / "part_b_summary.json").read_text())
        np.testing.assert_allclose(result.singular_values[:7], summary["singular_values"][:7], rtol=1e-9)
        np.testing.assert_allclose(result.metric["eigenvalues"], summary["L_eigenvalues"], rtol=1e-9)
        np.testing.assert_allclose(result.view_rms, summary["view_rms_px"], rtol=1e-8)
        self.assertAlmostEqual(result.scale, summary["scale_cm_per_unit"], places=12)
        cm = [[float(r[f"{a}_cm"]) for a in "XYZ"] for r in saved_rows("part_b_reconstructed_points_cm.csv")]
        np.testing.assert_allclose(result.cm, cm, atol=1e-9)
        for row, m in zip(saved_rows("part_b_measurements.csv"), result.measurements):
            self.assertEqual(row["edge"], m["label"])
            self.assertAlmostEqual(float(row["reconstructed_cm"]), m["reconstructed"], places=9)
            self.assertAlmostEqual(float(row["error_percent"]), m["error"], places=9)
        edges = result.measurements[:4]
        self.assertEqual([round(m["reconstructed"], 3) for m in edges], [13.108, 15.948, 21.346, 22.016])
        self.assertEqual([round(m["error"], 2) for m in edges], [9.60, 9.98, 1.63, 1.46])

    def test_measurement_matrix_factorization_and_residual(self):
        tracks = book_tracks()
        result = sfm.reconstruct(tracks, 14.5, 21.7)
        for v in range(4):
            np.testing.assert_array_equal(result.raw[2 * v:2 * v + 2], tracks[v].T)
        np.testing.assert_allclose(result.centered.mean(axis=1), 0, atol=1e-10)
        U, sv, Vt = np.linalg.svd(result.centered)
        rank3 = U[:, :3] @ np.diag(sv[:3]) @ Vt[:3]
        np.testing.assert_allclose(result.factors["M_hat"] @ result.factors["S_hat"], rank3, atol=1e-8)
        np.testing.assert_allclose(result.motion @ result.structure, rank3, atol=1e-8)  # the upgrade keeps MS
        residual = result.centered - result.motion @ result.structure
        self.assertAlmostEqual(np.linalg.norm(residual), np.linalg.norm(sv[3:]), places=7)
        np.testing.assert_allclose(
            result.view_rms, np.sqrt((residual.reshape(4, 2, 8) ** 2).sum(axis=1).mean(axis=1)), rtol=1e-12)
        np.testing.assert_allclose(result.Q @ result.Q.T, result.metric["L"], rtol=1e-10)

    def test_alignment_is_rigid_and_plane_distances_are_consistent(self):
        result = sfm.reconstruct(book_tracks(), 14.5, 21.7)
        frame, points = result.frame, result.structure.T
        np.testing.assert_allclose(frame["basis"] @ frame["basis"].T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(frame["basis"]), 1.0, places=12)
        np.testing.assert_allclose(frame["book"], (points - frame["origin"]) @ frame["basis"].T, atol=1e-9)
        np.testing.assert_allclose(frame["book"][:4].mean(axis=0), 0, atol=1e-8)
        distances = (points - frame["centroid"]) @ frame["normal"]
        np.testing.assert_allclose(frame["distances"], distances, atol=1e-9)
        smallest = np.linalg.svd(points - frame["centroid"], compute_uv=False)[-1]
        self.assertAlmostEqual(np.linalg.norm(distances) / smallest, 1.0, places=9)  # least-squares plane
        np.testing.assert_allclose(frame["book"][:, 2] - frame["plane_z"], distances, atol=1e-8)
        summary = json.loads((SAVED / "part_b_summary.json").read_text())
        self.assertAlmostEqual(float(np.abs(result.scale * distances).max()),
                               summary["max_plane_displacement_cm"], places=9)

    def test_metric_upgrade_recovers_orthonormal_camera_rows(self):
        motion = sfm.reconstruct(orthographic_tracks(), 10.0, 20.0).motion
        for v in range(4):
            rows = motion[2 * v:2 * v + 2]
            gram = rows @ rows.T
            np.testing.assert_allclose(gram / gram[0, 0], np.eye(2), atol=1e-8)

    def test_identical_inputs_give_identical_results(self):
        a = sfm.reconstruct(book_tracks(), 14.5, 21.7)
        b = sfm.reconstruct(book_tracks().copy(), 14.5, 21.7)
        np.testing.assert_array_equal(a.cm, b.cm)
        self.assertEqual(a.measurements, b.measurements)

    def test_exact_orthographic_shape_is_recovered(self):
        result = sfm.reconstruct(orthographic_tracks(), 10.0, 20.0)
        self.assertTrue(result.metric["positive_definite"])
        for m in result.measurements:
            self.assertLess(m["error"], 1e-6, m["label"])
        # the whole shape is recovered up to a rigid motion (and mirror): all pairwise distances agree
        def distances(points):
            return np.linalg.norm(points[:, None] - points[None], axis=2)
        np.testing.assert_allclose(distances(result.cm), distances(SHAPE), atol=1e-6)

    def test_scaling_uses_both_dimensions_in_least_squares(self):
        self.assertAlmostEqual(sfm.scale_factor(2.0, 4.0, 10.0, 20.0), 5.0)
        w, h = 2.0, 3.0
        s = sfm.scale_factor(w, h, 10.0, 20.0)
        self.assertAlmostEqual(s, (10 * w + 20 * h) / (w * w + h * h))
        with self.assertRaises(ValueError):
            sfm.scale_factor(0.0, 3.0, 10.0, 20.0)

    def test_invalid_inputs(self):
        tracks = orthographic_tracks()
        missing = tracks.copy()
        missing[2, 5] = np.nan
        duplicate = tracks.copy()
        duplicate[1, 6] = duplicate[1, 2]
        for bad in (missing, duplicate, tracks[:3], tracks[:, :7]):
            with self.assertRaises(sfm.PointError):
                sfm.reconstruct(bad, 10, 20)
        for width, height in ((0, 20), (10, -1), (float("nan"), 20), (None, 20)):
            with self.assertRaises(sfm.PointError):
                sfm.reconstruct(tracks, width, height)

    def test_degenerate_views_fail_clearly(self):
        flat = SHAPE.copy()
        flat[:, 2] = 0  # exactly planar + orthographic cameras: rank 2, no third singular value
        with self.assertRaises(vr.DegenerateError):
            sfm.reconstruct(orthographic_tracks(flat), 10, 20)

    def test_non_positive_definite_metric_is_reported_not_corrected(self):
        motion = np.random.default_rng(2).normal(size=(8, 3))  # seed chosen: L has a negative eigenvalue
        metric = sfm.solve_metric(motion)
        self.assertFalse(metric["positive_definite"])
        original = sfm.solve_metric
        sfm.solve_metric = lambda m: metric
        try:
            with self.assertRaises(vr.DegenerateError) as raised:
                sfm.reconstruct(orthographic_tracks(), 10, 20)
        finally:
            sfm.solve_metric = original
        self.assertIn("not positive definite", str(raised.exception))
        self.assertIn("No correction was applied", str(raised.exception))

    def test_points_csv(self):
        result = sfm.reconstruct(book_tracks(), 14.5, 21.7)
        text = vr.points_csv(result, list(BOOK_NAMES))
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(rows[0][:4], ["point", "X_cm", "Y_cm", "Z_cm"])
        self.assertEqual([r[0] for r in rows[1:9]], [f"P{i}" for i in range(1, 9)])
        self.assertEqual(rows[1][4:6], ["883.00", "1198.00"])
        start = rows.index(["edge", "measured_cm", "reconstructed_cm", "percentage_error"])
        self.assertEqual(rows[start + 1:start + 6], [
            ["Top edge", "14.5", "13.1082", "9.60"], ["Bottom edge", "14.5", "15.9475", "9.98"],
            ["Left edge", "21.7", "21.3462", "1.63"], ["Right edge", "21.7", "22.0164", "1.46"], []])
        self.assertNotIn("diagonal", text.lower())
        self.assertEqual(len(result.measurements), 6)  # the reconstruction itself is unchanged


class ProtectionTests(unittest.TestCase):
    def test_original_photographs_are_unchanged(self):
        if not all((PART_B / "input" / name).is_file() for name in BOOK_NAMES):
            self.skipTest("The original photographs are kept locally and are not in the repository.")
        for name, digest in book_selection()["original_photograph_sha256"].items():
            self.assertEqual(hashlib.sha256((PART_B / "input" / name).read_bytes()).hexdigest(), digest, name)

    def test_estimation_and_reconstruction_do_not_write_files(self):
        def digest():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PART_B.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts}
        before = digest()
        files, truth = planar_views()
        sfm.find_correspondences([vr.decode_image(d, n) for d, n in files], truth[0][:4])
        sfm.reconstruct(book_tracks(), 14.5, 21.7)
        self.assertEqual(digest(), before)


class MarkerRenderingTests(unittest.TestCase):
    def test_markers_and_outline_are_drawn_only_for_placed_points(self):
        rgb = np.full((200, 200, 3), 128, np.uint8)
        points = np.array([[40, 40], [160, 40], [160, 160], [40, 160], [100, 100], [np.nan, np.nan],
                           [70, 120], [130, 80]], float)
        statuses = ["automatic", "corrected", "automatic", "selected", "automatic", None, "automatic", "automatic"]
        out = vr.render_points(rgb, points, statuses, active=4)
        self.assertEqual(out.shape, rgb.shape)
        self.assertFalse(np.array_equal(out, rgb))
        np.testing.assert_array_equal(rgb, 128)  # input untouched
        self.assertTrue(np.array_equal(out[100, 100], vr.MARKER_ACTIVE))  # active center dot
        no_outline = vr.render_points(rgb, np.where(np.arange(8)[:, None] == 0, np.nan, points), statuses, active=4)
        self.assertTrue(np.array_equal(no_outline[40, 100], rgb[40, 100]))  # no P1-P2 edge without P1


if __name__ == "__main__":
    unittest.main()
