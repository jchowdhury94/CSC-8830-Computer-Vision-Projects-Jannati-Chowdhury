"""RGB upload sizing, session reuse, and Classical/SAM2 coordinate regression tests."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from Modules.Module_4 import rgb_preprocessing as prep
from Modules.Module_4.human_boundary import find_human_boundary


PAGE = Path(__file__).resolve().parents[1] / 'Pages/app_module4.py'


def encoded(image):
    success, data = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert success
    return data.tobytes()


class RGBPreparationTests(unittest.TestCase):
    def test_large_portrait_and_landscape(self):
        for shape, expected in [((6376, 4554, 3), (1500, 1071, 3)),
                                ((4554, 6376, 3), (1071, 1500, 3))]:
            with self.subTest(shape=shape):
                source = np.zeros(shape, np.uint8)
                source[..., 0] = 90
                with patch.object(prep, 'decode_rgb', return_value=source), \
                     patch.object(prep.cv2, 'resize', wraps=cv2.resize) as resize:
                    result, original = prep.prepare_rgb_upload(b'upload')
                self.assertEqual(original, shape)
                self.assertEqual(result.shape, expected)
                self.assertEqual(resize.call_args.kwargs['interpolation'], cv2.INTER_AREA)
                self.assertLessEqual(abs(result.shape[1] - shape[1] * 1500 / max(shape)), .5)

    def test_small_and_exact_limit_are_unchanged(self):
        for shape in [(48, 64, 3), (1500, 120, 3)]:
            image = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
            with patch.object(prep.cv2, 'resize', wraps=cv2.resize) as resize:
                result, original = prep.prepare_rgb_upload(encoded(image))
            np.testing.assert_array_equal(result, image)
            self.assertEqual(original, shape)
            resize.assert_not_called()

    def test_reuse_replacement_and_clear(self):
        state = {}
        payload = encoded(np.zeros((32, 48, 3), np.uint8))
        with patch.object(prep, 'prepare_rgb_upload', wraps=prep.prepare_rgb_upload) as prepare:
            first = prep.prepared_rgb_upload(state, payload, 'first')[0]
            self.assertIs(first, prep.prepared_rgb_upload(state, payload, 'first')[0])
            self.assertEqual(prepare.call_count, 1)
            prep.prepared_rgb_upload(state, payload, 'replacement')
            self.assertEqual(prepare.call_count, 2)
            state.pop('module4_rgb_prepared')
            prep.prepared_rgb_upload(state, payload, 'replacement')
            self.assertEqual(prepare.call_count, 3)

    def test_coordinate_mapping(self):
        self.assertEqual(prep.map_rectangle((100, 200, 400, 900),
                         (1500, 1071, 3), (6376, 4554, 3)),
                         (425, 850, 1702, 3826))
        self.assertEqual(prep.map_rectangle((10, 20, 30, 40),
                         (100, 100, 3), (100, 100, 3)), (10, 20, 30, 40))
        # Selection may touch the exclusive image edge without exceeding it.
        mapped = prep.map_rectangle((100, 200, 971, 1300),
                                    (1500, 1071, 3), (6376, 4554, 3))
        self.assertEqual(mapped[0] + mapped[2], 4554)
        self.assertEqual(mapped[1] + mapped[3], 6376)

    def test_preview_clicks_and_classical_outputs(self):
        # Execute only the existing pure click helper, without starting the page.
        tree = ast.parse(PAGE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'original_point')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(PAGE), 'exec'), namespace)
        self.assertEqual(namespace['original_point'](
            {'x': 250, 'y': 350, 'width': 500, 'height': 700}, (1500, 1071, 3)), (535, 750))
        image = np.full((1600, 800, 3), 30, np.uint8)
        image[300:1300, 200:600] = (190, 100, 70)
        prepared, _ = prep.prepare_rgb_upload(encoded(image))
        result = find_human_boundary(prepared, (100, 100, 550, 1300))
        for key in ('canny_edges', 'raw_mask', 'cleaned_mask'):
            self.assertEqual(result[key].shape, prepared.shape[:2])
        self.assertEqual(result['boundary_overlay'].shape, prepared.shape)
        self.assertIsNotNone(result['contour'])
        x, y, w, h = cv2.boundingRect(result['contour'])
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, prepared.shape[1])
        self.assertLessEqual(y + h, prepared.shape[0])
        expected = prepared.copy()
        cv2.drawContours(expected, [result['contour']], -1, (0, 255, 0), 2)
        np.testing.assert_array_equal(result['boundary_overlay'], expected)

    def test_page_reruns_and_sam2_original_inputs(self):
        image = np.full((1600, 800, 3), 50, np.uint8)
        image[300:1200, 200:500] = (190, 100, 70)
        payload = encoded(image)
        app = AppTest.from_file(str(PAGE))
        app.session_state['module4_section'] = 'rgb'
        app.session_state['module4_rgb_bytes'] = payload
        with patch.object(prep, 'prepare_rgb_upload', wraps=prep.prepare_rgb_upload) as prepare:
            app.run()
            self.assertFalse(app.exception)
            app.run()
            self.assertEqual(prepare.call_count, 1)
            self.assertTrue(any('800 × 1600 to 750 × 1500' in c.value for c in app.caption))
            app.session_state['module4_points'] = [(100, 100), (500, 1200)]
            app.run()
            with patch('Modules.Module_4.human_boundary.find_human_boundary', wraps=find_human_boundary) as classical:
                app.button(key='module4_detect').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(classical.call_args.args[0].shape, (1500, 750, 3))
                self.assertEqual(classical.call_args.args[1], (100, 100, 400, 1100))
                self.assertEqual(app.session_state['module4_results']['boundary_overlay'].shape, (1500, 750, 3))
            self.assertEqual(prepare.call_count, 1)
            original_rectangle = prep.map_rectangle((100, 100, 400, 1100), (1500, 750, 3), image.shape)
            # Avoid weights and inference; inspect the arguments across the UI boundary.
            with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
                 patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
                 patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
                 patch('Modules.Module_4.sam2_runtime.check_image_budget') as budget, \
                 patch('Modules.Module_4.sam2_comparison.run_sam2', return_value={}) as run:
                run.return_value = {'binary_mask': np.zeros(image.shape[:2], np.uint8),
                                    'boundary_overlay': image, 'predicted_mask_quality': .5,
                                    'metadata': {'device': 'cpu', 'empty_mask': False}}
                app.button(key='module4_run_sam2').click().run()
                self.assertFalse(app.exception)
                budget.assert_called_once()
                self.assertEqual(budget.call_args.args[0], image.shape)
                np.testing.assert_array_equal(run.call_args.args[0], image)
                self.assertEqual(run.call_args.args[1], original_rectangle)
            app.button(key='module4_compare').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Classical: 750 × 1500' in c.value and 'SAM2: original 800 × 1600' in c.value for c in app.caption))
            app.button(key='module4_reset_rgb').click().run()
            self.assertNotIn('module4_rgb_prepared', app.session_state.filtered_state)


if __name__ == '__main__':
    unittest.main()
