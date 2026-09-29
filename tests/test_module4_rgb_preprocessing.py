"""RGB upload sizing, session reuse, and Classical/SAM2 coordinate regression tests."""
import ast
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from Modules.Module_4 import rgb_preprocessing as prep
from Modules.Module_4.human_boundary import find_human_boundary
from Modules.Module_4.sam2_runtime import check_image_budget


PAGE = Path(__file__).resolve().parents[1] / 'Pages/app_module4.py'


def encoded(image):
    success, data = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert success
    return data.tobytes()


class RGBPreparationTests(unittest.TestCase):
    def test_large_portrait_and_landscape(self):
        for shape, expected in [((6376, 4554, 3), (836, 597, 3)),
                                ((4554, 6376, 3), (597, 836, 3))]:
            with self.subTest(shape=shape):
                source = np.zeros(shape, np.uint8)
                source[..., 0] = 90
                with patch.object(prep, 'decode_rgb', return_value=source), \
                     patch.object(prep.cv2, 'resize', wraps=cv2.resize) as resize:
                    result, original = prep.prepare_rgb_upload(b'upload')
                self.assertEqual(original, shape)
                self.assertEqual(result.shape, expected)
                self.assertEqual(resize.call_args.kwargs['interpolation'], cv2.INTER_AREA)
                self.assertLessEqual(result.shape[0] * result.shape[1], 500_000)
                self.assertLessEqual(max(result.shape[:2]), 1500)
                scale = min(1500 / max(shape[:2]), math.sqrt(500_000 / (shape[0] * shape[1])))
                self.assertLess(abs(result.shape[1] - shape[1] * scale), 1)
                check_image_budget(result.shape, 'cpu', 1_500_000)

    def test_small_and_exact_limit_are_unchanged(self):
        for shape in [(48, 64, 3), (1500, 120, 3), (1000, 500, 3)]:
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

    def test_square_longest_side_and_configured_budget(self):
        for shape, expected in [((1500, 1500, 3), (707, 707, 3)),
                                ((2000, 100, 3), (1500, 75, 3))]:
            with patch.object(prep, 'decode_rgb', return_value=np.zeros(shape, np.uint8)):
                result, _ = prep.prepare_rgb_upload(b'upload')
            self.assertEqual(result.shape, expected)
        for configured, expected_budget in [('100000', 100000), ('0', 500000), ('2000000', 500000)]:
            with patch.dict(os.environ, {'SAM2_CPU_MAX_PIXELS': configured}):
                self.assertEqual(prep.preprocessing_policy()[2], expected_budget)

    def test_policy_invalidates_prepared_cache(self):
        state = {}
        payload = encoded(np.zeros((800, 800, 3), np.uint8))
        first = prep.prepared_rgb_upload(state, payload, 'same')[0]
        with patch.object(prep, 'MAX_RGB_PIXELS', 100000):
            second = prep.prepared_rgb_upload(state, payload, 'same')[0]
        self.assertIsNot(first, second)
        self.assertLessEqual(second.shape[0] * second.shape[1], 100000)

    def test_preview_clicks_and_classical_outputs(self):
        # Execute only the existing pure click helper, without starting the page.
        tree = ast.parse(PAGE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'original_point')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(PAGE), 'exec'), namespace)
        self.assertEqual(namespace['original_point'](
            {'x': 250, 'y': 350, 'width': 500, 'height': 700}, (836, 597, 3)), (298, 418))
        image = np.full((1600, 800, 3), 30, np.uint8)
        image[300:1300, 200:600] = (190, 100, 70)
        prepared, _ = prep.prepare_rgb_upload(encoded(image))
        result = find_human_boundary(prepared, (50, 50, 400, 850))
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

    def test_page_reruns_and_identical_method_inputs(self):
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
            self.assertTrue(any('800 × 1600 to 500 × 1000' in c.value for c in app.caption))
            app.session_state['module4_points'] = [(50, 50), (450, 900)]
            app.run()
            with patch('Modules.Module_4.human_boundary.find_human_boundary', wraps=find_human_boundary) as classical:
                app.button(key='module4_detect').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(classical.call_args.args[0].shape, (1000, 500, 3))
                self.assertEqual(classical.call_args.args[1], (50, 50, 400, 850))
                self.assertEqual(app.session_state['module4_results']['boundary_overlay'].shape, (1000, 500, 3))
            self.assertEqual(prepare.call_count, 1)
            prepared_image = classical.call_args.args[0]
            selected_rectangle = classical.call_args.args[1]
            # Avoid weights and inference; inspect the arguments across the UI boundary.
            with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
                 patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
                 patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
                 patch('Modules.Module_4.sam2_runtime.check_image_budget', wraps=check_image_budget) as budget, \
                 patch('Modules.Module_4.sam2_comparison.run_sam2', return_value={}) as run:
                run.return_value = {'binary_mask': np.zeros(prepared_image.shape[:2], np.uint8),
                                    'boundary_overlay': prepared_image, 'predicted_mask_quality': .5,
                                    'metadata': {'device': 'cpu', 'empty_mask': False}}
                app.button(key='module4_run_sam2').click().run()
                self.assertFalse(app.exception)
                budget.assert_called_once()
                self.assertEqual(budget.call_args.args[0], prepared_image.shape)
                self.assertIs(run.call_args.args[0], prepared_image)
                np.testing.assert_array_equal(run.call_args.args[0], prepared_image)
                self.assertEqual(run.call_args.args[1], selected_rectangle)
            app.button(key='module4_compare').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Classical and SAM2: 500 × 1000' in c.value for c in app.caption))
            # Policy changes must clear selections and both methods' retained results.
            with patch.object(prep, 'MAX_RGB_PIXELS', 400000):
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['module4_points'], [])
                self.assertNotIn('module4_results', app.session_state)
                self.assertNotIn('module4_sam2_results', app.session_state)
            app.button(key='module4_reset_rgb').click().run()
            self.assertNotIn('module4_rgb_prepared', app.session_state)


if __name__ == '__main__':
    unittest.main()
