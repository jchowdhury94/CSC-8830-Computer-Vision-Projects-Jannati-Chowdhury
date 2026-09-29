"""Q2 shared image sizing, coordinate, cache, and workflow regressions."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from Modules.Module_4 import thermal_preprocessing as prep
from Modules.Module_4.thermal_boundary import find_thermal_boundary
from Modules.Module_4.sam2_runtime import check_image_budget

PAGE = Path(__file__).resolve().parents[1] / 'Pages/app_module4.py'


def encoded(image):
    success, data = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert success
    return data.tobytes()


def fixture():
    image = np.full((1440, 1080, 3), (0, 0, 255), np.uint8)
    image[200:1200, 300:800] = (255, 180, 0)
    return image


class ThermalPreparationTests(unittest.TestCase):
    def test_sizing_and_interpolation(self):
        for shape, expected in [((1440, 1080, 3), (816, 612, 3)),
                                ((1080, 1440, 3), (612, 816, 3)),
                                ((1500, 1500, 3), (707, 707, 3)),
                                ((2000, 100, 3), (1500, 75, 3))]:
            with self.subTest(shape=shape), patch.object(prep, 'decode_thermal', return_value=np.zeros(shape, np.uint8)), patch.object(prep.cv2, 'resize', wraps=cv2.resize) as resize:
                image, original = prep.prepare_thermal_upload(b'upload')
                self.assertEqual(original, shape)
                self.assertEqual(image.shape, expected)
                self.assertEqual(resize.call_args.kwargs['interpolation'], cv2.INTER_AREA)
                self.assertLessEqual(image.shape[0] * image.shape[1], 500000)
                self.assertLessEqual(max(image.shape[:2]), 1500)
                check_image_budget(image.shape, 'cpu', 1500000)

    def test_small_and_exact_limits_unchanged(self):
        for shape in [(32, 48, 3), (1000, 500, 3), (1500, 100, 3)]:
            image = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
            with patch.object(prep.cv2, 'resize', wraps=cv2.resize) as resize:
                result, original = prep.prepare_thermal_upload(encoded(image))
            np.testing.assert_array_equal(result, image)
            self.assertEqual(original, shape)
            resize.assert_not_called()

    def test_decode_preserves_thermal_conventions(self):
        image = fixture()
        with patch.object(prep.cv2, 'imdecode', wraps=cv2.imdecode) as decode:
            result = prep.decode_thermal(encoded(image))
        self.assertEqual(decode.call_args.args[1], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        np.testing.assert_array_equal(result, image)

    def test_cache_replacement_policy_and_decode_failure(self):
        state = {}
        payload = encoded(fixture())
        with patch.object(prep, 'prepare_thermal_upload', wraps=prep.prepare_thermal_upload) as prepare:
            first = prep.prepared_thermal_upload(state, payload, 'one')[0]
            self.assertIs(first, prep.prepared_thermal_upload(state, payload, 'one')[0])
            self.assertEqual(prepare.call_count, 1)
            self.assertIsNot(first, prep.prepared_thermal_upload(state, payload, 'two')[0])
            with patch.dict(os.environ, {'SAM2_CPU_MAX_PIXELS': '100000'}):
                smaller = prep.prepared_thermal_upload(state, payload, 'two')[0]
                self.assertLessEqual(np.prod(smaller.shape[:2]), 100000)
            state.pop('module4_q2_prepared')
            prep.prepared_thermal_upload(state, payload, 'two')
            self.assertEqual(prepare.call_count, 4)
            bad = prep.prepared_thermal_upload(state, b'bad image', 'bad')
            self.assertIsNone(bad[0])
            self.assertIsNotNone(bad[2])
            prep.prepared_thermal_upload(state, b'bad image', 'bad')
            self.assertEqual(prepare.call_count, 5)

    def test_cpu_policy(self):
        for configured, budget in [('100000', 100000), ('0', 500000), ('2000000', 500000)]:
            with patch.dict(os.environ, {'SAM2_CPU_MAX_PIXELS': configured}):
                self.assertEqual(prep.preprocessing_policy()[2], budget)

    def test_click_mapping_and_iou(self):
        tree = ast.parse(PAGE.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('original_point', 'thermal_mask_iou')]
        namespace = {'np': np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PAGE), 'exec'), namespace)
        self.assertEqual(namespace['original_point']({'x': 262.5, 'y': 350, 'width': 525, 'height': 700}, (816, 612, 3)), (306, 408))
        mask = np.zeros((816, 612), np.uint8)
        mask[20:40, 30:50] = 255
        self.assertEqual(namespace['thermal_mask_iou'](mask, mask.copy()), 1.0)
        with self.assertRaises(ValueError):
            namespace['thermal_mask_iou'](mask, np.zeros((1440, 1080), np.uint8))

    def test_both_workflow_orders_and_q1_isolation(self):
        for first in ('classical', 'sam2'):
            with self.subTest(first=first):
                app = AppTest.from_file(str(PAGE))
                app.session_state['module4_section'] = 'thermal'
                app.session_state['module4_q2_bytes'] = encoded(fixture())
                app.session_state['module4_rgb_bytes'] = b'q1 preserved'
                app.session_state['module4_points'] = [(5, 7)]
                with patch.object(prep, 'prepare_thermal_upload', wraps=prep.prepare_thermal_upload) as prepare, patch('Modules.Module_4.thermal_boundary.find_thermal_boundary', wraps=find_thermal_boundary) as classical, patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), patch('Modules.Module_4.sam2_runtime.check_image_budget', wraps=check_image_budget) as budget, patch('Modules.Module_4.sam2_comparison.run_sam2') as sam2:
                    app.run()
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertEqual(prepare.call_count, 1)
                    image = app.session_state['module4_q2_prepared'][1]
                    before = image.copy()
                    sam2.return_value = {'binary_mask': np.zeros(image.shape[:2], np.uint8), 'boundary_overlay': image.copy(), 'predicted_mask_quality': .5, 'metadata': {'device': 'cpu', 'empty_mask': True}}
                    self.assertTrue(any('1080 × 1440 to 612 × 816' in c.value for c in app.caption))
                    if first == 'classical':
                        app.button(key='module4_q2_run').click().run()
                        app.button(key='module4_q2_setup_after_classical').click().run()
                    else:
                        app.button(key='module4_q2_setup').click().run()
                    app.session_state['module4_q2_points'] = [(100, 80), (500, 750)]
                    app.run()
                    sam2.assert_not_called()
                    # Results must render in the click execution, without an explicit rerun.
                    with patch('streamlit.rerun') as rerun:
                        app.button(key='module4_q2_run_sam2').click().run()
                        rerun.assert_not_called()
                    self.assertFalse(app.exception)
                    sam2.assert_called_once()
                    self.assertIs(app.session_state['module4_q2_sam2_results'], sam2.return_value)
                    self.assertEqual(app.session_state['module4_q2_sam2_identity'],
                                     app.session_state['module4_q2_sam2_inputs'])
                    self.assertTrue(any(s.value == 'SAM2 Results' for s in app.subheader))
                    for _ in range(2):
                        app.run()
                        self.assertFalse(app.exception)
                        sam2.assert_called_once()
                        self.assertTrue(any(s.value == 'SAM2 Results' for s in app.subheader))
                    if first == 'sam2':
                        app.button(key='module4_q2_classical_next').click().run()
                    self.assertFalse(app.exception)
                    self.assertIs(classical.call_args.args[0], image)
                    self.assertEqual(len(classical.call_args.args), 1)
                    self.assertIs(sam2.call_args.args[0], image)
                    np.testing.assert_array_equal(image, before)
                    self.assertEqual(sam2.call_args.args[1], (100, 80, 400, 670))
                    self.assertEqual(budget.call_args.args[0], (816, 612, 3))
                    results = app.session_state['module4_q2_results']
                    self.assertEqual(results['cleaned_human_mask'].shape, sam2.return_value['binary_mask'].shape)
                    self.assertEqual(results['boundary_overlay'].shape, image.shape)
                    self.assertEqual((results['width'], results['height']), (612, 816))
                    app.button(key='module4_q2_compare_after_results').click().run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any('Classical and SAM2: 612 × 816' in c.value for c in app.caption))
                    self.assertEqual(prepare.call_count, 1)
                    app.session_state['module4_section'] = 'home'
                    app.run()
                    app.session_state['module4_section'] = 'thermal'
                    app.run()
                    self.assertIs(app.session_state['module4_q2_prepared'][1], image)
                    # A different valid rectangle invalidates SAM2 without running it.
                    app.session_state['module4_q2_points'] = [(110, 90), (490, 740)]
                    app.run()
                    self.assertFalse(app.exception)
                    sam2.assert_called_once()
                    self.assertNotIn('module4_q2_sam2_results', app.session_state)
                    self.assertNotIn('module4_q2_sam2_identity', app.session_state)
                    self.assertNotIn('module4_q2_comparison', app.session_state)
                    self.assertIs(app.session_state['module4_q2_results'], results)
                    self.assertFalse(any(s.value == 'SAM2 Results' for s in app.subheader))
                    self.assertIsNotNone(app.button(key='module4_q2_run_sam2'))
                    with patch.object(prep, 'MAX_THERMAL_PIXELS', 100000):
                        app.run()
                        self.assertFalse(app.exception)
                        self.assertEqual(app.session_state['module4_q2_points'], [])
                        self.assertNotIn('module4_q2_results', app.session_state)
                        self.assertNotIn('module4_q2_sam2_results', app.session_state)
                        self.assertNotIn('module4_q2_comparison', app.session_state)
                    app.button(key='module4_q2_reset').click().run()
                    self.assertNotIn('module4_q2_prepared', app.session_state)
                    self.assertEqual(app.session_state['module4_rgb_bytes'], b'q1 preserved')
                    self.assertEqual(app.session_state['module4_points'], [(5, 7)])


if __name__ == '__main__':
    unittest.main()
