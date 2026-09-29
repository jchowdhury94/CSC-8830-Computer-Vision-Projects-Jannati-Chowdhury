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

    def test_selection_without_redundant_upload_preview(self):
        app = AppTest.from_file(str(PAGE))
        app.session_state['module4_section'] = 'rgb'
        app.session_state['module4_rgb_bytes'] = encoded(np.zeros((100, 100, 3), np.uint8))
        with patch('streamlit_image_coordinates.streamlit_image_coordinates', return_value=None) as coordinates:
            app.run()
            self.assertFalse(app.exception)
            coordinates.assert_called_once()
            self.assertEqual(len(app.get('image')), 0)
            self.assertFalse(any(b.label == 'Compare Results' for b in app.button))
            # Exercise the existing component-return path for both corners.
            coordinates.side_effect = [
                {'x': 10, 'y': 10, 'width': 100, 'height': 100}, None,
            ]
            app.run()
            self.assertEqual(app.session_state['module4_points'], [(10, 10)])
            self.assertEqual(len(app.get('image')), 0)
            coordinates.side_effect = None
            coordinates.return_value = {'x': 90, 'y': 90, 'width': 100, 'height': 100}
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['module4_points'], [(10, 10), (90, 90)])
            self.assertEqual(len(app.get('image')), 1)
            self.assertEqual(app.get('image')[0].proto.imgs[0].caption,
                             'Selected corners and GrabCut rectangle')
            self.assertFalse(app.button(key='module4_run_sam2').disabled)

    def test_contextual_methods_in_both_orders(self):
        for first in ('classical', 'sam2'):
            with self.subTest(first=first):
                image = np.full((100, 100, 3), 30, np.uint8)
                image[25:75, 30:70] = (190, 100, 70)
                app = AppTest.from_file(str(PAGE))
                app.session_state['module4_section'] = 'rgb'
                app.session_state['module4_rgb_bytes'] = encoded(image)
                app.run()
                app.session_state['module4_points'] = [(10, 10), (90, 90)]
                app.run()
                def action_keys():
                    return [b.key for b in app.button
                            if b.label in ('Run SAM2', 'Classical Method')]
                self.assertEqual(action_keys(), ['module4_detect', 'module4_run_sam2'])
                self.assertEqual([b.key for b in app.button if b.label == 'Reset RGB Experiment'],
                                 ['module4_reset_rgb'])
                self.assertFalse(any(b.label in ('Reset Classical Method', 'Reset SAM2')
                                     for b in app.button))
                with patch('Modules.Module_4.human_boundary.find_human_boundary', wraps=find_human_boundary) as classical, \
                     patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
                     patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
                     patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
                     patch('Modules.Module_4.sam2_comparison.run_sam2') as sam2:
                    sam2.return_value = {
                        'binary_mask': np.zeros(image.shape[:2], np.uint8),
                        'boundary_overlay': image.copy(), 'predicted_mask_quality': .5,
                        'metadata': {'device': 'cpu', 'empty_mask': True},
                    }
                    top = 'module4_detect' if first == 'classical' else 'module4_run_sam2'
                    lower = ('module4_run_sam2_after_classical' if first == 'classical'
                             else 'module4_detect_after_sam2')
                    prepared_before = app.session_state['module4_rgb_prepared']
                    self.assertFalse(any(h.value == 'Processing Settings' for h in app.subheader))
                    self.assertEqual(len(app.slider), 0)
                    app.button(key=top).click().run()
                    if first == 'classical':
                        classical.assert_not_called()
                        sam2.assert_not_called()
                        self.assertTrue(any(h.value == 'Processing Settings' for h in app.subheader))
                        self.assertIs(app.session_state['module4_rgb_prepared'], prepared_before)
                        self.assertEqual(len(app.slider), 4)
                        self.assertEqual(len(app.selectbox), 2)
                        app.slider(key='module4_gaussian_sigma').set_value(1.2).run()
                        classical.assert_not_called()
                        app.button(key='module4_apply_classical').click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(classical.call_count, int(first == 'classical'))
                    self.assertEqual(sam2.call_count, int(first == 'sam2'))
                    self.assertEqual(action_keys(), [lower])
                    reset_key = 'module4_reset_classical' if first == 'classical' else 'module4_reset_sam2'
                    self.assertIn(reset_key, [b.key for b in app.button])
                    self.assertEqual(app.button[-1].key, 'module4_reset_rgb_bottom')
                    title = 'Results' if first == 'classical' else 'SAM2 Results'
                    self.assertTrue(any(h.value == title for h in app.subheader))
                    self.assertFalse(any(b.label == 'Compare Results' for b in app.button))
                    app.run()
                    self.assertEqual(classical.call_count, int(first == 'classical'))
                    self.assertEqual(sam2.call_count, int(first == 'sam2'))
                    self.assertFalse(any(h.value == 'Processing Settings' for h in app.subheader))
                    app.button(key=lower).click().run()
                    if first == 'sam2':
                        classical.assert_not_called()
                        sam2.assert_called_once()
                        self.assertTrue(any(h.value == 'Processing Settings' for h in app.subheader))
                        self.assertEqual(app.slider(key='module4_canny_lower').value, 50)
                        self.assertIs(app.session_state['module4_rgb_prepared'], prepared_before)
                        self.assertEqual(len(app.slider), 4)
                        self.assertEqual(len(app.selectbox), 2)
                        app.slider(key='module4_gaussian_sigma').set_value(1.2).run()
                        classical.assert_not_called()
                        app.button(key='module4_apply_classical').click().run()
                    self.assertFalse(app.exception)
                    classical.assert_called_once()
                    sam2.assert_called_once()
                    self.assertEqual(classical.call_args.kwargs['gaussian_sigma'], 1.2)
                    self.assertEqual(len(app.slider), 0)
                    self.assertFalse(any(b.label == 'Apply Classical Method' for b in app.button))
                    self.assertEqual(action_keys(), [])
                    for title in ('Results', 'SAM2 Results'):
                        self.assertTrue(any(h.value == title for h in app.subheader))
                    self.assertEqual(app.session_state['module4_sam2_identity'],
                                     app.session_state['module4_sam2_inputs'])
                    self.assertEqual(app.session_state['module4_classical_identity'],
                                     app.session_state['module4_result_inputs'])
                    self.assertEqual([b.key for b in app.button if b.label == 'Compare Results'],
                                     ['module4_compare_after_results'])
                    self.assertFalse(app.button(key='module4_compare_after_results').disabled)
                    self.assertIs(classical.call_args.args[0], sam2.call_args.args[0])
                    self.assertEqual(classical.call_args.args[1], sam2.call_args.args[1])
                    app.button(key='module4_compare_after_results').click().run()
                    app.run()
                    self.assertTrue(any(h.value == 'Compare Results' for h in app.subheader))
                    self.assertFalse(app.exception)
                    classical.assert_called_once()
                    sam2.assert_called_once()
                    prepared = app.session_state['module4_rgb_prepared']
                    pixels = classical.call_args.args[0].copy()
                    shared_keys = ('module4_rgb_bytes', 'module4_rgb_filename',
                                   'module4_rgb_processing_identity', 'module4_image_digest',
                                   'module4_points', 'module4_shared_inputs',
                                   'module4_upload_generation', 'module4_click_generation')
                    shared = {key: app.session_state[key] for key in shared_keys
                              if key in app.session_state}
                    for method in ('classical', 'sam2'):
                        own_result = 'module4_results' if method == 'classical' else 'module4_sam2_results'
                        other_result = 'module4_sam2_results' if method == 'classical' else 'module4_results'
                        other_identity = 'module4_sam2_identity' if method == 'classical' else 'module4_classical_identity'
                        retained = app.session_state[other_result]
                        identity = app.session_state[other_identity]
                        calls = (classical.call_count, sam2.call_count)
                        app.button(key=f'module4_reset_{method}').click().run()
                        self.assertFalse(app.exception)
                        for key in (own_result, f'module4_{method}_identity', 'module4_comparison'):
                            self.assertNotIn(key, app.session_state)
                        self.assertIs(app.session_state[other_result], retained)
                        self.assertEqual(app.session_state[other_identity], identity)
                        self.assertIs(app.session_state['module4_rgb_prepared'], prepared)
                        for key, value in shared.items():
                            self.assertEqual(app.session_state[key], value)
                        self.assertFalse(any(b.label == 'Compare Results' for b in app.button))
                        self.assertEqual((classical.call_count, sam2.call_count), calls)
                        if method == 'classical':
                            self.assertEqual(app.slider(key='module4_gaussian_sigma').value, 1.2)
                            app.slider(key='module4_canny_lower').set_value(60).run()
                            self.assertTrue(app.session_state['module4_classical_config_open'])
                            action = 'module4_apply_classical'
                        else:
                            action = 'module4_run_sam2_after_classical'
                        app.button(key=action).click().run()
                        self.assertFalse(app.exception)
                        self.assertEqual(classical.call_count, calls[0] + int(method == 'classical'))
                        self.assertEqual(sam2.call_count, calls[1] + int(method == 'sam2'))
                        self.assertEqual(classical.call_args.kwargs['canny_lower'], 60)
                        self.assertIs(classical.call_args.args[0], sam2.call_args.args[0])
                        np.testing.assert_array_equal(classical.call_args.args[0], pixels)
                        self.assertIs(app.session_state['module4_rgb_prepared'], prepared)
                        self.assertEqual(classical.call_args.args[1], sam2.call_args.args[1])
                        self.assertFalse(app.button(key='module4_compare_after_results').disabled)
                        app.button(key='module4_compare_after_results').click().run()
                        self.assertTrue(any(h.value == 'Compare Results' for h in app.subheader))
                        self.assertEqual(app.button[-1].key, 'module4_reset_rgb_bottom')
                    app.session_state['module4_points'] = [(15, 15), (85, 85)]
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertEqual(classical.call_count, 2)
                    self.assertEqual(sam2.call_count, 2)
                    for key in ('module4_results', 'module4_sam2_results',
                                'module4_sam2_identity', 'module4_comparison'):
                        self.assertNotIn(key, app.session_state)
                    self.assertEqual(action_keys(), ['module4_detect', 'module4_run_sam2'])

    def test_full_reset_buttons_share_behavior_and_preserve_q2(self):
        for button in ('module4_reset_rgb', 'module4_reset_rgb_bottom'):
            with self.subTest(button=button):
                app = AppTest.from_file(str(PAGE))
                app.session_state['module4_section'] = 'rgb'
                app.session_state['module4_rgb_bytes'] = encoded(np.zeros((100, 100, 3), np.uint8))
                app.run()
                app.session_state['module4_points'] = [(10, 10), (90, 90)]
                app.run()
                app.button(key='module4_detect').click().run()
                app.button(key='module4_apply_classical').click().run()
                self.assertFalse(app.exception)
                q2 = {'module4_q2_results': {'mask': 'retained'},
                      'module4_q2_sam2_results': {'mask': 'retained SAM2'},
                      'module4_q2_points': [(1, 2), (3, 4)],
                      'module4_q2_comparison': ('classical', 'sam2'),
                      'module4_q2_thermal_bytes': b'unchanged'}
                for key, value in q2.items():
                    app.session_state[key] = value
                app.session_state['module4_sam2_results'] = {'old': True}
                app.session_state['module4_sam2_identity'] = 'old'
                app.session_state['module4_comparison'] = 'old'
                generation = app.session_state['module4_upload_generation']
                app.button(key=button).click().run()
                self.assertFalse(app.exception)
                for key in ('module4_rgb_bytes', 'module4_rgb_filename', 'module4_rgb_prepared',
                            'module4_results', 'module4_classical_identity',
                            'module4_sam2_results', 'module4_sam2_identity', 'module4_comparison'):
                    self.assertNotIn(key, app.session_state)
                self.assertEqual(app.session_state['module4_points'], [])
                self.assertFalse(app.session_state['module4_classical_config_open'])
                self.assertFalse(any(h.value == 'Processing Settings' for h in app.subheader))
                self.assertIsNone(app.session_state['module4_rgb_processing_identity'])
                self.assertEqual(app.session_state['module4_upload_generation'], generation + 1)
                for key, value in q2.items():
                    self.assertEqual(app.session_state[key], value)
                self.assertEqual([b.key for b in app.button if b.label == 'Reset RGB Experiment'],
                                 ['module4_reset_rgb'])

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
                app.button(key='module4_apply_classical').click().run()
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
                app.button(key='module4_run_sam2_after_classical').click().run()
                self.assertFalse(app.exception)
                budget.assert_called_once()
                self.assertEqual(budget.call_args.args[0], prepared_image.shape)
                self.assertIs(run.call_args.args[0], prepared_image)
                np.testing.assert_array_equal(run.call_args.args[0], prepared_image)
                self.assertEqual(run.call_args.args[1], selected_rectangle)
            app.button(key='module4_compare_after_results').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Classical and SAM2: 500 × 1000' in c.value for c in app.caption))
            prepared_cache = app.session_state['module4_rgb_prepared']
            processing_identity = app.session_state['module4_rgb_processing_identity']
            for method in ('classical', 'sam2'):
                app.button(key=f'module4_reset_{method}').click().run()
                self.assertFalse(app.exception)
                self.assertIs(app.session_state['module4_rgb_prepared'], prepared_cache)
                self.assertEqual(app.session_state['module4_rgb_processing_identity'], processing_identity)
                self.assertEqual(prepared_image.shape, (1000, 500, 3))
                self.assertEqual(prepare.call_count, 1)
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
