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

    CURRENT = ':red[:material/arrow_forward: Current step]'
    DONE = ':green[:material/check_circle: Completed]'
    STEPS = ['Step 1 · Upload Thermal Image', 'Step 2 · Classical Method', 'Step 3 · SAM2', 'Step 4 · Compare Results']

    @staticmethod
    def keys(app):
        return [b.key for b in app.button if not (b.key or '').startswith('nav')]

    def marks(self, app):
        names = {self.CURRENT: 'current', self.DONE: 'done'}
        out = []
        for caption in (c.value for c in app.caption):
            if caption in names:
                out.append(names[caption])
            elif caption.startswith(':material/lock:'):
                out.append('locked')
        return out

    def open_page(self, with_image=True):
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['module4_section'] = 'thermal'
        if with_image:
            app.session_state['module4_q2_bytes'] = encoded(fixture())
        app.run()
        self.assertFalse(app.exception)
        return app

    @staticmethod
    def fake_sam2(image):
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:700, 150:450] = 255
        return {'binary_mask': mask, 'boundary_overlay': image.copy(), 'predicted_mask_quality': .5,
                'metadata': {'device': 'cpu', 'empty_mask': False, 'model_name': 'sam2.1_hiera_tiny', 'prompt_type': 'box-only'}}

    def run_sam2(self, app, image):
        with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
             patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
             patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
             patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.fake_sam2(image)) as run:
            app.button(key='module4_q2_run_sam2').click().run()
        self.assertFalse(app.exception)
        return run

    def test_steps_are_sequential_and_locked_steps_hide_their_content(self):
        app = self.open_page(with_image=False)
        self.assertEqual([t.value for t in app.title], ['Human Boundary Detection in Thermal Images'])
        self.assertEqual([h.value for h in app.subheader], self.STEPS)
        self.assertEqual(self.marks(app), ['current', 'locked', 'locked', 'locked'])
        captions = [c.value for c in app.caption]
        for n in (2, 3, 4):
            self.assertIn(f':material/lock: Locked. Complete Step {n - 1} to continue.', captions)
        self.assertTrue(any('Maximum upload size: 200 MB' in c for c in captions))
        self.assertIn('Upload a false-color thermal image of one person, preferably with warm body colors against a blue or purple background.', [m.value for m in app.markdown])
        self.assertFalse(any('actual temperature' in m.value for m in app.markdown))
        self.assertEqual(self.keys(app), ['module4_q2_reset'])
        self.assertEqual(len(app.expander) + len(app.get('image')), 0)
        text = ' '.join(m.value for m in app.markdown).lower()
        for word in ('question 2', 'module 4 –', 'set up sam2', 'magenta'):
            self.assertNotIn(word, text)
        app = self.open_page()
        self.assertEqual(self.marks(app), ['done', 'current', 'locked', 'locked'])
        self.assertTrue(any('1080 × 1440 to 612 × 816' in c.value for c in app.caption))
        self.assertEqual(self.keys(app), ['module4_q2_clear_image', 'module4_q2_run', 'module4_q2_reset'])

    def test_classical_runs_without_a_rectangle_with_fixed_parameters(self):
        from Modules.Module_4.thermal_boundary import CLOSING_ITERATIONS, CLOSING_KERNEL_SIZE, HSV_RANGES
        self.assertEqual(HSV_RANGES, (((0, 60, 60), (85, 255, 255)), ((170, 60, 60), (179, 255, 255))))
        self.assertEqual((CLOSING_KERNEL_SIZE, CLOSING_ITERATIONS), (3, 1))
        app = self.open_page()
        self.assertEqual([e.label for e in app.expander], ['View Processing Parameters'])
        self.assertFalse(app.expander[0].proto.expanded)
        table = next(t.value for t in app.table if 'HSV hue range' in list(t.value.index))
        self.assertEqual({k: v['Value'] for k, v in table.iterrows()}, {
            'HSV hue range': '0–85 and 170–179', 'Minimum saturation': '60', 'Minimum value': '60',
            'Morphological closing': '3 × 3 elliptical kernel', 'Closing iterations': '1', 'Connected components': '8-connectivity'})
        self.assertEqual(len(app.slider) + len(app.selectbox) + len(app.checkbox), 0)  # nothing adjustable
        image = app.session_state['module4_q2_prepared'][1]
        self.assertEqual(app.session_state['module4_q2_points'], [])
        with patch('Modules.Module_4.thermal_boundary.find_thermal_boundary', wraps=find_thermal_boundary) as classical:
            app.button(key='module4_q2_run').click().run()
        self.assertFalse(app.exception)
        self.assertIs(classical.call_args.args[0], image)
        self.assertEqual(len(classical.call_args.args), 1)
        results = app.session_state['module4_q2_results']
        self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked'])
        # the supplementary measurements are no longer shown, though they are still calculated
        self.assertEqual([list(t.value.index) for t in app.table if 'Foreground components' in list(t.value.index)], [])
        self.assertGreater(results['foreground_components'], 0)
        self.assertGreater(results['selected_component_area'], 0)
        self.assertNotIn('Image occupancy', ' '.join(c.value for c in app.caption))
        self.assertTrue(any(m.value == 'The human boundary is extracted from the cleaned mask and displayed in black.' for m in app.markdown))
        self.assertEqual(len(app.metric), 0)
        self.assertIn('Initial HSV Mask', [i.proto.imgs[0].caption for i in app.get('image')])
        self.assertEqual((results['width'], results['height']), (612, 816))
        self.assertIn('module4_q2_reset_classical', self.keys(app))

    def test_rectangle_sam2_comparison_and_resets(self):
        app = self.open_page()
        app.session_state['module4_rgb_bytes'] = b'rgb preserved'
        app.session_state['module4_points'] = [(5, 7)]
        image = app.session_state['module4_q2_prepared'][1]
        before = image.copy()
        app.button(key='module4_q2_run').click().run()
        results = app.session_state['module4_q2_results']
        self.assertEqual([e.label for e in app.expander], ['View Processing Parameters', 'How to select the rectangle'])
        self.assertFalse(any(e.proto.expanded for e in app.expander))
        self.assertTrue(app.button(key='module4_q2_reset_rectangle').disabled)
        self.assertNotIn('module4_q2_run_sam2', self.keys(app))  # no rectangle yet
        app.session_state['module4_q2_points'] = [(100, 80), (500, 750)]
        app.run()
        self.assertIn('module4_q2_run_sam2', self.keys(app))
        self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked'])
        run = self.run_sam2(app, image)
        # SAM2 receives only the processing image and the selected rectangle
        run.assert_called_once()
        self.assertIs(run.call_args.args[0], image)
        self.assertEqual(run.call_args.args[1], (100, 80, 400, 670))
        np.testing.assert_array_equal(image, before)
        self.assertEqual(self.marks(app), ['done', 'done', 'done', 'current'])
        # no model table or explanatory notes under the SAM2 results
        self.assertEqual([t for t in app.table if 'Device used' in list(t.value.index)], [])
        captions = ' '.join(c.value for c in app.caption)
        self.assertNotIn('mask-quality', captions)
        self.assertNotIn('Maximum upload', ' '.join(m.value for m in app.markdown))
        self.assertNotIn('module4_q2_comparison', app.session_state)
        app.button(key='module4_q2_compare').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.marks(app), ['done'] * 4)
        sam2_mask = app.session_state['module4_q2_sam2_results']['binary_mask']
        classical_mask = results['cleaned_human_mask']
        foreground, other = classical_mask != 0, sam2_mask != 0
        expected = np.count_nonzero(foreground & other) / np.count_nonzero(foreground | other)
        markdown = [m.value for m in app.markdown]
        self.assertIn('##### Mask Agreement (IoU)', markdown)
        self.assertIn(f'### {expected:.4f}', markdown)
        self.assertEqual([t for t in app.table if 'Intersection over Union (IoU)' in list(t.value.index)], [])
        captions = [i.proto.imgs[0].caption for i in app.get('image')]
        for caption in ('Classical Method — Cleaned Human Mask', 'SAM2 — Foreground Mask',
                        'Classical Human Boundary', 'SAM2 Human Boundary'):
            self.assertIn(caption, captions)
        text = ' '.join(w.value for w in app.markdown)
        for removed in ('same processing image', 'Compare the white', 'Compare the black'):
            self.assertNotIn(removed, text)
        self.assertIn('- SAM2 results depend on image quality, background complexity, and rectangle selection.', text)
        self.assertIn('- Measuring actual segmentation accuracy requires a ground-truth mask.', text)
        for _ in range(2):
            app.run()
            self.assertFalse(app.exception)
            run.assert_called_once()
        # Reset SAM2: only SAM2 and the comparison go
        app.button(key='module4_q2_reset_sam2').click().run()
        self.assertNotIn('module4_q2_sam2_results', app.session_state)
        self.assertNotIn('module4_q2_comparison', app.session_state)
        self.assertIs(app.session_state['module4_q2_results'], results)
        self.assertEqual(app.session_state['module4_q2_points'], [(100, 80), (500, 750)])
        self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked'])
        run = self.run_sam2(app, image)
        app.button(key='module4_q2_compare').click().run()
        # Reset Classical Method: Step 2 current again, rectangle and SAM2 kept; no SAM2 rerun when restored
        kept = app.session_state['module4_q2_sam2_results']
        app.button(key='module4_q2_reset_classical').click().run()
        self.assertNotIn('module4_q2_results', app.session_state)
        self.assertNotIn('module4_q2_comparison', app.session_state)
        self.assertIs(app.session_state['module4_q2_sam2_results'], kept)
        self.assertEqual(self.marks(app), ['done', 'current', 'locked', 'locked'])
        app.button(key='module4_q2_run').click().run()
        self.assertEqual(self.marks(app), ['done', 'done', 'done', 'current'])
        run.assert_called_once()
        # Reset Rectangle: rectangle, SAM2 and comparison go; classical stays
        app.button(key='module4_q2_compare').click().run()
        app.button(key='module4_q2_reset_rectangle').click().run()
        self.assertEqual(app.session_state['module4_q2_points'], [])
        for key in ('module4_q2_sam2_results', 'module4_q2_comparison'):
            self.assertNotIn(key, app.session_state)
        self.assertIn('module4_q2_results', app.session_state)
        self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked'])
        # a different rectangle invalidates SAM2 without running it
        app.session_state['module4_q2_points'] = [(110, 90), (490, 740)]
        app.run()
        run = self.run_sam2(app, image)
        app.session_state['module4_q2_points'] = [(100, 80), (500, 750)]
        app.run()
        self.assertNotIn('module4_q2_sam2_results', app.session_state)
        run.assert_called_once()
        self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked'])
        # a processing-policy change (like a new image) invalidates everything
        with patch.object(prep, 'MAX_THERMAL_PIXELS', 100000):
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['module4_q2_points'], [])
            self.assertNotIn('module4_q2_results', app.session_state)
            self.assertNotIn('module4_q2_sam2_results', app.session_state)
        # full reset
        app.button(key='module4_q2_run').click().run()
        app.button(key='module4_q2_reset').click().run()
        self.assertFalse(app.exception)
        self.assertNotIn('module4_q2_prepared', app.session_state)
        self.assertNotIn('module4_q2_bytes', app.session_state)
        self.assertEqual(self.marks(app), ['current', 'locked', 'locked', 'locked'])
        self.assertEqual(self.keys(app), ['module4_q2_reset'])
        self.assertEqual(app.session_state['module4_rgb_bytes'], b'rgb preserved')
        self.assertEqual(app.session_state['module4_points'], [(5, 7)])

    def test_sam2_model_resource_from_an_earlier_module_copy_is_rebuilt(self):
        import streamlit as st
        from Modules.Module_4.sam2_comparison import SAM2ModelResource
        st.cache_resource.clear()
        genuine = SAM2ModelResource(object(), 'sam2.1_hiera_tiny', 'config', Path('/unused'), 'cpu')
        app = self.open_page()
        image = app.session_state['module4_q2_prepared'][1]
        app.button(key='module4_q2_run').click().run()
        app.session_state['module4_q2_points'] = [(100, 80), (500, 750)]
        app.run()
        with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
             patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
             patch('Modules.Module_4.sam2_comparison.load_sam2_model', side_effect=[object(), genuine]) as load, \
             patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.fake_sam2(image)) as run:
            app.button(key='module4_q2_run_sam2').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(load.call_count, 2)
        self.assertIs(run.call_args.args[2], genuine)
        self.assertIn('module4_q2_sam2_results', app.session_state)
        self.assertNotIn('module4_q2_sam2_error', app.session_state)
        st.cache_resource.clear()

    def test_black_outlines_leave_the_segmentation_unchanged(self):
        app = self.open_page()
        image = app.session_state['module4_q2_prepared'][1]
        app.button(key='module4_q2_run').click().run()
        results = app.session_state['module4_q2_results']
        direct = find_thermal_boundary(image)
        np.testing.assert_array_equal(results['cleaned_human_mask'], direct['cleaned_human_mask'])
        np.testing.assert_array_equal(results['initial_hsv_mask'], direct['initial_hsv_mask'])
        self.assertEqual(results['selected_component_area'], direct['selected_component_area'])
        expected = image.copy()
        cv2.drawContours(expected, [direct['contour']], -1, (0, 0, 0), 5)  # same contour, 5 px, in black
        np.testing.assert_array_equal(results['boundary_overlay'], expected)
        app.session_state['module4_q2_points'] = [(100, 80), (500, 750)]
        app.run()
        run = self.run_sam2(app, image)
        sam2 = app.session_state['module4_q2_sam2_results']
        np.testing.assert_array_equal(sam2['binary_mask'], run.return_value['binary_mask'])
        contours, _ = cv2.findContours(sam2['binary_mask'].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = image.copy()
        cv2.drawContours(expected, contours, -1, (0, 0, 0), 5)
        np.testing.assert_array_equal(sam2['boundary_overlay'], expected)
        text = ' '.join([*(m.value for m in app.markdown), *(c.value for c in app.caption)])
        self.assertNotIn('magenta', text)

    def test_rectangle_instructions_live_only_inside_the_collapsed_expander(self):
        app = self.open_page()
        app.button(key='module4_q2_run').click().run()
        self.assertEqual(len(app.expander), 2)
        guide = app.expander[1]
        self.assertEqual(guide.label, 'How to select the rectangle')
        self.assertFalse(guide.proto.expanded)
        inside = [m.value for m in guide.markdown]
        self.assertEqual(sum('Click two opposite corners' in v for v in inside), 1)
        self.assertEqual(sum('small margin around the body' in v for v in inside), 1)
        everywhere = [m.value for m in app.markdown]
        self.assertEqual(sum('Click two opposite corners' in v for v in everywhere), 1)  # not repeated outside the expander
        self.assertEqual(sum('small margin around the body' in v for v in everywhere), 1)
        self.assertFalse(any('used only by SAM2' in v for v in everywhere))
        self.assertIn('SAM2 uses the selected rectangle to segment the person from the background.', everywhere)
        self.assertIn('##### Select Person Rectangle', everywhere)
        self.assertIn('Select the first corner.', [i.value for i in app.info])
        self.assertNotIn('module4_q2_run_sam2', self.keys(app))
        self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked'])


if __name__ == '__main__':
    unittest.main()
