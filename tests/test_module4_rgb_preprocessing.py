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

    def open_page(self, image=None, points=None):
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['module4_section'] = 'rgb'
        if image is not None:
            app.session_state['module4_rgb_bytes'] = encoded(image)
        app.run()
        if points:
            app.session_state['module4_points'] = points
            app.run()
        return app

    @staticmethod
    def keys(app):
        return [b.key for b in app.button if not (b.key or '').startswith('nav')]

    @staticmethod
    def person():
        image = np.full((100, 100, 3), 30, np.uint8)
        image[25:75, 30:70] = (190, 100, 70)
        return image

    @staticmethod
    def fake_sam2(image):
        return {'binary_mask': np.zeros(image.shape[:2], np.uint8), 'boundary_overlay': image.copy(),
                'predicted_mask_quality': .5,
                'metadata': {'device': 'cpu', 'empty_mask': False, 'model_name': 'sam2.1_hiera_tiny',
                             'prompt_type': 'box-only'}}

    CURRENT = ':red[:material/arrow_forward: Current step]'
    DONE = ':green[:material/check_circle: Completed]'

    @staticmethod
    def lock(n):
        return f':material/lock: Locked. Complete Step {n} to continue.'

    def marks(self, app):
        """The status line of each step, in order: 'locked', 'current' or 'done'."""
        names = {self.CURRENT: 'current', self.DONE: 'done'}
        out = []
        for caption in (c.value for c in app.caption):
            if caption in names:
                out.append(names[caption])
            elif caption.startswith(':material/lock:'):
                out.append('locked')
        return out

    def run_classical(self, app):
        app.button(key='module4_apply_classical').click().run()
        self.assertFalse(app.exception)

    def run_sam2(self, app, image):
        with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
             patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
             patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
             patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.fake_sam2(image)) as run:
                app.button(key='module4_run_sam2').click().run()
        self.assertFalse(app.exception)
        return run

    def test_five_steps_visible_and_only_step_one_current_before_upload(self):
        app = self.open_page()
        self.assertFalse(app.exception)
        self.assertEqual([t.value for t in app.title], ['Human Boundary Detection in RGB Images'])
        self.assertEqual([h.value for h in app.subheader], ['Step 1 · Upload RGB Image', 'Step 2 · Select Person Rectangle', 'Step 3 · Classical Method', 'Step 4 · SAM2', 'Step 5 · Compare Results'])
        self.assertEqual(self.marks(app), ['current', 'locked', 'locked', 'locked', 'locked'])
        captions = [c.value for c in app.caption]
        for n in (2, 3, 4, 5):
            self.assertIn(self.lock(n - 1), captions)
        self.assertTrue(any('Maximum upload size' in c for c in captions))
        # a locked step shows only its heading and message: no controls, images or expander
        self.assertEqual(len(app.slider) + len(app.selectbox) + len(app.checkbox), 0)
        self.assertEqual(len(app.expander), 0)
        self.assertEqual(len(app.get('image')), 0)
        self.assertEqual(self.keys(app), ['module4_reset_rgb'])

    def test_upload_completes_step_one_and_makes_step_two_current(self):
        app = self.open_page(self.person())
        self.assertFalse(app.exception)
        self.assertIn('Upload an RGB image of one person, preferably fully visible and distinguishable from the background.',
                      [m.value for m in app.markdown])
        # all rectangle instructions are inside the collapsed expander, once, and nowhere else
        guide = app.expander[0]
        self.assertFalse(guide.proto.expanded)
        for phrase in ('Click two opposite corners', 'small margin around the body', 'Leave some background outside the rectangle'):
            self.assertEqual(sum(phrase in m.value for m in guide.markdown), 1)
            self.assertEqual(sum(phrase in m.value for m in app.markdown), 1)
        self.assertEqual(self.marks(app), ['done', 'current', 'locked', 'locked', 'locked'])
        self.assertEqual([e.label for e in app.expander], ['How to select the rectangle'])
        self.assertEqual(len(app.get('image')), 1)  # the instructional example inside the collapsed expander
        for key in ('module4_apply_classical', 'module4_run_sam2', 'module4_compare'):
            self.assertNotIn(key, self.keys(app))
        self.assertTrue(app.button(key='module4_reset_rectangle').disabled)

    def test_rectangle_selection_unlocks_step_three_only(self):
        app = AppTest.from_file(str(PAGE))
        app.session_state['module4_section'] = 'rgb'
        app.session_state['module4_rgb_bytes'] = encoded(np.zeros((100, 100, 3), np.uint8))
        with patch('streamlit_image_coordinates.streamlit_image_coordinates', return_value=None) as coordinates:
            app.run()
            self.assertFalse(app.exception)
            coordinates.assert_called_once()
            self.assertNotIn('module4_apply_classical', self.keys(app))
            self.assertTrue(any(i.value == 'Select the first corner.' for i in app.info))
            coordinates.side_effect = [{'x': 10, 'y': 10, 'width': 100, 'height': 100}, None]
            app.run()
            self.assertEqual(app.session_state['module4_points'], [(10, 10)])
            self.assertTrue(any(i.value == 'Select the second corner.' for i in app.info))
            coordinates.side_effect = None
            coordinates.return_value = {'x': 90, 'y': 90, 'width': 100, 'height': 100}
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['module4_points'], [(10, 10), (90, 90)])
            self.assertIn('Selected corners and GrabCut rectangle', [i.proto.imgs[0].caption for i in app.get('image')])
            self.assertFalse(app.button(key='module4_reset_rectangle').disabled)
            self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked', 'locked'])
            self.assertIn('module4_apply_classical', self.keys(app))
            self.assertNotIn('module4_run_sam2', self.keys(app))
            self.assertNotIn('module4_compare', self.keys(app))
            self.assertEqual(len(app.slider), 4)
            self.assertEqual(app.slider(key='module4_gaussian_sigma').value, 0.8)
            self.assertFalse(app.checkbox(key='module4_cleanup_enabled').value)

    def test_steps_progress_in_order_and_resets_invalidate_dependants(self):
        image = self.person()
        app = self.open_page(image, [(10, 10), (90, 90)])
        prepared = app.session_state['module4_rgb_prepared']
        with patch('Modules.Module_4.human_boundary.find_human_boundary', wraps=find_human_boundary) as classical:
            app.slider(key='module4_gaussian_sigma').set_value(1.2).run()
            classical.assert_not_called()
            self.run_classical(app)
            classical.assert_called_once()
            self.assertEqual(classical.call_args.kwargs['gaussian_sigma'], 1.2)
            self.assertEqual(self.marks(app), ['done', 'done', 'done', 'current', 'locked'])
            self.assertNotIn('module4_compare', self.keys(app))
            sam2 = self.run_sam2(app, image)
            sam2.assert_called_once()
            self.assertIs(classical.call_args.args[0], sam2.call_args.args[0])  # the same processing image
            self.assertEqual(classical.call_args.args[1], sam2.call_args.args[1])  # and rectangle; SAM2 gets no classical output
            self.assertIs(app.session_state['module4_rgb_prepared'], prepared)
            self.assertEqual(self.marks(app), ['done', 'done', 'done', 'done', 'current'])
            app.button(key='module4_compare').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(self.marks(app), ['done'] * 5)
            captions = [i.proto.imgs[0].caption for i in app.get('image')]
            for caption in ('Classical Method — GrabCut Foreground Mask', 'SAM2 — Foreground Mask',
                            'Classical Human Boundary', 'SAM2 Human Boundary'):
                self.assertIn(caption, captions)
            self.assertTrue(any('Observations & Limitations' in m.value for m in app.markdown))
            # no model table or explanatory notes under the SAM2 results, and no contour measurement table after classical
            self.assertEqual([t for t in app.table if 'Device used' in list(t.value.index)], [])
            self.assertEqual([t for t in app.table if 'Selected contour area' in list(t.value.index)], [])
            texts = ' '.join([*(m.value for m in app.markdown), *(c.value for c in app.caption)])
            for removed in ('mask-quality', 'Compare the white', 'Compare the green', 'same processing image and selected rectangle'):
                self.assertNotIn(removed, texts)
            self.assertIn('The human boundary is extracted from the largest external contour of the segmentation mask.', texts)
            self.assertIn('- Measuring actual segmentation accuracy requires a ground-truth mask.', texts)
            self.assertEqual(texts.count('SAM2 results depend on image quality'), 1)  # consolidated into Step 5
            # resetting SAM2 only discards SAM2 and the comparison
            retained = app.session_state['module4_results']
            app.button(key='module4_reset_sam2').click().run()
            self.assertIs(app.session_state['module4_results'], retained)
            self.assertNotIn('module4_sam2_results', app.session_state)
            self.assertNotIn('module4_comparison', app.session_state)
            self.assertEqual(self.marks(app), ['done', 'done', 'done', 'current', 'locked'])
            sam2 = self.run_sam2(app, image)
            app.button(key='module4_compare').click().run()
            # resetting the classical method locks Step 4 and 5 again but keeps the still-valid SAM2 result
            kept = app.session_state['module4_sam2_results']
            app.button(key='module4_reset_classical').click().run()
            self.assertNotIn('module4_results', app.session_state)
            self.assertNotIn('module4_comparison', app.session_state)
            self.assertIs(app.session_state['module4_sam2_results'], kept)
            self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked', 'locked'])
            self.assertNotIn('module4_run_sam2', self.keys(app))
            calls = (classical.call_count, sam2.call_count)
            self.run_classical(app)
            self.assertEqual(classical.call_count, calls[0] + 1)
            self.assertEqual(sam2.call_count, calls[1])  # SAM2 was not run again
            self.assertEqual(self.marks(app), ['done', 'done', 'done', 'done', 'current'])
            # changing a classical setting discards the classical result and re-locks the later steps
            app.slider(key='module4_canny_lower').set_value(60).run()
            self.assertNotIn('module4_results', app.session_state)
            self.assertIn('module4_sam2_results', app.session_state)
            self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked', 'locked'])
            # changing the rectangle discards everything that depended on it
            app.session_state['module4_points'] = [(15, 15), (85, 85)]
            app.run()
            for key in ('module4_results', 'module4_sam2_results', 'module4_comparison'):
                self.assertNotIn(key, app.session_state)
            self.assertEqual(self.marks(app), ['done', 'done', 'current', 'locked', 'locked'])

    def test_sam2_model_resource_from_an_earlier_module_copy_is_rebuilt(self):
        # Streamlit reloads the app's modules when a source file changes; a cached resource created before that is rejected by
        # run_sam2 (a different class object), so the page must rebuild it instead of failing.
        import streamlit as st
        from Modules.Module_4.sam2_comparison import SAM2ModelResource
        st.cache_resource.clear()
        image = self.person()
        genuine = SAM2ModelResource(object(), 'sam2.1_hiera_tiny', 'config', Path('/unused'), 'cpu')
        app = self.open_page(image, [(10, 10), (90, 90)])
        self.run_classical(app)
        with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
             patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
             patch('Modules.Module_4.sam2_comparison.load_sam2_model', side_effect=[object(), genuine]) as load, \
             patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.fake_sam2(image)) as run:
            app.button(key='module4_run_sam2').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(load.call_count, 2)
        self.assertIs(run.call_args.args[2], genuine)
        self.assertIn('module4_sam2_results', app.session_state)
        self.assertNotIn('module4_sam2_error', app.session_state)
        st.cache_resource.clear()

    def test_comparison_shows_the_final_classical_mask(self):
        image = self.person()
        for cleanup, caption in ((False, 'Classical Method — GrabCut Foreground Mask'),
                                 (True, 'Classical Method — Cleaned Foreground Mask')):
            with self.subTest(cleanup=cleanup):
                app = self.open_page(image, [(10, 10), (90, 90)])
                if cleanup:
                    app.checkbox(key='module4_cleanup_enabled').check().run()
                self.run_classical(app)
                self.run_sam2(app, image)
                app.button(key='module4_compare').click().run()
                self.assertFalse(app.exception)
                self.assertIn(caption, [i.proto.imgs[0].caption for i in app.get('image')])
                result = app.session_state['module4_results']
                self.assertEqual(result['cleaned_mask'] is result['raw_mask'], not cleanup)

    def test_grabcut_masks_follow_the_cleanup_setting(self):
        image = self.person()
        expected = {False: ['GrabCut Foreground Mask'],
                    True: ['Raw GrabCut Foreground Mask', 'Foreground Mask After Morphological Cleanup']}
        for cleanup, captions in expected.items():
            with self.subTest(cleanup=cleanup):
                app = self.open_page(image, [(10, 10), (90, 90)])
                if cleanup:
                    app.checkbox(key='module4_cleanup_enabled').check().run()
                self.run_classical(app)
                shown = [i.proto.imgs[0].caption for i in app.get('image')]
                self.assertEqual([c for c in shown if 'GrabCut Foreground Mask' in c or 'After Morphological' in c], captions)
                text = ' '.join(w.value for w in app.markdown)
                self.assertEqual('cannot reliably recover' in text, cleanup)

    def test_cleanup_kernel_is_shown_only_with_cleanup(self):
        app = self.open_page(self.person(), [(10, 10), (90, 90)])
        self.assertNotIn('module4_morphology_kernel', [s.key for s in app.selectbox])
        self.assertTrue(any('Morphological cleanup helps refine the foreground mask' in c.value for c in app.caption))
        app.checkbox(key='module4_cleanup_enabled').check().run()
        self.assertEqual(app.selectbox(key='module4_morphology_kernel').value, 3)
        self.assertEqual(app.selectbox(key='module4_morphology_kernel').options, ['3', '5'])
        app.checkbox(key='module4_cleanup_enabled').uncheck().run()
        self.assertNotIn('module4_morphology_kernel', [s.key for s in app.selectbox])

    def test_invalid_canny_thresholds_disable_classical(self):
        app = self.open_page(self.person(), [(10, 10), (90, 90)])
        app.slider(key='module4_canny_lower').set_value(200).run()
        self.assertTrue(app.button(key='module4_apply_classical').disabled)
        self.assertTrue(any('lower threshold' in w.value for w in app.warning))

    def test_full_reset_restores_defaults_initial_state_and_preserves_q2(self):
        image = self.person()
        app = self.open_page(image, [(10, 10), (90, 90)])
        app.slider(key='module4_gaussian_sigma').set_value(1.7).run()
        app.checkbox(key='module4_cleanup_enabled').check().run()
        self.run_classical(app)
        self.run_sam2(app, image)
        q2 = {'module4_q2_results': {'mask': 'retained'},
              'module4_q2_sam2_results': {'mask': 'retained SAM2'},
              'module4_q2_points': [(1, 2), (3, 4)],
              'module4_q2_comparison': ('classical', 'sam2'),
              'module4_q2_thermal_bytes': b'unchanged'}
        for key, value in q2.items():
            app.session_state[key] = value
        app.session_state['module4_comparison'] = 'old'
        generation = app.session_state['module4_upload_generation']
        app.button(key='module4_reset_rgb').click().run()
        self.assertFalse(app.exception)
        for key in ('module4_rgb_bytes', 'module4_rgb_filename', 'module4_rgb_prepared', 'module4_results',
                    'module4_classical_identity', 'module4_sam2_results', 'module4_sam2_identity', 'module4_comparison'):
            self.assertNotIn(key, app.session_state)
        self.assertEqual(app.session_state['module4_points'], [])
        self.assertIsNone(app.session_state['module4_rgb_processing_identity'])
        self.assertEqual(app.session_state['module4_upload_generation'], generation + 1)
        self.assertEqual([h.value for h in app.subheader], ['Step 1 · Upload RGB Image', 'Step 2 · Select Person Rectangle', 'Step 3 · Classical Method', 'Step 4 · SAM2', 'Step 5 · Compare Results'])
        self.assertEqual(self.marks(app), ['current', 'locked', 'locked', 'locked', 'locked'])
        self.assertEqual(self.keys(app), ['module4_reset_rgb'])
        for key in ('module4_gaussian_sigma', 'module4_cleanup_enabled'):
            self.assertNotIn(key, app.session_state)
        for key, value in q2.items():
            self.assertEqual(app.session_state[key], value)
        # a new upload shows every classical setting at its original default
        app.session_state['module4_rgb_bytes'] = encoded(image)
        app.run()
        app.session_state['module4_points'] = [(10, 10), (90, 90)]
        app.run()
        self.assertEqual(app.slider(key='module4_gaussian_sigma').value, 0.8)
        self.assertEqual(app.slider(key='module4_canny_lower').value, 50)
        self.assertEqual(app.slider(key='module4_canny_upper').value, 150)
        self.assertEqual(app.slider(key='module4_grabcut_iterations').value, 5)
        self.assertEqual(app.selectbox(key='module4_gaussian_kernel').value, 3)
        self.assertFalse(app.checkbox(key='module4_cleanup_enabled').value)

    def test_page_reruns_and_identical_method_inputs(self):
        image = np.full((1600, 800, 3), 50, np.uint8)
        image[300:1200, 200:500] = (190, 100, 70)
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['module4_section'] = 'rgb'
        app.session_state['module4_rgb_bytes'] = encoded(image)
        with patch.object(prep, 'prepare_rgb_upload', wraps=prep.prepare_rgb_upload) as prepare:
            app.run()
            self.assertFalse(app.exception)
            app.run()
            self.assertEqual(prepare.call_count, 1)
            self.assertTrue(any('800 × 1600 to 500 × 1000' in c.value for c in app.caption))
            app.session_state['module4_points'] = [(50, 50), (450, 900)]
            app.run()
            with patch('Modules.Module_4.human_boundary.find_human_boundary', wraps=find_human_boundary) as classical:
                self.run_classical(app)
                self.assertEqual(classical.call_args.args[0].shape, (1000, 500, 3))
                self.assertEqual(classical.call_args.args[1], (50, 50, 400, 850))
                self.assertEqual(app.session_state['module4_results']['boundary_overlay'].shape, (1000, 500, 3))
            self.assertEqual(prepare.call_count, 1)
            prepared_image = classical.call_args.args[0]
            selected_rectangle = classical.call_args.args[1]
            with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
                 patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
                 patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
                 patch('Modules.Module_4.sam2_runtime.check_image_budget', wraps=check_image_budget) as budget, \
                 patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.fake_sam2(prepared_image)) as run:
                app.button(key='module4_run_sam2').click().run()
                self.assertFalse(app.exception)
                budget.assert_called_once()
                self.assertEqual(budget.call_args.args[0], prepared_image.shape)
                self.assertIs(run.call_args.args[0], prepared_image)
                self.assertEqual(run.call_args.args[1], selected_rectangle)
            app.button(key='module4_compare').click().run()
            self.assertFalse(app.exception)
            prepared_cache = app.session_state['module4_rgb_prepared']
            processing_identity = app.session_state['module4_rgb_processing_identity']
            for method in ('sam2', 'classical'):
                app.button(key=f'module4_reset_{method}').click().run()
                self.assertFalse(app.exception)
                self.assertIs(app.session_state['module4_rgb_prepared'], prepared_cache)
                self.assertEqual(app.session_state['module4_rgb_processing_identity'], processing_identity)
                self.assertEqual(prepare.call_count, 1)
            # Policy changes must clear selections and any retained results.
            self.run_classical(app)
            with patch.object(prep, 'MAX_RGB_PIXELS', 400000):
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['module4_points'], [])
                self.assertNotIn('module4_results', app.session_state)
                self.assertNotIn('module4_sam2_results', app.session_state)
            app.button(key='module4_reset_rgb').click().run()
            self.assertNotIn('module4_rgb_prepared', app.session_state)

    def test_visitor_page_has_no_development_or_report_text(self):
        app = self.open_page()
        text = ' '.join([*(m.value for m in app.markdown), *(c.value for c in app.caption), *(t.value for t in app.title)]).lower()
        for word in ('module 4 –', 'question 1', 'run your own', 'report', 'todo', 'claude', 'chatgpt'):
            self.assertNotIn(word, text)


class MaskAgreementTests(unittest.TestCase):
    """The shared Mask Agreement (IoU) section on both visitor pages."""

    @staticmethod
    def iou_function():
        tree = ast.parse(PAGE.read_text())
        wanted = ('thermal_mask_iou', 'overlap_distance')
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        namespace = {'np': np, 'math': math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PAGE), 'exec'), namespace)
        return namespace

    def test_iou_values(self):
        functions = self.iou_function()
        iou = functions['thermal_mask_iou']
        a = np.zeros((10, 10), np.uint8)
        a[0:5, 0:5] = 255
        same, disjoint, half, empty = a.copy(), np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
        disjoint[5:10, 5:10] = 255
        half[0:5, 0:5] = 255
        half[0:5, 0:5][:, 2:] = 0                       # 10 of the 25 pixels, all inside a: 10 / 25
        self.assertEqual(iou(a, same), 1.0)
        self.assertEqual(f"{iou(a, same):.4f}", '1.0000')
        self.assertEqual(f"{iou(a, disjoint):.4f}", '0.0000')
        self.assertAlmostEqual(iou(a, half), 10 / 25)
        self.assertEqual(f"{iou(a, half):.4f}", '0.4000')
        self.assertEqual(iou(a, empty), 0.0)            # one empty mask: 0.0000
        self.assertEqual(iou(empty, a), 0.0)
        self.assertIsNone(iou(empty, empty.copy()))      # both empty: N/A
        with self.assertRaises(ValueError):
            iou(a, np.zeros((4, 4), np.uint8))

    def test_partial_example_circles_have_an_iou_of_one_half(self):
        distance = self.iou_function()['overlap_distance'](0.5)
        radius, size = 300, 1400
        yy, xx = np.mgrid[0:size, 0:size]
        first = (xx - 500) ** 2 + (yy - 700) ** 2 <= radius ** 2
        second = (xx - 500 - distance * radius) ** 2 + (yy - 700) ** 2 <= radius ** 2
        self.assertAlmostEqual(np.count_nonzero(first & second) / np.count_nonzero(first | second), 0.5, delta=0.005)

    @staticmethod
    def rgb_app():
        image = np.full((100, 100, 3), 30, np.uint8)
        image[25:75, 30:70] = (190, 100, 70)
        app = AppTest.from_file(str(PAGE), default_timeout=60)
        app.session_state['module4_section'] = 'rgb'
        app.session_state['module4_rgb_bytes'] = encoded(image)
        app.run()
        app.session_state['module4_points'] = [(10, 10), (90, 90)]
        app.run()
        return app, image

    @staticmethod
    def sam2_result(image, mask):
        return {'binary_mask': mask, 'boundary_overlay': image.copy(), 'predicted_mask_quality': .5,
                'metadata': {'device': 'cpu', 'empty_mask': not bool(mask.any())}}

    def run_sam2(self, app, image, mask_for):
        """Run SAM2 (mocked) with a mask derived from the classical mask that is already in the session."""
        mask = mask_for(app.session_state['module4_results']['cleaned_mask'])
        with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
             patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
             patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
             patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.sam2_result(image, mask)):
            app.button(key='module4_run_sam2').click().run()
        self.assertFalse(app.exception)

    @staticmethod
    def svgs(app):
        return [e.proto.body for e in app.get('html') if e.proto.body.lstrip().startswith('<svg')]

    @staticmethod
    def expected_iou(first, second):
        a, b = first != 0, second != 0
        return np.count_nonzero(a & b) / np.count_nonzero(a | b)

    def test_rgb_section_uses_the_final_classical_mask_and_appears_only_after_comparison(self):
        app, image = self.rgb_app()
        app.checkbox(key='module4_cleanup_enabled').check().run()
        app.button(key='module4_apply_classical').click().run()
        def half(mask):
            out = mask.copy()
            out[:, : mask.shape[1] // 2] = 0
            return out
        self.run_sam2(app, image, half)
        self.assertEqual(self.svgs(app), [])            # nothing before Compare Results
        self.assertNotIn('##### Mask Agreement (IoU)', [m.value for m in app.markdown])
        app.button(key='module4_compare').click().run()
        results, sam2 = app.session_state['module4_results'], app.session_state['module4_sam2_results']
        expected = f"{self.expected_iou(results['cleaned_mask'], sam2['binary_mask']):.4f}"
        markdown = [m.value for m in app.markdown]
        self.assertIn('##### Mask Agreement (IoU)', markdown)
        self.assertIn(f'### {expected}', markdown)
        self.assertEqual(len(self.svgs(app)), 4)       # the diagram and the three examples
        # the section sits between the boundary comparison and Observations & Limitations
        order = [m for m in markdown if m.startswith('##### ')]
        self.assertEqual(order[-3:], ['##### Human Boundary Comparison', '##### Mask Agreement (IoU)', '##### Observations & Limitations'])
        self.assertTrue(any('Illustration of mask overlap.' in c.value for c in app.caption))
        self.assertEqual(len(app.latex), 1)
        # stale values are never shown: a changed setting removes the section, and a new comparison recalculates it
        app.slider(key='module4_canny_lower').set_value(60).run()
        self.assertNotIn('##### Mask Agreement (IoU)', [m.value for m in app.markdown])
        self.assertEqual(self.svgs(app), [])
        app.button(key='module4_apply_classical').click().run()
        app.button(key='module4_compare').click().run()
        self.assertIn('##### Mask Agreement (IoU)', [m.value for m in app.markdown])
        app.button(key='module4_reset_sam2').click().run()
        self.assertEqual(self.svgs(app), [])

    def test_rgb_empty_masks_give_zero_or_not_available(self):
        for name, mask_for, shown in (('identical', lambda m: m.copy(), '1.0000'), ('one empty', lambda m: np.zeros_like(m), '0.0000')):
            with self.subTest(name):
                app, image = self.rgb_app()
                app.button(key='module4_apply_classical').click().run()
                self.run_sam2(app, image, mask_for)
                app.button(key='module4_compare').click().run()
                self.assertIn(f'### {shown}', [m.value for m in app.markdown])
        app, image = self.rgb_app()
        app.button(key='module4_apply_classical').click().run()
        app.session_state['module4_results']['cleaned_mask'] = np.zeros_like(app.session_state['module4_results']['cleaned_mask'])
        self.run_sam2(app, image, lambda m: np.zeros_like(m))
        app.button(key='module4_compare').click().run()
        self.assertIn('### N/A', [m.value for m in app.markdown])
        self.assertTrue(any('undefined because both masks' in c.value for c in app.caption))

    def test_both_visitor_pages_show_the_same_section(self):
        rgb, image = self.rgb_app()
        rgb.button(key='module4_apply_classical').click().run()
        self.run_sam2(rgb, image, lambda m: m.copy())
        rgb.button(key='module4_compare').click().run()
        thermal_image = np.full((1440, 1080, 3), (0, 0, 255), np.uint8)
        thermal_image[200:1200, 300:800] = (255, 180, 0)
        thermal = AppTest.from_file(str(PAGE), default_timeout=60)
        thermal.session_state['module4_section'] = 'thermal'
        thermal.session_state['module4_q2_bytes'] = encoded(thermal_image)
        thermal.run()
        thermal.button(key='module4_q2_run').click().run()
        thermal.session_state['module4_q2_points'] = [(100, 80), (500, 750)]
        thermal.run()
        thermal_mask = thermal.session_state['module4_q2_results']['cleaned_human_mask']
        processed = thermal.session_state['module4_q2_prepared'][1]
        with patch('Modules.Module_4.sam2_comparison.select_device', return_value='cpu'), \
             patch('Modules.Module_4.sam2_runtime.resolve_checkpoint', return_value=Path('/unused')), \
             patch('Modules.Module_4.sam2_comparison.load_sam2_model', return_value=object()), \
             patch('Modules.Module_4.sam2_comparison.run_sam2', return_value=self.sam2_result(processed, thermal_mask.copy())):
            thermal.button(key='module4_q2_run_sam2').click().run()
        self.assertEqual(self.svgs(thermal), [])
        thermal.button(key='module4_q2_compare').click().run()
        self.assertFalse(thermal.exception)
        self.assertEqual(self.svgs(rgb), self.svgs(thermal))
        self.assertEqual(len(self.svgs(thermal)), 4)
        for app in (rgb, thermal):
            markdown = [m.value for m in app.markdown]
            self.assertIn('### 1.0000', markdown)
            self.assertIn('Intersection over Union (IoU) measures how closely the foreground masks from the classical method and SAM2 overlap.', markdown)
            self.assertEqual([m for m in markdown if m.startswith('##### Mask Agreement')], ['##### Mask Agreement (IoU)'])  # not duplicated
            self.assertEqual([t for t in app.table if 'Intersection over Union (IoU)' in list(t.value.index)], [])
        svg = ' '.join(self.svgs(rgb))
        for text in ('Classical Mask', 'SAM2 Mask', 'Intersection', 'Union — combined foreground area',
                     'No overlap', 'Partial overlap', 'Identical masks', '#478BD1', '#E7A33D', '#459B70'):
            self.assertIn(text, svg)
        # the diagrams use the current text color, so they follow the theme
        self.assertIn('currentColor', svg)


if __name__ == '__main__':
    unittest.main()
