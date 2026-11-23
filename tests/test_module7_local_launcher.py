"""Two-button room page (Cloud vs local mode), local_mode.py and the one-command launcher run_local.py."""
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import local_mode  # noqa: E402
import run_local  # noqa: E402
from test_module7_calibration_wizard import AppTest, PAGE  # noqa: E402

RUN = "m7_open_room_run"


def room_page(view="3_visitor", env=None, running=False):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CSC8830_")}
    clean.update(env or {})
    with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(local_mode, "visitor_app_running", return_value=running):
        at = AppTest.from_file(str(PAGE), default_timeout=120)
        at.session_state["module7_part"] = view
        return at.run()


def run_button(at):
    return next((b for b in at.button if b.key == RUN), None)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class TwoButtonPageTests(unittest.TestCase):
    def test_cloud_run_button_is_disabled_with_note_and_no_link(self):
        at = room_page(running=True)               # even a listening port is ignored outside local mode
        self.assertFalse(at.exception)
        self.assertEqual([b.key for b in at.button if (b.key or "").startswith("m7_open_room_")], ["m7_open_room_setup", RUN])
        self.assertTrue(run_button(at).disabled)
        self.assertEqual(len(at.get("link_button")), 0)
        self.assertTrue(any("Available when running locally. See Setup Instructions." in m.value for m in at.markdown))
        self.assertEqual(len(at.get("file_uploader")), 0)

    def test_setup_button_is_enabled_on_cloud_and_local_and_opens_the_guide(self):
        for env in ({}, {local_mode.LOCAL_MODE_ENV: "1"}):
            at = room_page(env=env)
            setup = next(b for b in at.button if b.key == "m7_open_room_setup")
            self.assertFalse(setup.disabled)
            setup.click().run()
            self.assertEqual(at.session_state["module7_part"], "3_setup")
            self.assertEqual([t.value for t in at.title], ["Setup Instructions"])
            self.assertTrue(any("python run_local.py" in c.value for c in at.get("code")))

    def test_only_exactly_1_enables_local_mode(self):
        for value in ("", "0", "true", "yes", "11"):
            at = room_page(env={local_mode.LOCAL_MODE_ENV: value}, running=True)
            self.assertTrue(run_button(at).disabled, value)

    def test_local_mode_with_running_visitor_app_offers_a_loopback_link(self):
        at = room_page(env={local_mode.LOCAL_MODE_ENV: "1"}, running=True)
        self.assertFalse(at.exception)
        links = at.get("link_button")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].proto.url, "http://127.0.0.1:8640")
        self.assertIsNone(run_button(at))

    def test_local_mode_without_visitor_app_is_disabled_and_explains(self):
        at = room_page(env={local_mode.LOCAL_MODE_ENV: "1"}, running=False)
        self.assertTrue(run_button(at).disabled)
        self.assertEqual(len(at.get("link_button")), 0)
        self.assertTrue(any("not running" in m.value and "python run_local.py" in m.value for m in at.markdown))

    def test_visitor_url_is_always_loopback_whatever_the_environment_says(self):
        with mock.patch.dict(os.environ, {local_mode.VISITOR_PORT_ENV: "evil.example.com"}):
            self.assertEqual(local_mode.visitor_url(), "http://127.0.0.1:8640")
        with mock.patch.dict(os.environ, {local_mode.VISITOR_PORT_ENV: "8650"}):
            self.assertEqual(local_mode.visitor_url(), "http://127.0.0.1:8650")


class LauncherTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        a = run_local.parse_args([])
        self.assertEqual((a.website_port, a.visitor_port), (8501, 8640))
        for bad in (["--website-port", "80"], ["--visitor-port", "70000"], ["--website-port", "9000", "--visitor-port", "9000"]):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                run_local.parse_args(bad)

    def test_commands_bind_loopback_and_visitor_launcher_gets_only_a_port(self):
        web = run_local.website_command(8501)
        self.assertEqual(web[web.index("--server.address") + 1], "127.0.0.1")
        self.assertTrue(web[web.index("run") + 1].endswith("home.py"))
        vis = run_local.visitor_command(8640)
        self.assertTrue(vis[1].endswith("run_f8_integrated_app.py"))
        self.assertEqual(vis[2:], ["8640"])

    def test_child_env_sets_local_mode_and_ignores_inherited_values(self):
        with mock.patch.dict(os.environ, {"CSC8830_VISITOR_PORT": "1", "CSC8830_LOCAL_MODE": "0", "CSC8830_OTHER": "x"}):
            env = run_local.child_env(8501, 8641)
        self.assertEqual(env["CSC8830_LOCAL_MODE"], "1")
        self.assertEqual(env["CSC8830_VISITOR_PORT"], "8641")
        self.assertNotIn("CSC8830_OTHER", env)

    def test_port_conflict_is_reported_and_nothing_is_started(self):
        with socket.socket() as probe:                      # a port that is certainly free for the visitor application (independent of anything running on 8640)
            probe.bind(("127.0.0.1", 0)); free = probe.getsockname()[1]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); s.listen(64)          # a deep backlog: nobody accepts, and every probe connection must find room in the queue
            busy = s.getsockname()[1]
            self.assertNotEqual(busy, free)
            problems = run_local.check_ports(busy, free)
            self.assertEqual(len(problems), 1)
            self.assertIn(str(busy), problems[0])
            with mock.patch.object(run_local, "start") as start, mock.patch("sys.stderr"):
                self.assertEqual(run_local.main(["--website-port", str(busy), "--visitor-port", str(free)]), 1)
            start.assert_not_called()

    def test_startup_failure_stops_everything_and_returns_1(self):
        failing = mock.Mock(); failing.poll.return_value = 3; failing.pid = 1
        with mock.patch.object(run_local, "check_ports", return_value=[]), mock.patch.object(run_local, "start", return_value=failing) as start, \
                mock.patch.object(run_local, "stop") as stop, mock.patch("sys.stderr"), mock.patch("builtins.print"):
            self.assertEqual(run_local.main([]), 1)
        self.assertEqual(start.call_count, 1)          # the website is never started when the visitor application fails
        stop.assert_called()


if __name__ == "__main__":
    unittest.main()
