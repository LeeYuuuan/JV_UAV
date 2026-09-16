"""Rendering must remain observational and correctly flag terminal records."""
import copy
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_core import build_env


def test_render_is_read_only_and_closes_figures():
    _, scene, _, _, env = build_env()
    env.reset()
    env.step(np.zeros(scene.num_uavs, dtype=np.int8))
    arrays = {name: getattr(scene, name).copy() for name in (
        "uav_pos", "uav_battery", "uav_status", "sensor_backlog", "sensor_last_visit_sec"
    )}
    trace = copy.deepcopy(env.trace.to_serializable())
    rng = copy.deepcopy(scene.rng.bit_generator.state)
    clock = (scene.now_sec, scene.upper_step, scene.low_step_total, scene.step_in_frame)
    for view in ("dashboard", "heatmaps"):
        fig = env.render(view=view, show=False, dpi=65)
        output = BytesIO()
        fig.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
        assert not plt.fignum_exists(fig.number)
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(scene, name), value)
    assert env.trace.to_serializable() == trace
    assert scene.rng.bit_generator.state == rng
    assert clock == (scene.now_sec, scene.upper_step, scene.low_step_total, scene.step_in_frame)


def test_initial_and_unsettled_terminal_views():
    _, scene, _, _, env = build_env()
    env.reset()
    env.render(show=False, sensor_value="last_visit", trail_frames=0, dpi=65)
    scene.uav_battery[0] = 0.01
    scene.uav_battery[1] = 0.4
    _, _, terminated, _, _ = env.step(np.array([0, 1, 0, 0, 0, 0]))
    assert terminated
    frame = env.trace.frames[-1]
    assert not frame.settled
    assert frame.termination_reason == "battery_depleted_during_low_step"
    np.testing.assert_allclose(scene.uav_battery[1], 0.4)
    fig = env.render(show=False, dpi=65)
    assert any("UNSETTLED TERMINAL FRAME" in text.get_text() for text in fig.texts)
    saved = env.trace.to_serializable()["frames"][-1]
    assert saved["settled"] is False
    assert saved["termination_reason"] == frame.termination_reason


def test_final_return_failure_is_settled_and_reported():
    _, scene, _, _, env = build_env()
    env.reset()
    scene.uav_battery.fill(0.44)
    _, _, terminated, _, _ = env.step(np.zeros(scene.num_uavs, dtype=np.int8))
    assert terminated
    assert env.trace.frames[-1].settled
    assert env.trace.frames[-1].termination_reason == "cannot_return_to_airship"
    fig = env.render(show=False, dpi=65)
    assert any("cannot return to airship" in text.get_text() for text in fig.texts)


def test_history_coverage_includes_only_serving_step_endpoints():
    _, scene, _, _, env = build_env()
    env.reset()
    env.step(np.array([1, 1, 0, 0, 0, 0]))
    env.step(np.array([0, 0, 1, 1, 0, 0]))
    for trail, count in [(1, 40), (2, 80)]:
        fig = env.render(show=False, trail_frames=trail, dpi=65)
        history = [c for c in fig.axes[0].collections if c.get_gid() == 'historical_coverage']
        assert len(history) == 1 and len(history[0].get_paths()) == count
        assert f'{trail * 10} low steps' in fig.axes[0].get_title(loc='left')
    fig = env.render(show=False, draw_coverage=False, dpi=65)
    assert not any(c.get_gid() == 'historical_coverage' for c in fig.axes[0].collections)
