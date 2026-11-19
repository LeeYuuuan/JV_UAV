"""Physical/reward/bootstrapping/resume checks for the independent waypoint task."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/lower_waypoint")]
from jv_uav.config import load_config
from waypoint_env import WaypointEnv
from waypoint_trainer import Trainer
from waypoint_sac import compact


def configs():
    env = load_config(ROOT / "configs/default.yaml")
    cfg = yaml.safe_load((ROOT / "configs/training_lower_waypoint.yaml").read_text())
    return env, cfg


def tiny():
    env, cfg = configs()
    cfg.update(episode_sec=3200.0)
    cfg["sac"].update(hidden=16, heads=2, batch_size=2, replay_capacity=64, learning_starts=0, random_steps=0)
    return env, cfg


def test_sync_wait_real_energy_and_return_clock():
    ec, cfg = configs()
    env = WaypointEnv(ec, cfg)
    env.ids = np.array([0, 1])
    action = np.array([[0, 0], [1, 1]], np.float32)
    _, _, done, row = env.step(action)
    move = np.sqrt(2000**2 + 2000**2) / 6
    duration = move + 30
    a, b = row["return_events"]
    assert not done
    assert np.isclose(row["service_sec"], duration)
    assert np.isclose(a["hover_sec"], duration)
    assert np.isclose(b["hover_sec"], 30, atol=1e-4)
    assert np.isclose(a["service_soc"], 326 * duration / 450000)
    assert np.isclose(b["service_soc"], (257 * move + 326 * 30) / 450000)
    assert np.isclose(row["return_sec"], move + 10)
    assert np.isclose(env.scene.now_sec, duration + move + 10)
    assert row["failures"] == 0
    assert all(x["return_paid_soc"] > 0 for x in row["return_events"])


def test_union_topk_and_integral_reward_exact_constant_backlog():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    cfg["reward"]["top_k"] = 2
    env = WaypointEnv(ec, cfg)
    env.ids = np.array([0, 1])
    env.scene.sensor_pos[:] = [0, 0]
    env.scene.sensor_pos[0] = [2000, 2000]
    env.scene.sensor_backlog[:] = 0
    env.scene.sensor_backlog[:2] = [100, 80]
    _, reward, _, row = env.step(np.zeros((2, 2), np.float32))
    assert row["covered_ids"] == [0]
    assert np.isclose(row["h_k"], 100 / 180)
    assert row["collected_packets"] == 100
    assert row["collected_by_uav"][0] == 100 and sum(row["collected_by_uav"][1:]) == 0
    # Service: max100 for30s. After clear, max80 for the10s climb/return.
    assert row["backlog_integral_packets_sec"] == 100 * 30 + 80 * 10
    expected = -3800 / (300 * 60) + .5 * 80 / 300 * (100 / 180)
    assert np.isclose(reward, expected)
    assert np.isclose(reward, sum(row["reward_terms"].values()))


def test_failure_is_charged_once_and_does_not_end_episode():
    ec, cfg = configs()
    env = WaypointEnv(ec, cfg)
    env.ids = np.array([0, 1])
    env.scene.uav_battery[env.ids] = .4
    _, reward, done, row = env.step(np.ones((2, 2), np.float32))
    assert not done and row["failures"] == 2
    assert all(x["service_survived"] and not x["success"] for x in row["return_events"])
    assert all(x["return_paid_soc"] == 0 for x in row["return_events"])
    assert row["reward_terms"]["failure"] == -1000
    assert np.isclose(row["reward_terms"]["failure_distance"], -500 * 2 * np.sqrt(2000**2 + 2000**2) / 1800)
    assert reward < -1000
    assert np.all(env.scene.uav_battery == 1)


def test_flight_depletion_does_not_teleport_or_clear():
    ec, cfg = configs()
    env = WaypointEnv(ec, cfg)
    env.ids = np.array([0])
    env.scene.uav_battery[0] = .01
    env.scene.sensor_pos[:] = [4000, 4000]
    _, _, done, row = env.step(np.ones((1, 2), np.float32))
    event = row["return_events"][0]
    assert not done and not event["service_survived"]
    assert np.linalg.norm(np.array(event["position"]) - env.scene.airship_pos) < 200
    assert not row["covered_ids"]


def test_queues_persist_sorties_and_reset_only_episode():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    cfg["episode_sec"] = 80.0
    env = WaypointEnv(ec, cfg)
    env.scene.sensor_pos[:] = [0, 0]
    env.scene.sensor_backlog[:] = 123
    _, _, done, row = env.step(np.zeros((len(env.ids), 2), np.float32))
    assert not done and np.all(env.scene.sensor_backlog == 123)
    assert np.all(env.observation()["soc"] == 1)
    assert env.observation()["remaining"] == .5
    _, _, done, _ = env.step(np.zeros((len(env.ids), 2), np.float32))
    assert done and np.all(env.scene.sensor_backlog == 123)
    try:
        env.step(np.zeros((len(env.ids), 2), np.float32))
        raise AssertionError("step after done allowed")
    except RuntimeError:
        pass
    env.reset()
    assert np.all(env.scene.sensor_backlog == 0)


def test_random_ids_reproducible_and_all_counts_present():
    ec, cfg = configs()
    a, b = WaypointEnv(ec, cfg), WaypointEnv(ec, cfg)
    counts, seen = set(), set()
    for _ in range(60):
        assert np.array_equal(a.ids, b.ids)
        counts.add(len(a.ids)); seen.update(a.ids)
        if a.done:
            a.reset(); b.reset()
        x = np.zeros((len(a.ids), 2), np.float32)
        a.step(x); b.step(x)
    assert counts == {2, 3, 4} and seen == set(range(6))


def test_all_default_map_targets_are_physically_feasible():
    ec, cfg = configs()
    for target in [[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]]:
        env = WaypointEnv(ec, cfg)
        env.ids = np.array([0, 1])
        _, _, _, row = env.step(np.array([[0, 0], target], np.float32))
        assert row["failures"] == 0
        assert max(e["service_soc"] + e["return_paid_soc"] for e in row["return_events"]) < .59


def test_poisson_arrivals_clock_and_integral_bounds():
    ec, cfg = configs()
    env = WaypointEnv(ec, cfg)
    initial = float(env.scene.sensor_backlog.max())
    before = env.scene.sensor_backlog.copy()
    area, peak = env._advance(123.4)
    assert initial * 123.4 <= area <= peak * 123.4
    arrivals = float((env.scene.sensor_backlog - before).sum())
    mean = env.scene.num_sensors * .5 * 123.4
    assert abs(arrivals - mean) < 6 * np.sqrt(mean)
    assert np.isclose(env.scene.now_sec, 123.4)


def test_input_coordinates_backlog_and_uav_identity():
    ec, cfg = tiny()
    trainer = Trainer(ec, cfg)
    obs = trainer.env.observation()
    batch = trainer.sac.pack([obs])
    n = len(obs["ids"])
    assert np.allclose(batch["sensors"][0, :, :2].cpu(), obs["sensor_pos"] / 4000)
    assert np.allclose(batch["sensors"][0, :, 2].cpu(), obs["backlog"] / 3000)
    assert np.allclose(batch["uavs"][0, :n, 3].cpu(), obs["ids"] / 5)
    assert batch["mask"].sum() == n


def test_replay_done_discount_and_actor_gradient():
    ec, cfg = tiny()
    trainer = Trainer(ec, cfg)
    actor_before = {k: v.clone() for k, v in trainer.sac.actor.state_dict().items()}
    while not trainer.env.done:
        trainer.step()
    n = trainer.steps
    assert trainer.episodes == 1 and trainer.updates == n - 1
    rows = trainer.replay.items
    assert [r[4] for r in rows] == [False] * (n - 1) + [True]
    # The actual per-row discount uses physical seconds, not decision count.
    assert all(0 < .99 ** (r[5] / 60) < 1 for r in rows)
    assert any(not torch.equal(v, trainer.sac.actor.state_dict()[k]) for k, v in actor_before.items())
    assert all(np.isfinite(v) for v in trainer.last_losses.values())
    assert np.isclose(trainer.last_losses["actor_loss"], trainer.last_losses["policy_loss"] + trainer.last_losses["entropy_loss"])
    # A terminal target equals reward exactly, regardless of the future Q.
    terminal = rows[-1]
    result = trainer.sac.update([terminal, terminal])
    assert result["discount_mean"] == 0
    assert np.isclose(result["target_mean"], terminal[2], rtol=1e-5)


def test_exact_resume_mid_episode_and_at_reset_boundary():
    for at in (3, None):
        ec, cfg = tiny()
        first = Trainer(ec, cfg)
        if at is None:
            while not first.env.done:
                first.step()
        else:
            for _ in range(at):
                first.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            first.save(path)
            row_a = first.step()
            state_a = {k: v.clone() for k, v in first.sac.actor.state_dict().items()}
            second = Trainer.load(path)
            row_b = second.step()
            assert json.dumps(row_a, sort_keys=True) == json.dumps(row_b, sort_keys=True)
            assert all(torch.equal(v, second.sac.actor.state_dict()[k]) for k, v in state_a.items())


def test_smdp_nonterminal_target_matches_duration_formula():
    ec, cfg = tiny()
    trainer = Trainer(ec, cfg)
    obs = trainer.env.observation()
    action = np.zeros((len(obs["ids"]), 2), np.float32)
    row_a = (compact(obs), action, -2.0, compact(obs), False, 60.0)
    row_b = (compact(obs), action, -4.0, compact(obs), False, 120.0)
    trainer.sac.gamma = .5
    original = trainer.sac.actor.forward
    def actor_without_entropy(batch, deterministic=False):
        actions, logp = original(batch, deterministic)
        return actions, logp * 0
    trainer.sac.actor.forward = actor_without_entropy
    def constant_target(batch, actions):
        q = torch.full((len(actions),), 7.0)
        return q, q
    trainer.sac.target.forward = constant_target
    result = trainer.sac.update([row_a, row_b])
    assert np.isclose(result["discount_mean"], (.5 + .25) / 2)
    assert np.isclose(result["target_mean"], ((-2 + .5 * 7) + (-4 + .25 * 7)) / 2)


def test_eval_does_not_change_training_state_or_rng():
    ec, cfg = tiny()
    trainer = Trainer(ec, cfg)
    trainer.step()
    state = copy.deepcopy(trainer.env.state_dict())
    torch_rng = torch.get_rng_state().clone()
    sampling_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    a = trainer.evaluate(1001)
    b = trainer.evaluate(1001)
    assert a == b
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert sampling_rng == trainer.rng.bit_generator.state
    assert state["frames"] == trainer.env.frames
    assert np.array_equal(state["scene"]["sensor_backlog"], trainer.env.scene.sensor_backlog)


def test_invalid_actions_and_zero_topk_denominator():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    env = WaypointEnv(ec, cfg)
    for action in [np.zeros((1, 2)), np.full((len(env.ids), 2), np.nan), np.full((len(env.ids), 2), 2)]:
        try:
            env.step(action)
            raise AssertionError("invalid action accepted")
        except ValueError:
            pass
    _, reward, _, row = env.step(np.zeros((len(env.ids), 2), np.float32))
    assert row["h_k"] == 0 and reward == 0


def test_exact_time_horizon_interrupts_service_without_future_clear():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    cfg["episode_sec"] = 100.0
    env = WaypointEnv(ec, cfg)
    env.ids = np.array([0])
    env.scene.sensor_pos[:] = [4000, 4000]
    env.scene.sensor_backlog[:] = 100
    _, _, done, row = env.step(np.ones((1, 2), np.float32))
    assert done and env.scene.now_sec == 100
    assert not row["service_completed"] and row["covered_ids"] == []
    assert row["duration_sec"] == 100 and row["return_sec"] == 0
    assert np.isclose(np.linalg.norm(np.array(row["return_events"][0]["position"]) - env.scene.airship_pos), 600, atol=1e-3)
    assert np.all(env.scene.sensor_backlog == 100)


def test_exact_horizon_partial_return_only_charges_executed_energy():
    ec, cfg = configs()
    cfg["episode_sec"] = 35.0
    env = WaypointEnv(ec, cfg)
    _, _, done, row = env.step(np.zeros((len(env.ids), 2), np.float32))
    assert done and row["service_completed"]
    assert row["service_sec"] == 30 and row["return_sec"] == 5
    assert env.scene.now_sec == 35
    assert row["failures"] == 0
    for event in row["return_events"]:
        assert event["success"] and not event["return_completed"]
        assert np.isclose(event["return_paid_soc"], 400 * 5 / 450000)
        assert np.isclose(event["soc_end"], 1 - (326 * 30 + 400 * 5) / 450000)


def test_integral_linear_arrivals_matches_analytic_solution():
    ec, cfg = configs()
    env = WaypointEnv(ec, cfg)
    env.scene.sensor_backlog[:] = 100
    env.scene.arrival_rate[:] = 2
    class FluidArrivals:
        def poisson(self, expected):
            return expected
    env.scene.rng = FluidArrivals()
    integral, peak = env._advance(123.4)
    assert np.isclose(integral, 100 * 123.4 + 123.4**2)
    assert np.isclose(peak, 100 + 2 * 123.4)


def test_horizon_during_horizontal_return_tracks_actual_position():
    ec, cfg = configs()
    move = np.sqrt(2000**2 + 2000**2) / 6
    cfg["episode_sec"] = move + 30 + 100
    env = WaypointEnv(ec, cfg)
    env.ids = np.array([0])
    _, _, done, row = env.step(np.ones((1, 2), np.float32))
    event = row["return_events"][0]
    assert done and np.isclose(row["return_sec"], 100)
    assert event["success"] and not event["return_completed"]
    assert np.isclose(event["return_paid_soc"], (400 * 10 + 257 * 90) / 450000)
    assert np.isclose(np.linalg.norm(env.scene.uav_pos[0] - env.scene.airship_pos), np.sqrt(2000**2 + 2000**2) - 6 * 90, atol=.001)


def test_fixed_time_objective_prefers_collection_over_staying_home():
    ec, cfg = tiny()
    cfg["episode_sec"] = 28800.0
    trainer = Trainer(ec, cfg)
    home = trainer.evaluate(1001, mode="center")
    collect = trainer.evaluate(1001, mode="greedy_reference")
    assert home["simulation_sec"] == collect["simulation_sec"] == 28800
    assert collect["sensor_coverage_fraction"] > .95
    assert collect["lower_return"] > home["lower_return"]
    assert collect["lower_discounted_return"] > home["lower_discounted_return"]
    assert collect["time_mean_max_backlog"] < home["time_mean_max_backlog"]
    assert collect["failures"] == home["failures"] == 0


def test_every_eval_saves_trajectory_even_without_detailed_trace():
    ec, cfg = tiny()
    cfg["episode_sec"] = 1200.0
    trainer = Trainer(ec, cfg)
    rng_before = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory() as directory:
        light, full = Path(directory) / "light", Path(directory) / "full"
        a = trainer.evaluate(1001, light, detailed=False)
        b = trainer.evaluate(1001, full, detailed=True)
        assert a == b
        assert (light / "trajectory.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (light / "summary.json").exists()
        assert not (light / "trace.json").exists() and not (light / "dashboard.png").exists()
        assert all((full / name).exists() for name in ("trajectory.png", "dashboard.png", "trace.json", "summary.json"))
    assert torch.equal(rng_before, torch.get_rng_state())


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = []
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function(); print(f"PASS {name}", flush=True); results.append(name)
    print(json.dumps(dict(passed=len(results), tests=results)), flush=True)
