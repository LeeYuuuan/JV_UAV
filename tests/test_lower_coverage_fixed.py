"""Standalone physical-abstraction, scheduling, reward and SAC regression checks."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/lower_coverage_fixed")]
from jv_uav.config import load_config
from coverage_env import CoverageEnv
from coverage_trainer import Trainer
from coverage_sac import compact


def configs(tiny=False):
    ec = load_config(ROOT / "configs/default.yaml")
    cfg = yaml.safe_load((ROOT / "configs/training_lower_coverage_fixed.yaml").read_text())
    if tiny:
        cfg["episode_steps"] = 24
        cfg["sac"].update(hidden=16, heads=2, batch_size=2, replay_capacity=128, random_steps=0, learning_starts=0)
    return ec, cfg


def test_group_schedule_keeps_positions_and_no_return():
    ec, cfg = configs()
    env = CoverageEnv(ec, cfg)
    original_long = env.long_ids.copy()
    for n in range(20):
        ids = env.observation()["ids"]
        assert len(ids) == (6 if n < 10 else len(original_long))
        before = env.scene.uav_pos.copy()
        # Keep targets far from the airship to expose accidental return/reset.
        action = np.full((len(ids), 2), .6, np.float32)
        nxt, _, done, row = env.step(action)
        assert not done
        inactive = np.setdiff1d(np.arange(6), ids)
        assert np.array_equal(env.scene.uav_pos[inactive], before[inactive])
        assert np.allclose(env.scene.uav_pos[ids], 3200)
        assert row["step_in_block"] == n + 1
    assert len(nxt["ids"]) == 6
    assert np.allclose(env.scene.uav_pos, 3200)
    assert env.phase == 0 and env.scene.now_sec == 1200


def test_all_backlog_counts_without_topk_gate_and_overlap_once():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    env = CoverageEnv(ec, cfg)
    env.scene.sensor_pos[:] = [0, 0]
    env.scene.sensor_pos[0] = [2000, 2000]
    env.scene.sensor_backlog[:] = 100
    env.scene.sensor_backlog[0] = 10
    _, reward, _, row = env.step(np.zeros((6, 2), np.float32))
    assert row["covered_ids"] == [0] and row["collected_packets"] == 10
    assert sum(row["collected_by_uav"]) == 10
    fraction = 10 / (149 * 100 + 10)
    assert np.isclose(row["cleared_fraction"], fraction)
    assert np.isclose(row["reward_terms"]["all_sensor_clear"], .5 * 100 / 300 * fraction)
    assert row["reward_terms"]["all_sensor_clear"] > 0
    assert np.isclose(reward, -100 / 300 + .5 * 100 / 300 * fraction)


def test_short_inactive_uav_does_not_collect_from_its_held_position():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    env = CoverageEnv(ec, cfg)
    env.scene.sensor_pos[:] = [2000, 2000]
    for _ in range(10):
        action = np.ones((len(env.observation()["ids"]), 2), np.float32)
        env.step(action)
    env.scene.uav_pos[env.short_ids] = [2000, 2000]
    env.scene.sensor_backlog[:] = 50
    _, _, _, row = env.step(np.ones((len(env.long_ids), 2), np.float32))
    assert row["collected_packets"] == 0
    assert np.all(env.scene.sensor_backlog == 50)


def test_fixed_time_arrivals_persist_at_group_boundary_reset_at_episode_only():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    cfg["episode_steps"] = 21
    env = CoverageEnv(ec, cfg)
    env.scene.sensor_pos[:] = [0, 0]
    env.scene.sensor_backlog[:] = 123
    for n in range(21):
        _, _, done, _ = env.step(np.zeros((len(env.observation()["ids"]), 2), np.float32))
        assert done == (n == 20)
        assert np.all(env.scene.sensor_backlog == 123)
    assert env.scene.now_sec == 21 * 60
    env.reset()
    assert np.all(env.scene.sensor_backlog == 0) and env.scene.now_sec == 0


def test_no_energy_constraints_and_full_map_targets():
    ec, cfg = configs()
    env = CoverageEnv(ec, cfg)
    env.scene.uav_battery[:] = 0
    env.scene.sensor_pos[:] = [4000, 4000]
    _, _, done, row = env.step(np.ones((6, 2), np.float32))
    assert not done and row["collected_packets"] > 0
    assert np.all(env.scene.uav_pos == 4000)
    assert "soc" not in env.observation()
    assert set(row["reward_terms"]) == {"backlog_post", "all_sensor_clear"}


def test_inputs_contain_map_backlog_identity_and_remaining_service():
    ec, cfg = configs(True)
    trainer = Trainer(ec, cfg)
    obs = trainer.env.observation()
    batch = trainer.sac.pack([obs])
    assert np.allclose(batch["sensors"][0, :, :2], obs["sensor_pos"] / 4000)
    assert np.allclose(batch["sensors"][0, :, 2], obs["backlog"] / 3000)
    assert np.allclose(batch["uavs"][0, :, 2], np.arange(6) / 5)
    assert np.allclose(batch["uavs"][0, :, 3], obs["service_remaining"])
    assert set(obs["service_remaining"]) == {.5, 1.0}


def test_poisson_arrivals_have_correct_scale():
    ec, cfg = configs()
    env = CoverageEnv(ec, cfg)
    env.scene.sensor_pos[:] = [0, 0]
    initial = env.scene.sensor_backlog.copy()
    for _ in range(20):
        env.step(np.zeros((len(env.observation()["ids"]), 2), np.float32))
    count = float((env.scene.sensor_backlog - initial).sum())
    expected = 150 * .5 * 60 * 20
    assert abs(count - expected) < 6 * np.sqrt(expected)


def test_block_boundaries_bootstrap_and_only_episode_is_terminal():
    ec, cfg = configs(True)
    trainer = Trainer(ec, cfg)
    before = {k: v.clone() for k, v in trainer.sac.actor.state_dict().items()}
    for _ in range(24):
        trainer.step()
    rows = trainer.replay.items
    assert len(rows[9][3]["ids"]) in (3, 4) and not rows[9][4]
    assert len(rows[19][3]["ids"]) == 6 and not rows[19][4]
    assert [r[4] for r in rows] == [False] * 23 + [True]
    assert trainer.updates == 23 and trainer.episodes == 1
    assert all(row[5] == 60 for row in rows)
    assert any(not torch.equal(value, trainer.sac.actor.state_dict()[key]) for key, value in before.items())
    assert all(np.isfinite(x) for x in trainer.last_losses.values())
    terminal = trainer.sac.update([rows[-1], rows[-1]])
    assert terminal["discount_mean"] == 0
    assert np.isclose(terminal["target_mean"], rows[-1][2], rtol=1e-5)


def test_nonterminal_target_uses_gamma099_per_fixed_step():
    ec, cfg = configs(True)
    trainer = Trainer(ec, cfg)
    obs = trainer.env.observation()
    original = trainer.sac.actor.forward
    def no_entropy(batch, deterministic=False):
        action, logp = original(batch, deterministic)
        return action, logp * 0
    trainer.sac.actor.forward = no_entropy
    def target(batch, actions):
        q = torch.full((len(actions),), 7.0)
        return q, q
    trainer.sac.target.forward = target
    row = (compact(obs), np.zeros((6, 2), np.float32), -2, compact(obs), False, 60)
    result = trainer.sac.update([row, row])
    assert np.isclose(result["discount_mean"], .99)
    assert np.isclose(result["target_mean"], -2 + .99 * 7)


def test_exact_resume_at_short_long_and_episode_boundaries():
    for at in (9, 19, 24):
        ec, cfg = configs(True)
        first = Trainer(ec, cfg)
        for _ in range(at):
            first.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            first.save(path)
            row_a = first.step()
            actor_a = {key: value.clone() for key, value in first.sac.actor.state_dict().items()}
            second = Trainer.load(path)
            row_b = second.step()
            assert json.dumps(row_a, sort_keys=True) == json.dumps(row_b, sort_keys=True)
            assert all(torch.equal(value, second.sac.actor.state_dict()[key]) for key, value in actor_a.items())


def test_evaluation_is_reproducible_and_preserves_training_rng():
    ec, cfg = configs(True)
    trainer = Trainer(ec, cfg)
    trainer.step()
    state = trainer.env.state_dict()
    torch_rng = torch.get_rng_state().clone()
    rng = copy.deepcopy(trainer.rng.bit_generator.state)
    assert trainer.evaluate(1001) == trainer.evaluate(1001)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert rng == trainer.rng.bit_generator.state
    assert np.array_equal(state["scene"]["sensor_backlog"], trainer.env.scene.sensor_backlog)


def test_greedy_reference_improves_objective_and_reaches_all_sensors():
    ec, cfg = configs(True)
    cfg["episode_steps"] = 200
    trainer = Trainer(ec, cfg)
    home = trainer.evaluate(1001, mode="center")
    greedy = trainer.evaluate(1001, mode="greedy_reference")
    assert home["simulation_sec"] == greedy["simulation_sec"] == 12000
    assert greedy["sensor_coverage_fraction"] == 1
    assert greedy["outer_visited"] == 50
    assert greedy["lower_return"] > home["lower_return"]
    assert greedy["lower_discounted_return"] > home["lower_discounted_return"]
    assert greedy["mean_max_backlog"] < home["mean_max_backlog"]


def test_eval_always_saves_trajectory_light_and_full():
    ec, cfg = configs(True)
    cfg["episode_steps"] = 4
    trainer = Trainer(ec, cfg)
    with tempfile.TemporaryDirectory() as folder:
        light, full = Path(folder) / "light", Path(folder) / "full"
        a = trainer.evaluate(1001, light, detailed=False)
        b = trainer.evaluate(1001, full, detailed=True)
        assert a == b
        assert (light / "trajectory.png").read_bytes().startswith(b"\x89PNG")
        assert not (light / "trace.json").exists()
        assert all((full / name).exists() for name in ("trajectory.png", "dashboard.png", "trace.json", "summary.json"))


def test_invalid_actions_and_zero_backlog_are_handled():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    env = CoverageEnv(ec, cfg)
    for action in [np.zeros((1, 2)), np.full((6, 2), np.nan), np.full((6, 2), 2)]:
        try:
            env.step(action)
            raise AssertionError("invalid action accepted")
        except ValueError:
            pass
    _, reward, _, row = env.step(np.zeros((6, 2), np.float32))
    assert reward == 0 and row["cleared_fraction"] == 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    names = []
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function(); names.append(name); print("PASS", name, flush=True)
    print(json.dumps(dict(passed=len(names), tests=names)), flush=True)
