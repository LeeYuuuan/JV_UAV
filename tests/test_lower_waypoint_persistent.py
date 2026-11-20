"""Real motion, committed transitions, epsilon mixture, and SAC invariants."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/lower_waypoint_persistent")]
from jv_uav.config import load_config
from persistent_env import PersistentEnv
from persistent_trainer import Trainer
from waypoint_sampler import WaypointSampler


def configs(tiny=False):
    ec = load_config(ROOT / "configs/default.yaml")
    cfg = yaml.safe_load((ROOT / "configs/training_lower_waypoint_persistent.yaml").read_text())
    if tiny:
        cfg.update(episode_steps=8, block_steps=4, short_steps=2)
        cfg["sac"].update(hidden=16, heads=2, batch_size=2, replay_capacity=128)
    return ec, cfg


def test_real_speed_and_target_persistence_until_group_change():
    ec, cfg = configs()
    env = PersistentEnv(ec, cfg)
    env.scene.sensor_pos[:] = [4000, 4000]
    _, _, done, first = env.step(np.ones((6, 2), np.float32))
    assert not done and first["physical_steps"] == 10 and first["stop_reason"] == "group_change"
    assert first["reached_ids"] == [] and first["collected_packets"] == 0
    assert np.allclose(np.linalg.norm(env.scene.uav_pos - env.scene.airship_pos, axis=1), 1800, atol=.01)
    assert len(env.observation()["ids"]) in (3, 4)
    short_position = env.scene.uav_pos[env.short_ids].copy()
    _, _, done, second = env.step(np.ones((len(env.long_ids), 2), np.float32))
    assert not done and second["physical_steps"] == 6 and second["stop_reason"] == "all_arrived"
    assert second["collected_packets"] > 0
    assert np.allclose(env.scene.uav_pos[env.long_ids], 4000)
    assert np.array_equal(env.scene.uav_pos[env.short_ids], short_position)
    for row in env.records:
        assert max(row["move_distance_by_uav_m"]) <= 180.001
        assert max(row["move_time_sec"]) <= 30.001 and min(row["service_time_sec"]) >= 29.999


def test_early_arrivals_hold_and_continue_serving():
    ec, cfg = configs()
    env = PersistentEnv(ec, cfg)
    action = np.zeros((6, 2), np.float32); action[1:, 0] = .3
    _, _, _, result = env.step(action)
    assert result["physical_steps"] == 4 and result["stop_reason"] == "all_arrived"
    assert all(np.array_equal(r["positions_end"][0], [2000, 2000]) for r in env.records)
    assert all(r["move_time_sec"][0] == 0 and r["service_time_sec"][0] == 60 for r in env.records)
    assert env.scene.now_sec == 240


def test_discounted_macro_reward_matches_real_physical_steps():
    ec, cfg = configs()
    ec["sensors"]["arrival_rate_per_sec"] = 0
    cfg["reward"]["movement_weight"] = 0
    env = PersistentEnv(ec, cfg)
    env.scene.sensor_pos[:] = [0, 4000]
    env.scene.sensor_backlog[:] = 100
    _, reward, _, row = env.step(np.ones((6, 2), np.float32))
    expected = -100 / 300 * sum(.95 ** k for k in range(10))
    assert np.isclose(reward, expected)
    assert np.isclose(row["raw_reward_sum"], -100 / 300 * 10)
    assert np.isclose(reward, sum(row["reward_terms"].values()))
    assert row["duration_sec"] == 600


def test_episode_end_interrupts_without_future_collection_or_teleport():
    ec, cfg = configs()
    cfg["episode_steps"] = 3
    env = PersistentEnv(ec, cfg)
    env.scene.sensor_pos[:] = [4000, 4000]
    _, _, done, result = env.step(np.ones((6, 2), np.float32))
    assert done and result["physical_steps"] == 3 and result["stop_reason"] == "episode_end"
    assert result["collected_packets"] == 0 and len(result["interrupted_ids"]) == 6
    assert np.allclose(np.linalg.norm(env.scene.uav_pos - env.scene.airship_pos, axis=1), 540, atol=.01)
    assert env.scene.now_sec == 180


def test_epsilon_schedule_and_actor_branch_have_no_extra_noise():
    ec, cfg = configs()
    sampler = WaypointSampler(cfg["exploration"], ec, 123)
    assert sampler.epsilon(0) == 1
    assert np.isclose(sampler.epsilon(100000), .1)
    assert np.isclose(sampler.epsilon(200000), .1)
    pure = copy.deepcopy(cfg["exploration"]); pure.update(epsilon_start=0, epsilon_end=0)
    sampler = WaypointSampler(pure, ec, 123)
    env = PersistentEnv(ec, cfg); obs = env.observation()
    actor = np.linspace(-.9, .9, 12).reshape(6, 2).astype(np.float32)
    action, meta = sampler.propose(obs, actor, 0)
    assert np.allclose(action, actor, atol=2e-7) and meta["sources"] == ["actor"] * 6


def test_exploration_branch_frequencies_and_valid_world_targets():
    ec, cfg = configs()
    sampler = WaypointSampler(cfg["exploration"], ec, 543)
    obs = PersistentEnv(ec, cfg).observation()
    for _ in range(1500):
        action, info = sampler.propose(obs, np.zeros((6, 2), np.float32), 0)
        assert np.all(np.abs(action) <= 1)
    total = sum(sampler.counts.values())
    assert abs(sampler.counts["random"] / total - .25) < .025
    assert abs(sampler.counts["greedy"] / total - .5) < .025
    assert abs(sampler.counts["actor"] / total - .25) < .025


def test_greedy_top6_nearest_jitter_and_overlap_reservation():
    ec, cfg = configs()
    ex = copy.deepcopy(cfg["exploration"])
    ex.update(epsilon_start=1, epsilon_end=1, random_probability=0, greedy_probability=1, actor_probability=0)
    sampler = WaypointSampler(ex, ec, 43)
    positions = np.column_stack([np.arange(8) * 500, np.full(8, 1000)]).astype(np.float32)
    obs = dict(ids=np.arange(6), positions=np.tile([3500, 1000], (6, 1)),
               sensor_pos=positions, backlog=np.arange(80, 0, -10).astype(np.float32))
    action, info = sampler.propose(obs, np.zeros((6, 2), np.float32), 0)
    assert info["greedy_sensor_ids"][0] == 5  # nearest from top6, not nearest low-priority point7
    assert len(set(info["greedy_sensor_ids"])) == 6
    target = (action + 1) * 2000
    for xy, index in zip(target, info["greedy_sensor_ids"]):
        assert np.max(np.abs(xy - positions[index])) <= 20.001
        assert np.linalg.norm(xy - positions[index]) < ec["uavs"]["coverage_radius_m"]


def test_sampler_resume_reproduces_proposals_exactly():
    ec, cfg = configs()
    sampler = WaypointSampler(cfg["exploration"], ec, 52)
    obs = PersistentEnv(ec, cfg).observation(); actor = np.zeros((6, 2), np.float32)
    sampler.propose(obs, actor, 100)
    state = sampler.state_dict(); a, ma = sampler.propose(obs, actor, 101)
    sampler.load_state_dict(state); b, mb = sampler.propose(obs, actor, 101)
    assert np.array_equal(a, b) and ma == mb


def test_replay_uses_executed_mixed_action_and_duration_discount():
    ec, cfg = configs(True)
    trainer = Trainer(ec, cfg)
    while not trainer.env.done:
        row = trainer.step()
        replay = trainer.replay.items[-1]
        assert np.array_equal(replay[1], np.asarray(row["action"], np.float32))
        assert replay[2] == row["reward"] and replay[5] == row["physical_steps"] * 60
        if not row["done"]:
            assert not replay[4]
    assert trainer.episodes == 1 and trainer.steps == 8 and trainer.updates > 0
    terminal = trainer.replay.items[-1]
    out = trainer.sac.update([terminal, terminal])
    assert out["discount_mean"] == 0 and np.isclose(out["target_mean"], terminal[2], rtol=1e-5)


def test_multistep_bootstrap_target_matches_gamma_power():
    ec, cfg = configs(True); trainer = Trainer(ec, cfg)
    obs = trainer.env.observation(); original = trainer.sac.actor.forward
    def no_entropy(batch, deterministic=False):
        action, logp = original(batch, deterministic); return action, logp * 0
    trainer.sac.actor.forward = no_entropy
    def constant_target(batch, action):
        q = torch.full((len(action),), 7.0); return q, q
    trainer.sac.target.forward = constant_target
    row = (obs, np.zeros((6, 2), np.float32), -3.0, obs, False, 180.0)
    result = trainer.sac.update([row, row])
    assert np.isclose(result["discount_mean"], .95**3)
    assert np.isclose(result["target_mean"], -3 + .95**3 * 7)


def test_exact_resume_after_group_change_and_episode_end():
    for decisions in (1, 2, 4):
        ec, cfg = configs(True); first = Trainer(ec, cfg)
        for _ in range(decisions): first.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"; first.save(path)
            a = first.step(); weights = {k:v.clone() for k,v in first.sac.actor.state_dict().items()}
            second = Trainer.load(path); b = second.step()
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
            assert all(torch.equal(v, second.sac.actor.state_dict()[k]) for k,v in weights.items())


def test_eval_actor_only_mean_and_training_rng_unchanged():
    ec, cfg = configs(True); trainer = Trainer(ec, cfg)
    trainer.step(); sampler = trainer.sampler.state_dict(); torch_rng = torch.get_rng_state().clone()
    a = trainer.evaluate(1001); b = trainer.evaluate(1001)
    assert a == b and sampler == trainer.sampler.state_dict()
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert a["steps"] == 8 and a["simulation_sec"] == 480


def test_energy_is_disabled_and_physical_images_are_saved():
    ec, cfg = configs(True); trainer = Trainer(ec, cfg)
    trainer.env.scene.uav_battery[:] = 0
    trainer.step()
    assert trainer.env.steps > 0
    with tempfile.TemporaryDirectory() as folder:
        trainer.evaluate(1001, folder, detailed=True)
        assert all((Path(folder)/name).exists() for name in ["trajectory.png", "fleet_overlay.png", "dashboard.png", "trace.json"])
        trace = json.loads((Path(folder)/"trace.json").read_text())
        for row in trace["rows"]:
            assert max(row["move_distance_by_uav_m"]) <= 180.001


def test_inactive_context_and_noncontiguous_action_id_packing():
    ec, cfg = configs(True); trainer = Trainer(ec, cfg)
    env = trainer.env
    env.long_ids = np.array([1, 3, 5]); env.short_ids = np.array([0, 2, 4])
    env.horizons = np.array([2, 4, 2, 4, 2, 4]); env.phase = 2
    env.scene.uav_pos[:] = np.column_stack([np.arange(6) * 500, np.arange(6) * 200])
    obs = env.observation(); assert obs["ids"].tolist() == [1, 3, 5]
    packed = trainer.sac.pack([obs])
    assert packed["mask"][0].tolist() == [False, True, False, True, False, True]
    assert packed["context_mask"].all()
    assert np.allclose(packed["uavs"][0, :, :2], env.scene.uav_pos / 4000)
    assert packed["uavs"].shape[-1] == 7
    actual = np.array([[.1, .2], [.3, .4], [.5, .6]], np.float32)
    captured = []
    def hook(module, args):
        captured.append(args[1].detach().clone())
    handle = trainer.sac.critic.register_forward_pre_hook(hook)
    row = (obs, actual, -1.0, obs, False, 60.0)
    trainer.sac.update([row, row]); handle.remove()
    assert np.allclose(captured[0][0, [1, 3, 5]], actual)
    assert torch.equal(captured[0][0, [0, 2, 4]], torch.zeros(3, 2))
    with torch.no_grad():
        full, _ = trainer.sac.actor(packed, deterministic=True)
    assert np.allclose(trainer.sac.act(obs, deterministic=True), full[0, [1, 3, 5]])


if __name__ == "__main__":
    torch.set_num_threads(1)
    names = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); names.append(name); print("PASS", name, flush=True)
    print(json.dumps(dict(passed=len(names), tests=names)), flush=True)
