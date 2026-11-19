"""Independent mechanism, scheduling, learning-boundary and resume checks."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/async_waypoint")]
from jv_uav.config import load_config
from jv_uav.scene import Scene
from jv_uav.models import SimpleEnergyModel, LowestArrivalBatteryScheduler
from engine import Engine, Role, Phase, allocate, RETURN_PHASES
from learning import duration_gae, temporal_factors, upper_observation, Replay
from trainer import Trainer
from verify_reference import reference_run


def configs():
    return load_config(ROOT / "configs/default.yaml"), yaml.safe_load((ROOT / "configs/training_async_waypoint.yaml").read_text())


def tiny():
    ec, cfg = configs()
    cfg.update(episode_sec=180.0, upper_period_sec=60.0, upper_rollout_steps=4)
    cfg["upper"].update(hidden=16, epochs=2, minibatch_frames=2)
    cfg["lower"].update(hidden=16, heads=2, batch_size=2, replay_capacity=256, learning_starts=0, random_steps=0)
    return ec, cfg


def resolve_ready(env, action=None):
    while env.current_uid is not None:
        env.apply_lower(np.zeros(2) if action is None else action)


def test_allocator_matches_existing_controller_without_inbound_reservations():
    ec, cfg = configs()
    rng = np.random.default_rng(312)
    scene = Scene(ec); old = LowestArrivalBatteryScheduler(ec, SimpleEnergyModel(ec))
    for _ in range(250):
        order = rng.permutation(6)
        role = np.zeros(6, np.int64)
        nc, nw = rng.integers(0, 3, size=2)
        role[order[:nc]] = Role.CHARGE
        role[order[nc:nc + nw]] = Role.WAIT
        phase = np.where(role == Role.CHARGE, Phase.CHARGE, np.where(role == Role.WAIT, Phase.WAIT, Phase.MOVE))
        scene.uav_status[:] = role
        scene.uav_pos[:] = rng.uniform(0, 4000, (6, 2))
        scene.uav_pos[role != Role.SERVE] = scene.airship_pos
        scene.uav_battery[:] = rng.uniform(.01, 1, 6)
        action = rng.integers(0, 2, 6)
        trip = old.energy_model.return_trip(scene.uav_pos, scene.airship_pos)
        cost = trip.energy_frac.copy(); cost[role != Role.SERVE] = 0
        result, mask = allocate(action, role, phase, scene.uav_battery - cost, 2, 2)
        assert np.array_equal(result, old.make_plan(scene, action).assigned_status)
        assert mask.all()


def test_retained_chargers_waiters_priority_and_rejected_requests():
    role = np.array([2, 2, 1, 1, 0, 0])
    phase = np.array([5, 5, 6, 6, 1, 1])
    soc = np.array([.99, .98, .8, .7, .01, .02])
    assigned, _ = allocate(np.ones(6), role, phase, soc, 2, 2)
    assert assigned.tolist() == [2, 2, 1, 1, 0, 0]
    assigned, _ = allocate(np.array([0, 1, 1, 1, 1, 1]), role, phase, soc, 2, 2)
    assert assigned[0] == Role.SERVE and assigned[1] == Role.CHARGE
    assert assigned[3] == Role.CHARGE  # Existing waiter precedes lower-SOC newcomers.
    assert sum(assigned == Role.CHARGE) == sum(assigned == Role.WAIT) == 2


def test_same_time_decisions_commit_intentions_without_time_or_energy():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.apply_upper(np.zeros(6, int)); order = list(e.queue)
    before = e.soc.copy(); backlog = e.backlog.copy()
    uid = e.current_uid; e.apply_lower(np.array([.5, -.5]))
    obs = e.observation()
    assert e.now == 0 and np.array_equal(before, e.soc) and np.array_equal(backlog, e.backlog)
    assert np.allclose(obs["team"][uid, :2], [.5, .5])
    assert np.allclose(obs["team"][uid, 2:4], [.75, .25])
    assert e.phase[uid] == Phase.MOVE and e.current_uid == order[1]
    assert obs["uid"] == order[1] and obs["team"].shape == (6, 25)


def test_continue_service_preserves_target_and_remaining_timer():
    ec, cfg = configs(); cfg["upper_period_sec"] = 10.0
    e = Engine(ec, cfg); e.apply_upper(np.zeros(6, int)); resolve_ready(e)
    e.advance(); assert e.now == 10 and np.all(e.remaining == 20)
    targets = e.target.copy(); e.apply_upper(np.zeros(6, int))
    assert e.queue == [] and np.array_equal(e.target, targets) and np.all(e.remaining == 20)
    e.advance(); assert e.now == 20 and np.all(e.remaining == 10)
    e.apply_upper(np.zeros(6, int)); e.advance()
    assert e.now == 30 and e.upper_due and not e.queue
    assert len(e.collections) == 1


def test_continue_flight_preserves_target_and_actual_position():
    ec, cfg = configs(); cfg["upper_period_sec"] = 10
    e = Engine(ec, cfg); e.apply_upper(np.zeros(6, int)); resolve_ready(e, np.array([.1, 0]))
    e.advance(); assert e.now == 10
    assert np.allclose(e.pos[:, 0], 2060) and np.allclose(e.remaining, 200 / 6 - 10)
    e.apply_upper(np.zeros(6, int)); e.advance()
    assert np.allclose(e.pos[:, 0], 2120)
    assert np.allclose(e.target[:, 0], 2200)


def test_recall_starts_from_interpolated_position_and_cancels_old_task():
    ec, cfg = configs(); cfg["upper_period_sec"] = 10
    e = Engine(ec, cfg); e.apply_upper(np.zeros(6, int)); resolve_ready(e, np.array([.5, 0]))
    e.advance(); origin = e.pos[0].copy()
    request = np.zeros(6, int); request[0] = 1
    e.apply_upper(request)
    assert e.phase[0] == Phase.RETURN_CLIMB and e.role[0] == Role.CHARGE
    assert np.array_equal(e.pos[0], origin) and np.array_equal(e.target[0], e.home)
    assert not e.collections


def test_reservation_locks_across_short_upper_periods():
    ec, cfg = configs(); cfg["upper_period_sec"] = 60
    e = Engine(ec, cfg); e.pos[0] = [3200, 2000]
    request = np.zeros(6, int); request[0] = 1
    e.apply_upper(request); resolve_ready(e, np.array([1, 1]))
    assert e.role[0] == Role.CHARGE and e.phase[0] in RETURN_PHASES
    while not e.upper_due:
        e.advance(); resolve_ready(e, np.array([1, 1]))
    assert e.now == 60 and e.phase[0] == Phase.RETURN_FLY
    row = e.apply_upper(np.zeros(6, int))
    assert not row["policy_mask"][0] and e.role[0] == Role.CHARGE
    assert e.phase[0] == Phase.RETURN_FLY
    assert row["reserved_charging"] == 1
    while e.now < 240 - 1e-7:
        if e.upper_due:
            e.apply_upper(np.zeros(6, int))
        elif e.current_uid is not None:
            resolve_ready(e, np.array([1, 1]))
        else:
            e.advance()
    assert e.phase[0] == Phase.CHARGE and not e.done
    assert np.isclose(e.soc[0], 1 - (400 * 10 + 257 * 200) / 450000 + 500 * 30 / 450000)


def test_charging_begins_only_after_arrival_waiting_consumes_no_energy():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.apply_upper(np.array([1, 1, 1, 1, 0, 0])); resolve_ready(e)
    e.advance(); assert e.now == 10
    assert e.phase[:2].tolist() == [Phase.CHARGE] * 2
    assert e.phase[2:4].tolist() == [Phase.WAIT] * 2
    after_arrival = e.soc.copy()
    expected = 1 - 400 * 10 / 450000
    assert np.allclose(after_arrival[:4], expected)
    e.advance(); assert e.now == 30
    assert np.all(e.soc[:2] == 1) and np.array_equal(e.soc[2:4], after_arrival[2:4])
    assert np.all(e.phase[:2] == Phase.CHARGE)  # Full does not auto-release.


def test_union_clear_and_arrivals_are_counted_once():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.xy[:] = [0, 0]; e.xy[0] = e.home; e.backlog[:] = 80; e.backlog[0] = 100
    e.initial_packets = float(e.backlog.sum())
    class Fluid:
        def poisson(self, expected):
            return expected
    e.rng = Fluid()
    e.apply_upper(np.zeros(6, int)); resolve_ready(e); e.advance()
    assert e.now == 30 and e.collections[0]["sensor_ids"] == [0]
    assert np.isclose(sum(e.collections[0]["by_uid"]), 115)
    assert e.backlog[0] == 0 and np.all(e.backlog[1:] == 95)
    assert np.isclose(e.area, 100 * 30 + .5 * .5 * 30**2)
    resolve_ready(e); e.advance()
    assert np.isclose(sum(e.collections[1]["by_uid"]), 15)
    assert np.all(e.backlog[1:] == 110)


def test_service_complete_precedes_same_time_upper_recall():
    ec, cfg = configs(); cfg["upper_period_sec"] = 30
    e = Engine(ec, cfg); e.xy[:] = e.home; e.backlog[:] = 100
    e.initial_packets = float(e.backlog.sum())
    e.apply_upper(np.zeros(6, int)); resolve_ready(e); e.advance()
    assert e.upper_due and not e.queue and e.collected > 0
    e.apply_upper(np.ones(6, int))
    assert e.collections[0]["time"] == 30
    assert sum(e.role == Role.SERVE) == 2


def test_exact_terminal_time_does_not_complete_future_service():
    ec, cfg = configs(); cfg["episode_sec"] = 35
    e = Engine(ec, cfg); e.apply_upper(np.zeros(6, int)); resolve_ready(e)
    e.advance(); resolve_ready(e); e.advance()
    assert e.now == 35 and e.done and len(e.collections) == 1
    assert np.allclose(e.soc, 1 - 326 * 35 / 450000)


def test_depletion_precedes_collection_and_is_assessed_once():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.soc[:] = 326 * 30 / 450000
    e.apply_upper(np.zeros(6, int)); resolve_ready(e); e.advance()
    assert e.done and len(e.failures) == 6 and not e.collections
    assert all(f["upper_weight"] == 1 and f["lower_weight"] == 0 for f in e.failures)
    assert np.isclose(e.upper_total, 5 * 6 * 30 / 600 / 10 - 20 * 6 / 10)
    try:
        e._die(0, "service_depletion")
        raise AssertionError("duplicate death allowed")
    except RuntimeError:
        pass


def test_return_zero_soc_landing_is_consistent_with_approval():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.soc[0] = 400 * 10 / 450000
    action = np.zeros(6, int); action[0] = 1
    e.apply_upper(action); resolve_ready(e, np.array([1, 1])); e.advance()
    assert not e.done and e.phase[0] == Phase.CHARGE and np.isclose(e.soc[0], 0, atol=1e-7)


def test_lower_reference_failure_and_tail_are_once_and_weighted():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.pos[0] = [3200, 2000]; e.soc[0] = .01
    e.references[0] = dict(start_soc=.8, reference_required_soc=.6, optimistic_required_soc=.4)
    a = np.zeros(6, int); a[0] = 1
    e.apply_upper(a)
    assert e.done and len(e.failures) == 1
    f = e.failures[0]; assert f["lower_weight"] == 1 and f["upper_weight"] == 0
    assert e.terms["failure"] == -500 and np.isclose(e.terms["distance"], -500 * 1200 / 1800)
    assert e.terms["tail"] < 0 and np.isclose(e.lower_total, sum(e.terms.values()))
    old = e.lower_total; e._finish_if_needed(); assert e.lower_total == old


def test_shared_reference_responsibility_has_no_overlap():
    ec, cfg = configs(); e = Engine(ec, cfg)
    e.pos[0] = [3200, 2000]; e.soc[0] = .01
    e.references[0] = dict(start_soc=.5, reference_required_soc=.6, optimistic_required_soc=.4)
    a = np.zeros(6, int); a[0] = 1
    e.apply_upper(a)
    f = e.failures[0]
    assert f["upper_weight"] == f["lower_weight"] == .5
    assert f["upper_weight"] + f["lower_weight"] == 1
    assert e.terms["failure"] == -250 and e.upper_total == -1


def test_duration_gae_hand_calculation_and_reset_boundary():
    _, cfg = configs(); up = cfg["upper"]
    adv, target = duration_gae([1, 2], [10, 20], [False, True], [600, 300], 99999, up)
    assert np.allclose(adv, [10 - .95 * .95 * 18, -18])
    assert np.allclose(target, adv + [10, 20])
    adv, _ = duration_gae([1], [10], [False], [300], 20, up)
    assert np.isclose(adv[0], 1 + np.sqrt(.95) * 20 - 10)
    adv, _ = duration_gae([1, 999], [10, 99999], [True, False], [600, 600], 123, up)
    assert adv[0] == -9
    g, l = temporal_factors([0, 300, 600], .95, .95, 600)
    assert g[0] == l[0] == 1 and np.isclose(g[1], np.sqrt(.95))


def test_upper_rollout_spans_resets_and_updates_only_at_full_T():
    ec, cfg = tiny(); t = Trainer(ec, cfg)
    for _ in range(3):
        t.step()
    assert t.episodes == 1 and len(t.rollout) == 3 and t.upper_updates == 0
    t.step()
    assert t.upper_updates == 1 and t.last_losses["upper_rollout_T"] == 4 and not t.rollout
    assert np.isclose(t.last_losses["upper_rollout_sec"], 240)


def test_global_lower_stream_reward_conservation_and_zero_time_links():
    ec, cfg = tiny(); t = Trainer(ec, cfg)
    for _ in range(3):
        t.step()
    assert t.episode_over and len(t.replay.rows) == t.lower_steps
    assert t.replay.rows[-1][4] is True
    assert sum(r[5] == 0 for r in t.replay.rows) >= 1
    assert np.isclose(sum(r[2] for r in t.replay.rows) + t.unattributed_lower_reward, t.env.lower_total)
    assert np.isclose(sum(r[5] for r in t.replay.rows), t.env.now)
    assert all(np.isfinite(v) for v in t.last_losses.values())


def test_terminal_sac_target_and_single_uav_action_shape():
    ec, cfg = tiny(); t = Trainer(ec, cfg); t.step()
    obs = t.env.observation(); action = np.array([.2, -.1], np.float32)
    row = (obs, action, -123.0, obs, True, 60.0)
    result = t.lower.update([row, row])
    assert result["lower_target_mean"] == -123
    assert t.lower.act(obs).shape == (2,)
    assert np.isclose(result["lower_actor_loss"], result["lower_policy_loss"] + result["lower_entropy_loss"], rtol=1e-5)


def test_eval_does_not_change_any_training_rng_or_physics():
    ec, cfg = tiny(); t = Trainer(ec, cfg); t.step()
    state = t.env.state_dict(); rng = torch.get_rng_state().clone()
    replay_state = copy.deepcopy(t.replay.rng.bit_generator.state)
    first = t.evaluate(1001); second = t.evaluate(1001)
    assert first == second and torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(t.env.soc, state["values"]["soc"])
    assert t.env.now == state["values"]["now"]
    assert replay_state == t.replay.rng.bit_generator.state


def test_exact_resume_with_pending_lower_and_partial_ppo_rollout():
    for count in (1, 3, 4):
        ec, cfg = tiny(); a = Trainer(ec, cfg)
        for _ in range(count):
            a.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"; a.save(path)
            row_a = a.step()
            actor_a = {k: v.clone() for k, v in a.lower.actor.state_dict().items()}
            b = Trainer.load(path); row_b = b.step()
            assert json.dumps(row_a, sort_keys=True) == json.dumps(row_b, sort_keys=True)
            assert all(torch.equal(v, b.lower.actor.state_dict()[k]) for k, v in actor_a.items())


def test_return_cost_does_not_charge_climb_twice():
    ec, cfg = configs(); cfg["upper_period_sec"] = 5
    e = Engine(ec, cfg); e.pos[0] = [3200, 2000]
    request = np.zeros(6, int); request[0] = 1
    e.apply_upper(request); resolve_ready(e, np.ones(2))
    e.advance()
    assert e.now == 5 and e.phase[0] == Phase.RETURN_CLIMB
    assert np.isclose(e.return_cost()[0], (400 * 5 + 257 * 200) / 450000)
    e.apply_upper(np.zeros(6, int)); e.advance()
    assert e.phase[0] == Phase.RETURN_FLY
    assert np.isclose(e.return_cost()[0], 257 * 200 / 450000)


def test_energy_and_packet_conservation_over_reference_8_hours():
    ec, cfg = configs()
    for seed in (1001, 1002, 1003):
        e = reference_run(ec, cfg, seed)
        assert e.now == 28800 and not e.failures
        assert len(e.seen) == 150
        assert np.isclose(e.upper_total, 48)
        assert np.isclose(e.initial_packets + e.arrived_packets - e.collected, e.backlog.sum())
        for segment in e.segments:
            dt = segment["end"] - segment["start"]
            phase = np.array(segment["phase_start"])
            power = np.zeros(6)
            power[np.isin(phase, [Phase.MOVE, Phase.RETURN_FLY])] = 257
            power[phase == Phase.SERVICE] = 326
            power[phase == Phase.RETURN_CLIMB] = 400
            expected = np.maximum(np.array(segment["soc_start"]) - power * dt / 450000, 0)
            charging = phase == Phase.CHARGE
            expected[charging] = np.minimum(1, expected[charging] + 500 * dt / 450000)
            assert np.allclose(expected, segment["soc_end"], atol=1e-12)
            displacement = np.linalg.norm(np.array(segment["positions_end"]) - segment["positions_start"], axis=1)
            assert np.all(displacement <= 6 * dt + 1e-6)


def test_sac_nonterminal_time_discount_and_zero_time_links():
    ec, cfg = tiny(); t = Trainer(ec, cfg); t.step()
    obs = t.env.observation(); action = np.zeros(2, np.float32)
    rows = [(obs, action, -2.0, obs, False, 0.0), (obs, action, -4.0, obs, False, 60.0)]
    t.lower.cfg["gamma"] = .5
    original = t.lower.actor.forward
    def actor_no_entropy(batch, deterministic=False):
        a, logp = original(batch, deterministic)
        return a, logp * 0
    t.lower.actor.forward = actor_no_entropy
    def constant_target(batch, a):
        return torch.full((len(a),), 7.0)
    t.lower.t1.forward = constant_target; t.lower.t2.forward = constant_target
    result = t.lower.update(rows)
    assert np.isclose(result["lower_target_mean"], ((-2 + 7) + (-4 + .5 * 7)) / 2)
    assert result["lower_zero_dt_fraction"] == .5


def test_upper_observation_and_masked_actor_loss():
    ec, cfg = tiny(); t = Trainer(ec, cfg)
    obs = upper_observation(t.env)
    assert obs["actor"].shape == (6, 30) and obs["critic"].shape == (154,)
    action, logp, value, _ = t.upper.act(obs)
    row = dict(obs=obs, action=action, logp=logp, value=value, mask=np.zeros(6), reward=1.0, done=True, duration=60)
    before = {k: v.clone() for k, v in t.upper.actor.state_dict().items()}
    result = t.upper.update([row, row], 0)
    assert result["upper_actor_loss"] == 0
    assert all(torch.equal(v, t.upper.actor.state_dict()[k]) for k, v in before.items())


def test_map_seed_unchanged_when_training_seed_changes():
    ec, cfg = configs()
    a = Engine(ec, cfg, seed=43); b = Engine(ec, cfg, seed=44)
    assert np.array_equal(a.xy, b.xy)
    assert not np.array_equal(a.backlog, b.backlog)


def test_nonfull_charger_can_leave_and_release_slot():
    ec, cfg = configs(); cfg["upper_period_sec"] = 20
    e = Engine(ec, cfg); e.soc[0] = .5
    request = np.zeros(6, int); request[0] = 1
    e.apply_upper(request); resolve_ready(e); e.advance(); e.advance()
    assert e.now == 20 and e.phase[0] == Phase.CHARGE and e.soc[0] < 1
    e.apply_upper(np.zeros(6, int))
    assert e.phase[0] == Phase.READY and 0 in e.queue
    assert e.role[0] == Role.SERVE and np.sum(e.role == Role.CHARGE) == 0


def test_terminal_upper_boundary_has_no_extra_action_or_reward():
    ec, cfg = configs(); cfg["episode_sec"] = 600
    e = Engine(ec, cfg); e.apply_upper(np.zeros(6, int))
    while not e.done:
        if e.current_uid is not None:
            resolve_ready(e)
        else:
            e.advance()
    assert e.now == 600 and not e.upper_due and e.queue == []
    assert len(e.allocations) == 1 and len(e.collections) == 20
    assert np.isclose(e.upper_total, 3)


def test_replay_copy_cyclic_overwrite_and_sampler_resume():
    r = Replay(3, 17)
    obs = dict(value=np.array([1.0]))
    for i in range(5):
        r.add(obs, np.zeros(2), float(i), obs, False, 0)
    obs["value"][0] = 999
    assert [row[2] for row in r.rows] == [3, 4, 2]
    assert all(row[0]["value"][0] == 1 for row in r.rows)
    clone = Replay(3, 99); clone.load_state_dict(copy.deepcopy(r.state_dict()))
    assert [row[2] for row in r.sample(3)] == [row[2] for row in clone.sample(3)]
    try:
        r.add(obs, np.zeros(2), 0, obs, False, -1)
        raise AssertionError("negative transition duration accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    torch.set_num_threads(1)
    passed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed.append(name); print("PASS", name, flush=True)
    print(json.dumps(dict(passed=len(passed), tests=passed)), flush=True)
