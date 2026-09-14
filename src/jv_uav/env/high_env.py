from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .dynamics import compute_return_energy_frac
from .low_env import VariableUAVLowEnv
from .scene import UAVStatus


LowPolicy = Callable[[dict[str, np.ndarray]], np.ndarray]


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing config section: {name}")
    return value


@dataclass
class UpperTransition:
    """The resolved upper decision and its return-trip consequences."""

    requested_status: np.ndarray
    assigned_status: np.ndarray
    return_distance: np.ndarray
    return_time_sec: np.ndarray
    return_energy_frac: np.ndarray
    battery_after_return: np.ndarray
    charging_time_sec: np.ndarray


class VariableUAVHighEnv:
    """Synchronous upper-level charging environment.

    One call to :meth:`step` resolves a charging decision, runs one complete
    lower-level block, and then applies charging for the part of the frame
    remaining after each UAV's return trip.

    Upper action for each UAV is the integer value of ``UAVStatus``:

    - ``0``: SERVING
    - ``1``: WAITING
    - ``2``: CHARGING

    DEAD UAV actions are ignored.  No persistent RETURNING state is needed:
    return time and energy are accounted for inside the current frame.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        low_env: VariableUAVLowEnv | None = None,
    ) -> None:
        self.cfg = cfg
        self.low_env = low_env or VariableUAVLowEnv(cfg)
        self.scene = self.low_env.scene

        time_cfg = _section(cfg, "time")
        uav_cfg = _section(cfg, "uav")
        energy_cfg = _section(cfg, "energy")
        upper_cfg = _section(cfg, "upper")

        self.n_uavs = self.scene.max_uavs
        self.frame_sec = (
            float(time_cfg.get("low_step_sec", 60.0))
            * int(time_cfg["low_steps_per_high"])
        )
        self.speed = float(uav_cfg.get("speed", 6.0))
        self.p_horiz = float(energy_cfg.get("P_horiz", 257.0))
        self.battery_capacity_wh = float(
            energy_cfg.get("battery_capacity_Wh", 125.0)
        )

        self.max_charging_slots = int(upper_cfg.get("max_charging_slots", 2))
        self.max_waiting_slots = int(upper_cfg.get("max_waiting_slots", 2))
        self.charge_power_one_w = float(
            upper_cfg.get("charge_power_one_W", 600.0)
        )
        self.charge_power_multi_w = float(
            upper_cfg.get("charge_power_multi_W", 500.0)
        )
        self.strict_return_within_frame = bool(
            upper_cfg.get("strict_return_within_frame", True)
        )

        if self.max_charging_slots < 0 or self.max_waiting_slots < 0:
            raise ValueError("upper slot counts must be non-negative")
        if self.frame_sec <= 0.0:
            raise ValueError("upper frame duration must be positive")

        self.high_step = 0
        self.history: dict[str, list[Any]] = {}

    def reset(
        self,
        *,
        initial_status: np.ndarray | None = None,
        reset_dyn_rng: bool = False,
    ) -> dict[str, np.ndarray]:
        self.low_env.reset(
            initial_status=initial_status,
            reset_dyn_rng=reset_dyn_rng,
        )
        self.high_step = 0
        self.history = {
            "requested_status": [],
            "assigned_status": [self.scene.uav_status.copy()],
            "battery": [self.scene.uav_battery.copy()],
            "return_time_sec": [],
            "return_energy_frac": [],
            "charging_time_sec": [],
            "charge_added_frac": [],
            "upper_reward": [],
            "low_reward": [],
        }
        return self._build_obs()

    def step(
        self,
        upper_actions: np.ndarray,
        low_policy: LowPolicy,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Run one upper frame.

        ``low_policy`` receives the current lower observation once per lower
        step.  It may return either actions for all UAVs with shape
        ``(n_uavs, 2)`` or actions only for active serving UAVs with shape
        ``(n_active, 2)``.
        """
        status_at_frame_start = self.scene.uav_status.copy()
        transition = self._apply_upper_decision(upper_actions)

        total_low_reward = 0.0
        low_infos: list[dict[str, Any]] = []
        done = False
        for _ in range(self.low_env.low_steps_per_high):
            low_obs = self.low_env.get_obs()
            full_actions = self._expand_low_actions(low_policy(low_obs), low_obs)
            _, low_reward, done, low_info = self.low_env.step(full_actions)
            total_low_reward += float(low_reward)
            low_infos.append(low_info)
            if done:
                break

        charge_added = self._apply_charging(transition.charging_time_sec)
        upper_reward, reward_terms = self._compute_reward(
            total_low_reward=total_low_reward,
            previous_status=status_at_frame_start,
        )

        self.high_step += 1
        self._record(transition, charge_added, upper_reward, total_low_reward)
        obs = self._build_obs()
        info = {
            "high_step": int(self.high_step),
            "requested_status": transition.requested_status.copy(),
            "assigned_status": transition.assigned_status.copy(),
            "final_status": self.scene.uav_status.copy(),
            "return_distance": transition.return_distance.copy(),
            "return_time_sec": transition.return_time_sec.copy(),
            "return_energy_frac": transition.return_energy_frac.copy(),
            "battery_after_return": transition.battery_after_return.copy(),
            "charging_time_sec": transition.charging_time_sec.copy(),
            "charge_added_frac": charge_added.copy(),
            "low_reward": float(total_low_reward),
            "reward_terms": reward_terms,
            "low_infos": low_infos,
        }
        return obs, float(upper_reward), bool(done), info

    def _apply_upper_decision(
        self,
        actions: np.ndarray,
    ) -> UpperTransition:
        requested = np.asarray(actions, dtype=np.int8)
        if requested.shape != (self.n_uavs,):
            raise ValueError(
                f"upper_actions must have shape ({self.n_uavs},), "
                f"got {requested.shape}"
            )
        valid = [
            int(UAVStatus.SERVING),
            int(UAVStatus.WAITING),
            int(UAVStatus.CHARGING),
        ]
        alive = self.scene.uav_status != int(UAVStatus.DEAD)
        if not np.isin(requested[alive], valid).all():
            raise ValueError("alive UAV actions must be SERVING, WAITING, or CHARGING")

        return_distance, return_time, return_energy = self._return_features()
        battery_after_return = self.scene.uav_battery - return_energy
        assigned = self._resolve_slot_limits(requested, battery_after_return)
        assigned[~alive] = int(UAVStatus.DEAD)

        returning = alive & (assigned != int(UAVStatus.SERVING))
        if (
            self.strict_return_within_frame
            and np.any(return_time[returning] > self.frame_sec + 1e-6)
        ):
            ids = np.flatnonzero(returning & (return_time > self.frame_sec))
            raise ValueError(
                "UAVs cannot return within one upper frame: "
                f"{ids.tolist()}"
            )

        self.scene.uav_battery[returning] -= return_energy[returning]
        failed_return = returning & (self.scene.uav_battery <= 0.0)
        arrived = returning & ~failed_return
        self.scene.uav_battery[failed_return] = 0.0
        assigned[failed_return] = int(UAVStatus.DEAD)

        # Position is updated to the frame-end location.  Return travel remains
        # visible through the time and energy fields in UpperTransition.
        self.scene.uav_prev_pos[arrived] = self.scene.uav_pos[arrived]
        self.scene.uav_pos[arrived] = self.scene.airship_pos
        self.scene.set_uav_status(assigned)

        charging = assigned == int(UAVStatus.CHARGING)
        charging_time = np.zeros(self.n_uavs, dtype=np.float32)
        charging_time[charging] = np.maximum(
            0.0,
            self.frame_sec - return_time[charging],
        )
        return UpperTransition(
            requested_status=requested.copy(),
            assigned_status=assigned.copy(),
            return_distance=return_distance,
            return_time_sec=return_time,
            return_energy_frac=return_energy,
            battery_after_return=self.scene.uav_battery.copy(),
            charging_time_sec=charging_time,
        )

    def _resolve_slot_limits(
        self,
        requested: np.ndarray,
        battery_after_return: np.ndarray,
    ) -> np.ndarray:
        assigned = requested.copy()
        alive = self.scene.uav_status != int(UAVStatus.DEAD)

        charging = np.flatnonzero(
            alive & (requested == int(UAVStatus.CHARGING))
        )
        charging_order = charging[
            np.lexsort((charging, battery_after_return[charging]))
        ]
        accepted_charging = charging_order[: self.max_charging_slots]
        rejected_charging = charging_order[self.max_charging_slots :]
        assigned[rejected_charging] = int(UAVStatus.WAITING)

        waiting = np.flatnonzero(
            alive & (assigned == int(UAVStatus.WAITING))
        )
        waiting_order = waiting[
            np.lexsort((waiting, battery_after_return[waiting]))
        ]
        accepted_waiting = waiting_order[: self.max_waiting_slots]
        rejected_waiting = waiting_order[self.max_waiting_slots :]
        assigned[rejected_waiting] = int(UAVStatus.SERVING)

        # These variables make the two capacity decisions easy to inspect in a
        # debugger and avoid relying on incidental ordering in np.flatnonzero.
        _ = accepted_charging, accepted_waiting
        return assigned

    def _return_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distance = np.linalg.norm(
            self.scene.uav_pos - self.scene.airship_pos[None, :], axis=1
        ).astype(np.float32)
        return_time = (distance / self.speed).astype(np.float32)
        return_energy = compute_return_energy_frac(
            self.scene.uav_pos,
            self.scene.airship_pos,
            speed=self.speed,
            p_horiz=self.p_horiz,
            battery_capacity_wh=self.battery_capacity_wh,
        )
        return distance, return_time, return_energy

    def _expand_low_actions(
        self,
        actions: np.ndarray,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == (self.n_uavs, 2):
            return actions

        active_ids = np.asarray(obs["active_uav_ids"], dtype=np.int64)
        if actions.shape != (active_ids.size, 2):
            raise ValueError(
                "low_policy must return shape "
                f"({self.n_uavs}, 2) or ({active_ids.size}, 2), "
                f"got {actions.shape}"
            )
        full = np.zeros((self.n_uavs, 2), dtype=np.float32)
        full[active_ids] = actions
        return full

    def _apply_charging(self, charging_time_sec: np.ndarray) -> np.ndarray:
        charging_ids = self.scene.charging_ids()
        added = np.zeros(self.n_uavs, dtype=np.float32)
        if charging_ids.size == 0:
            return added

        power_w = (
            self.charge_power_one_w
            if charging_ids.size == 1
            else self.charge_power_multi_w
        )
        added[charging_ids] = (
            power_w
            * charging_time_sec[charging_ids]
            / 3600.0
            / self.battery_capacity_wh
        )
        before = self.scene.uav_battery.copy()
        self.scene.uav_battery[charging_ids] = np.minimum(
            1.0,
            self.scene.uav_battery[charging_ids] + added[charging_ids],
        )
        return (self.scene.uav_battery - before).astype(np.float32)

    def _build_obs(self) -> dict[str, np.ndarray]:
        upper_cfg = _section(self.cfg, "upper")
        distance, return_time, return_energy = self._return_features()
        battery_after_return = self.scene.uav_battery - return_energy

        status_onehot = np.eye(len(UAVStatus), dtype=np.float32)[
            self.scene.uav_status
        ]
        positions = self.scene.uav_pos.astype(np.float32, copy=True)
        if bool(upper_cfg.get("normalize_positions", True)):
            positions /= self.scene.world_size

        dmax = np.sqrt(2.0) * self.scene.world_size
        per_uav_obs = np.concatenate(
            [
                self.scene.uav_battery[:, None],
                status_onehot,
                positions,
                (distance / max(dmax, 1e-6))[:, None],
                (return_time / self.frame_sec)[:, None],
                return_energy[:, None],
                battery_after_return[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        max_backlog = (
            float(self.scene.user_pkts.max())
            if self.scene.user_pkts.size
            else 0.0
        )
        mean_backlog = (
            float(self.scene.user_pkts.mean())
            if self.scene.user_pkts.size
            else 0.0
        )
        backlog_ref = float(upper_cfg.get("backlog_ref", 800.0))
        global_obs = np.asarray(
            [
                self.scene.charging_ids().size / max(1, self.max_charging_slots),
                self.scene.waiting_ids().size / max(1, self.max_waiting_slots),
                self.scene.serving_ids().size / max(1, self.n_uavs),
                max_backlog / max(backlog_ref, 1e-6),
                mean_backlog / max(backlog_ref, 1e-6),
            ],
            dtype=np.float32,
        )
        return {
            "per_uav_obs": per_uav_obs,
            "global_obs": global_obs,
            "uav_status": self.scene.uav_status.copy(),
            "agent_active": self.scene.uav_status != int(UAVStatus.DEAD),
        }

    def _compute_reward(
        self,
        *,
        total_low_reward: float,
        previous_status: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        reward_cfg = _section(self.cfg, "upper_reward")
        newly_dead = int(
            np.sum(
                (previous_status != int(UAVStatus.DEAD))
                & (self.scene.uav_status == int(UAVStatus.DEAD))
            )
        )
        serving_ratio = self.scene.serving_ids().size / max(1, self.n_uavs)
        waiting_ratio = self.scene.waiting_ids().size / max(1, self.n_uavs)
        backlog = (
            float(self.scene.user_pkts.max())
            if self.scene.user_pkts.size
            else 0.0
        )
        backlog_ref = float(reward_cfg.get("backlog_ref", 800.0))

        terms = {
            "low": float(reward_cfg.get("low_reward_weight", 0.0))
            * total_low_reward,
            "serving": float(reward_cfg.get("serving_weight", 1.0))
            * serving_ratio,
            "waiting": -float(reward_cfg.get("waiting_weight", 1.0))
            * waiting_ratio,
            "backlog": -float(reward_cfg.get("backlog_weight", 1.0))
            * backlog
            / max(backlog_ref, 1e-6),
            "death": -float(reward_cfg.get("death_penalty", 10.0))
            * newly_dead,
        }
        return float(sum(terms.values())), terms

    def _record(
        self,
        transition: UpperTransition,
        charge_added: np.ndarray,
        upper_reward: float,
        low_reward: float,
    ) -> None:
        self.history["requested_status"].append(
            transition.requested_status.copy()
        )
        self.history["assigned_status"].append(self.scene.uav_status.copy())
        self.history["battery"].append(self.scene.uav_battery.copy())
        self.history["return_time_sec"].append(
            transition.return_time_sec.copy()
        )
        self.history["return_energy_frac"].append(
            transition.return_energy_frac.copy()
        )
        self.history["charging_time_sec"].append(
            transition.charging_time_sec.copy()
        )
        self.history["charge_added_frac"].append(charge_added.copy())
        self.history["upper_reward"].append(float(upper_reward))
        self.history["low_reward"].append(float(low_reward))