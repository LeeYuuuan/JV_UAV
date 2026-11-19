"""One per-UAV failure assessment shared by upper and lower rewards."""
from __future__ import annotations
import numpy as np


class FailureResponsibility:
    def __init__(self, cfg, energy):
        settings = cfg["failure_responsibility"]
        if settings.get("mode") != "reference_return_v1":
            raise ValueError("unknown failure responsibility mode")
        self.beta = float(settings.get("inward_fraction", .5))
        self.margin = float(settings.get("reference_margin_soc", .01))
        self.shared = float(settings.get("shared_upper_weight", .5))
        self.tolerance = float(settings.get("comparison_tolerance_soc", 1e-7))
        if not 0 < self.beta < 1 or not 0 <= self.shared <= 1:
            raise ValueError("inward fraction must be in (0,1), shared weight in [0,1]")
        if not np.isfinite(self.margin) or self.margin < 0 or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("responsibility margins must be finite and nonnegative")
        self.energy = energy
        self.references = {}
        self.starts = {}

    def begin_frame(self, plan, scene):
        self.references = {}
        self.frame_index = int(scene.upper_step)
        self.starts = {uid: dict(frame_start_soc=float(plan.frame_start_battery[uid]),
                                frame_start_xy=plan.frame_start_pos[uid].astype(float).tolist(),
                                frame_start_status=int(plan.frame_start_status[uid]),
                                assigned_status=int(plan.assigned_status[uid]))
                       for uid in range(scene.num_uavs)}
        n = scene.low_steps_per_frame
        dt = scene.low_step_sec
        speed = scene.speed_mps
        move_limit = scene.max_move_time_sec
        max_distance = n * speed * move_limit
        endpoint_energy = self.energy.serving_energy_fraction(np.array([0., move_limit]), np.array([dt, dt-move_limit]))
        service_lower_bound = float(endpoint_energy.min()) * n
        for uid in plan.serving_ids:
            start = np.asarray(plan.frame_start_pos[uid], np.float64)
            vector = np.asarray(scene.airship_pos, np.float64) - start
            distance = float(np.linalg.norm(vector))
            direction = vector / distance if distance else np.zeros(2)
            advance = self.beta * min(distance, max_distance)
            reference_end = start + direction * advance
            optimistic_end = start + direction * min(distance, max_distance)
            move_time = advance / (n * speed)
            reference_service = float(self.energy.serving_energy_fraction(np.array([move_time]), np.array([dt-move_time]))[0]) * n
            reference_trip = self.energy.return_trip(reference_end[None], scene.airship_pos)
            optimistic_trip = self.energy.return_trip(optimistic_end[None], scene.airship_pos)
            reference_return = float(reference_trip.energy_frac[0])
            lower_bound = service_lower_bound + float(optimistic_trip.energy_frac[0]) + scene.return_reserve_frac
            required = reference_service + reference_return + scene.return_reserve_frac + self.margin
            self.references[int(uid)] = dict(
                frame_start_soc=float(plan.frame_start_battery[uid]), frame_start_xy=start.tolist(),
                frame_start_distance_m=distance, optimistic_end_xy=optimistic_end.tolist(),
                optimistic_required_soc=lower_bound, reference_end_xy=reference_end.tolist(),
                reference_distance_m=float(np.linalg.norm(reference_end-scene.airship_pos)),
                reference_service_soc=reference_service, reference_return_soc=reference_return,
                reference_required_soc=required, reference_margin_soc=self.margin,
            )

    def assess(self, uid, cause, scene=None):
        uid = int(uid)
        reference = dict(self.references.get(uid, {}))
        if cause == "final_return_infeasible":
            if not reference:
                raise RuntimeError("return failure has no frame-start reference")
            battery = reference["frame_start_soc"]
            if battery + self.tolerance < reference["optimistic_required_soc"]:
                upper, category = 1., "upper_infeasible"
            elif battery + self.tolerance >= reference["reference_required_soc"]:
                upper, category = 0., "lower_trajectory"
            else:
                upper, category = self.shared, "shared_margin"
        elif cause in {"battery_depleted_during_low_step", "return_failed_during_settlement"}:
            # Preserve the agreed upper responsibility for mid-frame depletion.
            upper, category = 1., "upper_" + cause
        else:
            raise ValueError("unknown physical failure cause")
        snapshot = dict(self.starts[uid])
        snapshot.update(reference)
        record = dict(event_id=f"frame:{self.frame_index}:uav:{uid}", uav_id=uid,
                    failure_type=cause, category=category, upper_weight=upper,
                    lower_weight=1.-upper, **snapshot)
        if scene is not None:
            actual_trip = self.energy.return_trip(scene.uav_pos[uid:uid+1], scene.airship_pos)
            record.update(actual_end_xy=scene.uav_pos[uid].astype(float).tolist(),
                          actual_end_soc=float(scene.uav_battery[uid]),
                          actual_return_required_soc=float(actual_trip.energy_frac[0])+scene.return_reserve_frac,
                          actual_return_margin_soc=float(scene.uav_battery[uid])-float(actual_trip.energy_frac[0])-scene.return_reserve_frac)
        return record


def validate_failure_records(records, dead_ids):
    ids = [r["uav_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(map(int, dead_ids)):
        raise RuntimeError("each newly dead UAV must have exactly one failure assessment")
    if any(not np.isclose(r["upper_weight"]+r["lower_weight"], 1.) for r in records):
        raise RuntimeError("failure responsibility weights must sum to one")
