# Codex Continuation Guide: Hierarchical Multi-UAV Simulation

## 1. Purpose of this file

Continue developing and reviewing the hierarchical two-level RL simulation in
this folder. Do not restart the architecture and do not perform a broad rewrite.
The user is reviewing the code incrementally and wants to confirm the simulation
semantics before any RL implementation is added.

The user communicates mainly in Chinese. Keep explanations direct and inspect
one logical component at a time. Source-code comments and docstrings must be in
English only.

Before changing code:

1. Read this file completely.
2. Read `Note.md` completely if it is present in the imported folder.
3. Inspect the actual current source files; do not assume they exactly match an
   earlier archive or this handoff.
4. Run the existing tests or equivalent smoke checks.
5. Explain any semantic inconsistency before changing it.

The user-authored `Note.md` documents portions already reviewed and understood.
Treat those portions as confirmed unless a newer decision in this file overrides
them.

## 2. Research objective

The simulator models multiple UAVs collecting data from spatial sensors while
an airship provides charging docks and waiting positions.

The intended hierarchy is:

- Upper level: decides whether each UAV continues serving or requests charging.
  It is responsible for allocating UAVs across multiple upper frames so that a
  UAV does not exhaust its battery during a frame.
- Lower level: controls trajectories for the currently serving UAVs. It is
  responsible for ensuring that, after the final low step of a frame, a
  surviving UAV still has enough energy to return to the airship.

The primary system objective is to minimize the time-average maximum sensor
backlog while avoiding UAV death.

## 3. Time model

Current synchronous design:

- One low-level step is 60 seconds.
- One upper frame contains 10 low-level steps.
- One upper frame is therefore 600 seconds.
- A low step allows at most 30 seconds of horizontal movement at 6 m/s.
- Maximum physical displacement per low step is therefore 180 m.
- Remaining low-step time is service time.

Do not convert this version to event-driven simulation unless the user explicitly
requests it. Event-driven charging/arrival scheduling is a possible later
extension, not part of the currently reviewed core.

## 4. Architectural contract

`Scene` is the single source of truth for persistent physical state:

- sensor positions;
- sensor backlog;
- sensor last-visit time;
- UAV positions;
- UAV batteries;
- UAV statuses;
- simulation clocks and step counters.

The intended call path is:

```text
JointEpisodeRunner
  -> UpperEnv.step(upper_action)
       -> charging scheduler creates FramePlan
       -> Scene.begin_upper_frame(plan)
       -> LowFrameRunner.run_frame(plan)
            -> low policy(observation)
            -> LowEnv.step(normalized_action)
                 -> Scene.evolve_low_step(action_in_meters)
       -> Scene.complete_upper_frame(plan)
       -> release/promote UAVs
       -> next upper observation
```

Policies, neural networks, replay buffers, optimizers, and training updates must
not be placed inside `Scene`.

## 5. Confirmed lower-level action contract

This contract must remain consistent when attention SAC is implemented.

Suppose:

```python
active_uav_ids = np.array([0, 2, 5])
```

The policy action has shape `(3, 2)`, and the row mapping is:

```text
action[0] -> UAV 0
action[1] -> UAV 2
action[2] -> UAV 5
```

`active_uav_ids` contains integer UAV IDs. It is mapping metadata and should not
be directly encoded as one-hot policy input. The lower attention network uses
the corresponding UAV features, currently `[x, y, battery]`.

The policy produces normalized actions in `[-1, 1]`. `LowEnv.step()` performs:

```python
normalized = np.asarray(active_actions, dtype=np.float32)
normalized = np.clip(normalized, -1.0, 1.0)
actions = normalized * self.scene.max_move_distance_m
result = self.scene.evolve_low_step(
    actions,
    self.energy_model,
    self.service_model,
)
```

Inside `Scene.evolve_low_step()`, `actions` is already measured in meters. The
user prefers the simple name `actions`; do not introduce several verbose names
for the same array merely to restate its unit.

The physical action still needs radial norm clipping. Component-wise mapping of
`[1, 1]` gives `[180, 180]`, whose norm exceeds 180 m. Radial clipping preserves
direction while ensuring the total displacement does not exceed 180 m.

`requested_displacement` means the displacement after radial distance limiting.
`executed_displacement` means the actual displacement after additionally
clipping the endpoint to the map boundary.

## 6. Confirmed low-step order

The reviewed low-step transition is:

1. Obtain integer `serving_ids`.
2. Validate the action shape `(len(serving_ids), 2)`.
3. Radially limit action norm.
4. Compute intended endpoints.
5. Clip endpoints to the map and record out-of-bounds information.
6. Update UAV positions.
7. Compute actual distance, movement time, and service time.
8. Deduct movement and hover/service energy.
9. Mark UAVs whose battery is depleted during the low step.
10. Independently sample Poisson arrivals for every sensor.
11. Add arrivals to sensor backlog and increment last-visit time.
12. Determine which surviving serving UAVs can collect data.
13. Apply the current service model.
14. Update backlog and last-visit values for served sensors.
15. Build `LowStepResult` and advance the clocks.

The current Poisson call uses a vector of per-sensor means:

```python
self.rng.poisson(self.arrival_rate * self.low_step_sec)
```

Therefore sensors share the same configured mean by default but receive
independent random arrival samples.

`pre_service` already includes arrivals generated during the current step. Its
meaning is: backlog after this step's arrivals, immediately before collection.
The user knows this convention and does not currently want it renamed.

Use the words "service" or "collection" in explanations. Do not casually call
the general process "clearing" because full-clear is only one special service
model.

## 7. Current service model

`FullClearNearestService` is the confirmed first-version service model.

For `N` sensors and `M` serving UAVs, it constructs an `N x M` pairwise-distance
matrix using NumPy broadcasting. Each covered sensor is assigned to its nearest
serving UAV. An uncovered sensor has owner `-1`.

Confirmed result meanings:

- `sensor_owner`: UAV owner ID for every sensor, or `-1`.
- `collected_per_sensor`: collected amount for every sensor.
- `collected_by_uav`: total amount collected by every UAV.
- `per_uav_owned_max_pre_service`: maximum pre-service backlog among sensors
  assigned to each UAV.

Keep the service model replaceable. A later version will likely introduce a
rate-based communication model; do not embed full-clear assumptions throughout
the rest of the simulator.

## 8. `LowStepResult` semantics

`LowStepResult` is a per-step execution report. It is not an observation and is
not persistent scene state. It supports reward calculation, termination,
trajectory/energy/backlog metrics, replay construction, evaluation, and plots.

Important confirmed meanings:

- `step_in_frame`: index of this low step inside its upper frame.
- `serving_ids`: integer IDs corresponding to action rows at step start.
- `position_before`, `position_after`: positions of all UAVs.
- `requested_displacement`: full-UAV displacement requested after radial limit.
- `executed_displacement`: full-UAV displacement actually executed after map
  boundary handling.
- `move_distance_m`, `move_time_sec`, `service_time_sec`: full-UAV arrays.
- `energy_used_frac`: battery fractions consumed in this step.
- `arrivals`: independently sampled sensor arrivals.
- `packets_pre_service`: backlog after arrivals and before collection.
- `packets_post_service`: backlog after collection.
- `covered_max_pre_service`: maximum pre-service backlog among covered sensors.
- `system_max_pre_service`: system-wide maximum before service.
- `system_max_post_service`: system-wide maximum after service.
- `system_mean_post_service`: system-wide mean after service.
- `oob_mask`: whether each UAV's requested endpoint crossed the map boundary.
- `oob_overflow_distance_m`: distance beyond the map for each requested endpoint.
- `dead_during_step`: UAV IDs that depleted their battery during the frame.
- `return_unsafe_ids`: serving UAV IDs that survive the final low step but lack
  enough energy to return to the airship.
- `reward`, `reward_terms`: calculated lower reward and its components.
- `normalized_action`: optional full-UAV normalized action record if retained by
  the actual implementation.

Keep field names consistent with the actual dataclass. Do not rename fields in
isolation. If the imported folder uses `_m` suffixes, first inspect all callers
and discuss the naming choice with the user.

## 9. Confirmed lower reward and responsibility split

The user has confirmed this responsibility assignment:

```text
Battery depleted during one of the 10 low steps
    -> upper-level responsibility

Alive after the final low step but unable to return to the airship
    -> lower-level responsibility
```

Therefore `LowReward` must not penalize `result.dead_during_step`.

The currently confirmed lower reward structure is:

```python
covered_term = covered_weight * result.per_uav_owned_max_pre_service.sum()
system_term = -system_weight * result.system_max_post_service
oob_term = -oob_weight * float(result.oob_mask.sum())
return_failure_term = -death_weight * final_return_failure_count
reward = covered_term + system_term + oob_term + return_failure_term
```

`covered_weight` is divided by the configured total UAV count during
initialization so that the positive reward scale does not automatically grow
with fleet size:

```python
self.covered_weight = (
    float(reward["low_covered_max_sum_weight"])
    / float(uavs["count"])
)
```

The confirmed final-return penalty method is:

```python
def apply_final_return_penalty(
    self,
    result: LowStepResult,
    unsafe_serving_ids: np.ndarray,
) -> None:
    result.return_unsafe_ids = unsafe_serving_ids.copy()
    result.reward, result.reward_terms = self.reward_model(
        result,
        final_return_failure_count=len(unsafe_serving_ids),
    )
```

This recomputes the final low-step reward with the return penalty included.

### Important override to `Note.md`

Do not use `deferred_return_failure_ids` in lower reward calculation. Do not pass
`plan.return_failure_mask` into `LowEnv.apply_final_return_penalty()`.

`return_failure_mask` may temporarily remain in `FramePlan` for physical
settlement or diagnostics, but its need and semantics must be reviewed with the
user when reviewing the upper scheduler. It must not cause a second lower
penalty.

## 10. Confirmed `LowFrameRunner` behavior

The user prefers keeping the final-step check inside the loop:

```python
for index in range(total):
    obs = self.low_env.observation()
    action = self.low_policy(obs)
    result = self.low_env.step(action)
    steps.append(result)

    if result.dead_during_step.size:
        reason = "battery_depleted_during_low_step"
        break

    if index == total - 1:
        unsafe = self.scene.mark_unable_to_return_dead(
            self.energy_model
        )
        self.low_env.apply_final_return_penalty(
            result,
            unsafe,
        )
        if unsafe.size:
            reason = "cannot_return_to_airship"
```

This has the desired behavior:

- a mid-frame battery depletion terminates the low-frame loop immediately;
- return feasibility is checked only after reaching the tenth low step;
- the return penalty is attached only to the final lower result.

## 11. Upper-level code is not yet confirmed

Do not assume the current `FramePlan`, scheduler, `begin_upper_frame()`,
`complete_upper_frame()`, `UpperReward`, or `UpperEnv.step()` is final. These are
the next main review targets.

The intended high-level timeline still needs careful confirmation:

1. Observe the scene and zone occupancy.
2. Obtain upper charging requests.
3. Resolve actual serving/charging/waiting roles.
4. Run lower trajectory control for serving UAVs.
5. Account for return travel and charging during the frame.
6. At frame end, release fully charged UAVs and promote waiting UAVs.
7. Generate the next upper observation after release/promotion.

The user specifically wants release/promotion at the frame end so the next
upper action observes correct charging/waiting occupancy.

Do not finalize upper death penalties without user confirmation. The confirmed
principle is that `dead_during_step` belongs to upper-level responsibility,
whereas `return_unsafe_ids` belongs to lower-level responsibility. Episode
termination can still occur for either event.

## 12. Evaluation data requirements

The intended evaluation record includes:

- post-service system maximum backlog after every low step;
- post-service system mean backlog after every low step;
- the cold-start backlog value before the first low step;
- each UAV's per-step owned maximum pre-service backlog;
- overall covered maximum pre-service backlog;
- per-frame serving count;
- dead UAV IDs and failure cause;
- UAV trajectories;
- battery start/end values;
- return time and return energy;
- charging and waiting duration;
- packets collected by UAV and by sensor;
- sensor visit-count heatmap;
- sensor collected-packet heatmap;
- spatial coverage heatmap;
- UAV endpoint/presence heatmap.

Per-frame UAV records should preserve role during the frame separately from
status after frame-end release/promotion.

## 13. Future lower-level RL contract

Do not implement this until the environment core is confirmed.

The planned lower policy is a centralized variable-length attention SAC:

- one shared actor/critic handles a variable number of serving UAVs;
- per-UAV token: `[x, y, battery]`;
- global input includes sensor last-visit values;
- sensor backlog is not part of the policy observation;
- non-serving UAVs do not produce actions;
- integer active UAV IDs map action rows back to real UAVs;
- active UAV IDs are not policy features;
- padded batches must include a valid-token mask;
- replay entries must preserve the observation/action row mapping and masks;
- actor output is normalized to `[-1, 1]` before physical action conversion;
- critic/replay design must correctly handle changes in active UAV count between
  current and next states.

Do not silently reuse a fixed-UAV replay format.

## 14. Future upper-level RL

The final algorithm and observation are not yet confirmed. Possible methods
include multi-agent DQN or MAPPO. Do not lock the simulator to either one.

The simulator should expose a stable environment interface first. Agent-specific
upper observations, identity features, action semantics, transition storage,
and reward weights must be reviewed with the user before implementation.

## 15. Immediate continuation tasks

When Codex opens the user's current folder, proceed in this order:

1. Compare actual `LowReward`, `LowEnv.apply_final_return_penalty()`, and
   `LowFrameRunner.run_frame()` against Sections 9 and 10.
2. Make only the small consistency patch needed to remove the obsolete
   `deferred_return_failure_ids` lower-reward path.
3. Verify normalized action mapping and radial norm clipping with a unit test.
4. Verify integer action-row mapping, e.g. active IDs `[0, 2, 5]`.
5. Verify independent per-sensor Poisson arrivals.
6. Verify nearest-UAV ownership and service accounting.
7. Verify mid-frame depletion stops remaining low steps without adding a lower
   death penalty.
8. Verify final-step return infeasibility adds a lower penalty.
9. Then review `FramePlan` and the charging scheduler with the user, one section
   at a time.
10. Review `Scene.begin_upper_frame()` and `Scene.complete_upper_frame()`.
11. Review `UpperReward` and `UpperEnv.step()`; do not infer final semantics from
    old code.
12. Review metrics/trace persistence and plotting inputs.
13. Run deterministic heuristic rollouts before implementing RL.
14. Only after the simulation is confirmed, implement attention SAC and the
    selected upper-level agent.

## 16. Working style requested by the user

- Do not invent problems merely to criticize the design.
- When a real inconsistency exists, identify it directly and explain its effect.
- Avoid adding classes, fields, or verbose variable names without a concrete
  need.
- Prefer simple names such as `actions` when context already establishes units.
- Keep source comments in English only.
- Use service/collection terminology; reserve "full-clear" for the specific
  first-version service model.
- Preserve already confirmed behavior.
- Do not make broad changes while the user is reviewing one function.
- When asked a simple question, answer briefly.
- Before writing RL, verify the full environment using deterministic policies
  and explicit invariants.

## 17. Open questions that require user confirmation

- Exact upper action/state-transition semantics for existing charging and
  waiting UAVs.
- Whether scheduler priority is existing occupancy, lowest arrival battery,
  first arrival, or a later event-driven rule.
- Exact need and meaning of `FramePlan.return_failure_mask` after final-step
  return-safety enforcement.
- Timing of return-energy deduction and charging-energy addition.
- Whether a UAV that fills its battery mid-frame remains occupying its dock
  until frame end.
- Final upper reward and precise responsibility window.
- Final global versus per-agent upper observation.
- Exact episode length for the new large-map setup.
- Rate-based service model and communication-resource allocation.
- Final trace serialization format and output directory layout.

Do not resolve these questions silently. Present the smallest relevant choice
when its code section is reached.
