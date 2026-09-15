# Two-level UAV simulation core

This rewrite starts from the complete upper-frame execution flow and adds only
the interfaces needed by that flow. It intentionally contains no SAC, DQN,
PPO, replay buffer, optimizer, or checkpoint code.

## The state and time contract

`Scene` is the only owner of persistent physical state, including hidden sensor
state. An upper frame is executed in this exact order:

1. `UpperEnv.step(upper_action)` asks the scheduler for a frozen `FramePlan`.
2. `Scene.begin_upper_frame(plan)` fixes every UAV's role for this frame.
3. `LowFrameRunner` calls `LowEnv.step()` up to ten times.
4. On the tenth low step, serving UAVs are tested for return feasibility. A
   failed requested return is also attached to this final low-step reward.
5. `Scene.complete_upper_frame()` settles return energy, charging energy, and
   positions for non-serving UAVs.
6. Full UAVs are released and waiting UAVs are promoted.
7. The next upper observation is built, so its zone occupancy is current.

The first version uses whole-frame slot reservation. Continuing charging UAVs
keep their docks, existing waiters receive vacancy priority, and new requests
use lowest-arrival-battery priority. It does not simulate a charging slot
changing hands midway through a frame. A true first-arrival scheduler can
replace `LowestArrivalBatteryScheduler` later without changing `UpperEnv`.

## Observation boundary

The temporary lower observation contains:

- active/serving UAV IDs (mapping metadata, not a learned ID feature);
- variable-length rows `[x, y, battery]` for those UAVs;
- global sensor last-visit values.

The temporary upper observation is global and contains UAV position, battery,
status one-hot, and zone occupancy. Sensor backlog and sensor positions remain
inside `Scene` and do not appear in either observation. Agent-specific upper
observations are intentionally not fixed yet.

## Backlog metrics

For every low step, arrivals are generated first and full-clear collection is
then applied. The names are explicit:

- `system_max_pre_service`: system maximum after arrivals, before collection;
- `system_max_post_service`: system maximum after collection;
- `covered_max_pre_service`: maximum over all covered sensors before clearing;
- `per_uav_owned_max_pre_service`: maximum for sensors assigned to each UAV by
  the nearest-UAV rule.

The episode timeline stores the cold-start backlog followed by the post-service
maximum and mean after every low step. Thus a full 100-frame episode has 1001
timeline samples.

## Evaluation trace

`EpisodeTrace.frames` is a list of per-frame records. Every UAV record includes
its role, end status, continuous trajectory, initial/final battery, return time
and energy, charging/waiting time, collected packets, and mean owned maximum
backlog.

The trace also accumulates:

- sensor visit count;
- packets collected from each sensor;
- a spatial any-covered-per-step grid;
- a UAV endpoint/presence grid.

The aggregate grids are designed to be saved once per evaluation episode. They
are enough to compare coverage drift across training checkpoints without saving
a large grid for every individual step. Exact step-level ownership remains in
`LowStepResult` while the episode is running.

## Current deliberate approximations

- A low-step battery failure is detected at the end of that one-minute step.
  The failed UAV is excluded from that step's collection. Event-time partial
  motion is not modeled yet.
- Return and charging happen physically during the frame but are settled at the
  frame boundary. Their time and energy are frozen in `FramePlan`.
- Waiting UAVs cannot begin charging midway through a frame.
- The upper observation and upper reward are first-version replaceable defaults.
- The energy model is the agreed horizontal-return plus fixed vertical docking
  model. It is behind an interface and can later be replaced by a lookup table
  or 3-D optimizer.

## Run

From this directory:

```bash
python run_smoke_test.py
pytest -q
```

`run_smoke_test.py` uses deterministic zero-displacement lower actions and a
simple lowest-battery upper request policy. These are smoke-test policies, not
research baselines.
