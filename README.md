# Two-level UAV simulation core

The simulation starts from the complete upper-frame execution flow. Training
is isolated in `src/jv_uav/rl`: an attention SAC lower policy and a MAPPO upper
policy share this environment without putting optimizers or replay in Scene.
See [TRAINING.md](TRAINING.md) for the confirmed training contract and commands.

## The state and time contract

`Scene` is the only owner of persistent physical state, including hidden sensor
state. An upper frame is executed in this exact order:

1. `UpperEnv.step(upper_action)` asks the scheduler for a frozen `FramePlan`.
2. `Scene.begin_upper_frame(plan)` fixes every UAV's role for this frame.
3. `LowFrameRunner` calls `LowEnv.step()` up to ten times.
4. On the tenth low step, serving UAVs are tested for return feasibility. A
   final return-safety failure is penalized only in the lower reward.
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


## Render

From the project root, using the Python environment with dependencies installed:

```powershell
python run_render_demo.py --frames 12 --output outputs/render_demo --gif
```

The demo uses deterministic diagnostic policies, not trained agents. Its fixed
map-derived waypoints do not change the RL observation contract. Outputs are
`dashboard.png`, `heatmaps.png`, `trace.json`, and optionally `episode.gif`.
The GIF shows frame boundaries, not continuous low-step flight animation.
Omit `--gif` for faster PNG-only rendering; add `--show` for Matplotlib windows.

Render any existing upper environment after reset or step:

```python
env.render(save_path="outputs/dashboard.png", show=False)
env.render(view="heatmaps", save_path="outputs/heatmaps.png", show=False)
env.render(sensor_value="last_visit", trail_frames=10, show=True)
```

The standalone `jv_uav.render_env(env, ...)` accepts the same options. Sensor
values stay in physical units; `last_visit` is displayed in minutes. UAV colors
are consistent across flight paths and battery panels. The role timeline shows
roles assigned during frames, while the fleet panel shows current status after
release/promotion. Solid lines are recorded service trajectories; dashed return
lines connect endpoints and do not imply sampled flight timing.

Rendering does not advance the environment or consume its random state.
`show=False` closes the Matplotlib figure after saving and returns the Figure
object for callers that need it. PNG, SVG and PDF paths are supported through
Matplotlib. Matplotlib is imported lazily, so simulation-only imports do not
open windows. Optional GIF export uses Pillow (a Matplotlib dependency).

Frames now include `settled` and `termination_reason`. An early terminal frame
is hatched and marked **unsettled**: non-serving battery/position is not settled,
and return/charging/waiting durations in its records remain planned values.
Death still ends the episode immediately; no partial settlement is introduced.

Coverage heatmaps retain the current recorder's geometric definition: endpoints
of UAVs serving at step start, including UAVs that fail during that step. They
must not be interpreted as successful collection. Sensor service counts and
collected-packet maps use actual service results. Frame-end battery lines are
boundary samples, not within-frame charge curves.


## Confirmed geometry and per-step evaluation fields

UAV service altitude is 100 m above ground; airship altitude is 150 m above
ground. The return climb is therefore 50 m. With a coverage angle of 60 degrees
from vertical (30 degrees elevation from the ground), the configured coverage
radius is 173.2 m. The service model, metrics and renderer share this radius.

`EpisodeTrace` additionally persists three arrays with one row per executed low
step, including steps in early terminal frames:

- `covered_max_pre_service_timeline`: maximum backlog among covered sensors,
  after current-step arrivals and before collection; zero if none are covered.
- `per_uav_owned_max_pre_service_timeline`: full-fleet rows, with column `i`
  always corresponding to integer UAV ID `i`. Values are zero when the UAV
  owns no sensors, including non-serving UAVs.
- `serving_ids_timeline`: IDs serving at step start, preserving the action-row
  mapping and distinguishing inactive UAVs from active UAVs owning no sensors.

These arrays have length T. Existing backlog max/mean timelines have length
T+1 because they include the cold-start sample. Low-step row k corresponds to
backlog row k+1 and time `(k+1) * low_step_sec`; pre-service and post-service
quantities must remain distinguished. `to_serializable()` and the render demo's
`trace.json` include these fields. Rendering keeps physical units and evaluation
metrics do not enter policy observations.


## Joint training

After activating the Torch-enabled environment:

```powershell
python train_joint.py --smoke --device cuda --output runs/smoke
python train_joint.py --device cuda --upper-steps 1000 --output runs/joint_001
```

The default schedule updates MAPPO after 512 upper frames accumulated across
100-frame episodes (or earlier death resets), using 3 epochs of 128-frame
minibatches. SAC uses 10000 waypoint warm-up steps, then one update per finalized
lower transition throughout training. Evaluation and checkpoints run every two
MAPPO updates (1024 frames). Read `TRAINING.md` for reward and entropy settings.
