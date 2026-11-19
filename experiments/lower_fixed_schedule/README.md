# Fixed upper schedule, lower-only SAC

This experiment replays all 100 frames of one saved Paper1 role schedule. It creates and updates only lower SAC networks. Exact serving/waiting/charging roles replace the upper request policy and allocation ranking.

Source is the gamma=0.99, GAE lambda=0.95, decaying-entropy experiment, episode 12050 at update 900. The source training episode has exactly 350 serving UAV-frames. Every JV episode restarts this same role sequence. UAV energy, return time and charging remain physically simulated.

## Feasibility

Paper1 uses 0.376933 SOC per serving frame with zero return cost. JV's minimum service cost per frame is 0.388667 because every 60-second step has at most 30 seconds at 257 W and the remaining time at 326 W.

An optimistic calculation with zero return energy/time, minimum service energy every step, and full 500 W charging on every charging frame exhausts U4/U5 in frame 20. This exact schedule cannot finish 100 frames under current JV energy settings.

Physical stationary replay terminates in frame 5. Alternating maximum movement near the airship terminates in frame 8. These are feasibility checks. The user selected actual JV energy. Training keeps this exact schedule and records its physical failures. Schedule energy infeasibility limits interpretation of collection-learning progress.

## Run

From JV_UAV, run python experiments/lower_fixed_schedule/train.py --device cuda --frames 100000 --reward topk --output ../UAV_Joint_Simulation/user_runs/lower_fixed_schedule_topk_v1

This uses the physical schedule above. Early deaths are expected with the current energy mismatch. Use --reward legacy and a separate output directory for an identical-schedule reward baseline.

Default data reward is (-B_after + 0.5*B_after*h_5)/300. K=5, rho=0.5, offset=0 and scale=300 are CLI parameters. They replace the old p-norm and quadratic backlog components. Existing boundary and final-return penalties remain. Lower observations, SAC, normalization, replay and waypoint warmup reuse JV code. There is no upper actor, critic or optimizer.

train.jsonl logs every ten frames and episode ends. episodes.jsonl logs completed episodes. live_status.json and training_curves.png are overwritten. Evaluation saves summary.json, trace.json, dashboard.png and heatmaps.png every 100 upper frames (1000 lower steps when frames complete).

checkpoint.pt saves SAC, optimizers, replay, normalizer, exploration, pending transition, environment and RNG state. Resume with --resume PATH --frames NEW_TOTAL in the original output directory. Frames denotes cumulative upper frames.

Agent tests belong under UAV_Joint_Simulation/agent_tests. User runs belong under UAV_Joint_Simulation/user_runs.




## Terminal future backlog cost

Fresh top-K runs add the discounted remaining-horizon data cost exactly once on any physical failure, including mid-step battery exhaustion, final-return failure and non-serving return settlement failure. The actual episode and physics still stop immediately.

The virtual future assumes no further collection and deterministic mean arrivals for each sensor. It uses the current full backlog vector and the lower SAC gamma. Future top-K bonus and future control penalties are zero. Future offset is retained. This is an explicit failure convention, not an exact estimate of the expected stochastic maximum.

For remaining steps k=1..H, the terminal addition is sum(gamma**k * (offset - max_i(B_i + arrival_rate_i * low_step_sec * k)) / scale). Successful horizon completion adds zero. Replay and lower diagnostics receive the adjusted final transition before training.

Logs include terminal_future_backlog under lower reward components. No schedule-phase input was added. Existing checkpoints preserve their old reward semantics; start a fresh run to enable this reward because existing replay contains old terminal labels. Use --no-terminal-tail for a fresh-run ablation.

The source average service power was 282.7 W, with 257 W horizontal and 326 W hover applied over whole lower steps. JV limits movement to 30 out of 60 seconds per step, giving minimum service average power 291.5 W. Source energy is 0.376933 SOC/frame, JV minimum 0.388667. An optimistic per-UAV audit, ignoring all return cost and using full charging, first exhausts U4/U5 at frame 20. It does not exhaust every UAV at frame 20.
