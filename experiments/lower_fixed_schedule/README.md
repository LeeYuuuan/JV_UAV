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

train.jsonl logs every ten frames and episode ends. episodes.jsonl logs completed episodes. live_status.json and training_curves.png are overwritten. Evaluation saves summary.json, trace.json, dashboard.png and heatmaps.png every 1000 frames.

checkpoint.pt saves SAC, optimizers, replay, normalizer, exploration, pending transition, environment and RNG state. Resume with --resume PATH --frames NEW_TOTAL in the original output directory. Frames denotes cumulative upper frames.

Agent tests belong under UAV_Joint_Simulation/agent_tests. User runs belong under UAV_Joint_Simulation/user_runs.

