# Continuing small-map SAC experiment

1 UAV, 6 sensors on a radius140m hexagon, 600m map, 40m coverage, real movement
limit180m per60s. No energy, return, rotation or finite task horizon.
Reward = -(post-max backlog /300)*(1-.8*h_top18) - movement. Top18 uses all6
sensors here; it is selected before collection after arrivals, with union clearing.
Gamma=.95; active-status input, GELU and SAC learning parameters retained.
Exploration is unchanged: epsilon1->.1 over100k; inside branch25%random,
50%oldest-sensor greedy(candidate_count1),25%sampled actor; outside actor.

Use train_continuing.py. Every15 physical steps produces a statistics window;
only diagnostic buffers are cleared, not scene/RNG/positions/queues/ages. All
replay transitions remain nonterminal at windows and training-budget end.
Checkpoint captures the scene, partial window, replay, optimizer and RNG states.
Start fresh; finite-horizon checkpoints are explicitly rejected.
Initial training budget60k steps. Save/eval every1000. Pure actor eval is one
continuous30-step rollout per seed, with two images: eval_steps_01_15.png and
eval_steps_16_30.png. Evaluation cutoffs never enter training replay.
windows.jsonl.gz and training_curves.png report per-step reward and backlog.
