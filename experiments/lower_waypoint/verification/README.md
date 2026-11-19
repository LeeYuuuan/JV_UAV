Verification only; not trained-policy convergence results.

tests.json records physical, numerical, SAC target, evaluation, and exact-resume checks.
objective_sanity.json compares three untrained reference controls on the same map and 8-hour physical horizon.
center: no collection; random: uniform whole-map waypoints; greedy_reference: current-backlog^4 weighted footprint selection with overlap removed.
All references are evaluation-only and never enter SAC replay. Reward includes the explicit time-integrated backlog and Top-K terms described in the parent README.
