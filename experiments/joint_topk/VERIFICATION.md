# Joint implementation verification, 2026-11-18

- All thirteen files from the previous implementation were present and byte-identical to the tested patch. No previous write approval remains necessary for that implementation.
- Broad CPU verification: 85 tests passed. Four older test mocks were updated for the existing `include_probabilities` interface. Two additional checks cover frozen joint Top-K evaluation and final-return failure costs in terminal replay.
- Checked GAE reset masks and partial-rollout bootstrap, 1024-frame accumulation across resets, lower pending transitions after upper allocation, dense-to-matched update counts, exact CPU checkpoint continuation, reward term accounting, and training/evaluation isolation.
- Ran the actual default configuration with the full network sizes for three training frames, followed by complete early-termination evaluations on all three seeds. Checkpoint, scalar logs, full traces and rendering completed. Initial untrained deaths are expected in this execution check; this is not a convergence result.
- No learning, reward or physical configuration changes were made during this recheck. Upper gamma=.95, GAE lambda=.95, entropy=.001; lower SAC gamma=.99; the SAC stage threshold is 250000 optimizer updates.
- CUDA execution and long-run convergence were not tested in this CPU review.
- Full verification output is in the local research workspace `joint_review_20261118/`; temporary tests did not start a persistent training process.
