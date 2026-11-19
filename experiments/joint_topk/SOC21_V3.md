# Actor21: dedicated own return-energy feature

Fresh-run default: configs/training_joint_topk_soc21_v3.yaml.
Actor input order: own current SOC(1), own status onehot(4), C/W occupancy(2),
normalized ID(1), all current SOC(6), all return-energy SOC(6), own return SOC(1).
The last feature is selected exactly by the current UAV identity. Existing20
columns retain their order. Actor21 features, critic38, raw actor observation24.

No other experiment settings changed from v2: upper gamma .95/lambda .95,
constant entropy .01, lower Top-K20, physical energy, seeds, rewards, staged
updates and evaluation/logging remain the same. Credit diagnostics are enabled.

Legacy14/20input checkpoint loading is preserved. Resuming an old checkpoint
keeps its saved input dimension. Use a fresh run to use21inputs. Optional old
Paper1 actor import adds seven zero columns to the original14-feature input;
it preserves the original current-SOC policy initially.

94 CPU tests and the CLI migration smoke passed, including exact self-feature
selection under reordered actor rows, old transfer probability equivalence,
legacy20shape,21shape and exact resume. These are execution checks, not a
convergence experiment.

After local commit/push, run on the server:

```bash
cd ~/Code/JV_UAV
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=1 python3 -u train_joint.py --train-config configs/training_joint_topk_soc21_v3.yaml --device cuda --output ./runs/joint_soc21_top20_v3
```
