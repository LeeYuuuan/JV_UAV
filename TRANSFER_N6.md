# 六机上层 Actor 迁移训练

## 目标

从 Paper1_v1 的 `train_charging_mappo_n6_decay_4000.ipynb` 重新训练得到的六机 Actor 参数开始，在当前 JV_UAV 环境继续联合训练。只迁移上层 Actor；新 Critic、下层 SAC、优化器和经验缓存从头建立。没有冻结期，更新频率沿用当前训练配置。

## 输入输出与结构

- 旧 notebook Actor：14 → 512 → 512 → 512 → 512 → 2，隐藏层 ReLU。
- 当前迁移模式完整保留该 Actor 权重和结构，外接无可训练参数的特征/输出转换。默认普通 MAPPO 的两层 Tanh 网络不参与此次 Actor 迁移。
- Actor 特征顺序：自身 SOC(1)、自身状态 one-hot(4)、C/W 比例(2)、ID/5(1)、全部 SOC(6)。合计14维。
- 按已确认方案，旧 current SOC 替换为 current SOC 减返航所需 SOC。已在 C/W 的 UAV 不再扣返航成本，负裕量保留。SOC 单位仍为电池容量比例，不乘100、不再除容量；这改变了语义和分布，不保证复制后策略效果不变。
- 状态顺序保持 SERVING / WAITING / CHARGING / DEAD。占用数各除对应容量。ID 从0到5除以5。上层不使用 RunningMeanStd；下层 last-visit 归一化独立。
- 旧动作0=充电，1=服务；当前动作1=充电。用旧logit[0]-旧logit[1]作为Bernoulli logit，保留充电概率。
- 当前环境是240 sensors、6 UAV、2C/2W；旧调度和能量环境并非完全相同。迁移配置单独覆盖上层 reward，使用实际迁入的 Paper1 六机权重对应公式。下层 reward 沿用 JV_UAV。

## 先准备真正的六机权重

重跑旧 notebook 时确认输出 `n_uav=6, actor_obs_dim=14, critic_obs_dim=32`，并把最终显式配置的保存路径改为独立目录，例如 `checkpoints/charging/mappo_n6_decay4000_verified/ckpt.pt`，避免其他七机 notebook 再写入同一路径。
目前已有 `mappo_charging_n6/ckpt.pt` 实际15维，会被拒绝；不补虚拟 UAV，不截断参数。
将新权重复制到当前项目/服务器 `pretrained/paper1_n6_decay4000.pt`。该文件不由本项目自动生成，也不是当前已存在的合格训练权重。

## 新建迁移训练（Linux / server）

```bash
python train_joint.py \
  --env-config configs/sensors_spread_240.yaml \
  --train-config configs/training_transfer_n6.yaml \
  --upper-pretrained pretrained/paper1_n6_decay4000.pt \
  --output runs/transfer_n6_240_v1 \
  --run-note "paper1-n6-decay4000 actor warm-start; arrival SOC; joint fine-tune"
```

独立配置强制六机、512宽度和提供预训练权重；漏传权重会报错，不能静默随机初始化。
首次可添加 `--smoke --upper-steps 2` 并使用另一输出目录验证接线；smoke会缩小下层/新Critic并改变训练设置，不是正式实验。

## 继续当前实验

```bash
python train_joint.py --resume runs/transfer_n6_240_v1/checkpoint.pt --output runs/transfer_n6_240_v1
```

resume 使用完整新实验 checkpoint，不再传旧权重，也不能和 --upper-pretrained 同时使用。
resolved_config.json / provenance 会记录 Actor 架构、输入映射、源路径与SHA256。源路径仅用于溯源；恢复训练不需要旧文件。

## 验证范围

接口与训练接线可使用随机生成的六机同构权重测试，但它不能代替训练好的旧策略，不代表迁移效果或收敛性。正式对照实验等待重新训练的六机权重。

## 本次测试位置

`D:/VScode/UAV_Joint_Simulation/experiments/transfer_n6_smoke_20261112/` 保存生成脚本、明确标记的 RANDOM_UNTRAINED 源权重、run/checkpoint.pt、评估图、日志和验证报告。训练后参数更新与断点恢复已验证；不得将测试权重误当作旧收敛策略。新临时实验均放到 UAV_Joint_Simulation/experiments，最终成型代码预留在 final_code。

## 当前迁移训练节奏和奖励

`configs/training_transfer_n6.yaml` 默认运行 350 次 MAPPO 更新，每次跨 episode 累计 1024 个上层帧，共 358400 帧。每次更新 4 轮、每批 128 帧。控制台按 update 输出，并在每次更新评估、保存 checkpoint。SAC 继续按原配置训练。

上层 reward 为 `(5*S - 10*D - 0.5*min(B,2400)/800)/10`，其中 S 是这帧分配后存活的服务 UAV 数量，D 是死亡数，B 是实际执行下层步的收集前系统最大 backlog 的平均值。有效覆盖项记录在 `resolved_config.json`。旧 checkpoint 恢复时使用其保存的配置；用新配置请从旧 Actor 权重发起一次新的迁移训练。

新迁移使用 Paper1 的两类 Categorical 策略和 512 宽三层 ReLU Critic。Actor/Critic 学习率均为 1e-4，梯度裁剪分别为 0.5 和 2.0，Critic MSE 系数为 0.5。熵系数固定为 0.01。Actor 和 Critic 使用 JV 的返航后 SOC，backlog 来自 JV 实际采集。
