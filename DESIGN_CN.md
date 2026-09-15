# 两层 UAV 仿真核心：第一版规范

## 1. 最终执行入口

整个 episode 的控制流只有一条：

```text
JointEpisodeRunner
  -> upper_policy(upper_obs)
  -> UpperEnv.step(upper_action)
       -> scheduler.make_plan(...)
       -> Scene.begin_upper_frame(...)
       -> LowFrameRunner.run_frame(...)
            -> low_policy(low_obs)
            -> LowEnv.step(low_action) × 最多 10
                 -> Scene.evolve_low_step(...)
       -> Scene.complete_upper_frame(...)
       -> release / promote
       -> next_upper_obs
```

Agent、replay buffer 和训练更新都不属于 Scene。当前两个 heuristic policy 只用于确认整条流程能运行。

## 2. Scene 的边界

Scene 保存唯一一份真实状态：

- 可观测候选：UAV 位置、电量、status、sensor last visit；
- 隐藏状态：sensor 位置、真实 backlog、随机 arrival 状态；
- 时钟：`now_sec`、`upper_step`、`low_step_total`、`step_in_frame`。

三个演化方法分别代表三个时间边界：

- `begin_upper_frame(plan)`：冻结这个 frame 的 SERVING / WAITING / CHARGING；
- `evolve_low_step(action, ...)`：演化一分钟；
- `complete_upper_frame(plan, ...)`：10 分钟结束时结算返航、充电、release 和 promote。

因此没有采用容易混淆的 `upper_update()`。如果以后做 event-driven，`complete_upper_frame` 可以替换为事件推进器，而低层接口不需要全部推倒。

## 3. 上层 action 和 slot 分配

第一版 action 是每架 UAV 一个二元请求：

- `0 = SERVE`
- `1 = REQUEST_CHARGE`

环境最终分配才是 `SERVING / WAITING / CHARGING / DEAD`。所有请求者先计算：

```text
arrival_battery = current_battery - return_energy
```

跨 frame 已占用的 zone 必须有实际意义，因此先遵守 occupancy：

1. 继续请求的现有 CHARGING 保留 charging slot；
2. 空出的 charging slot 优先给继续请求的现有 WAITING；
3. 继续请求的现有 WAITING 优先保留 waiting slot；
4. 剩余新请求者才按 `arrival_battery` 从低到高分配空位；
5. 没有位置的请求者继续 SERVING，不发生返航。

同一优先级内由 UAV ID 打破平局。这个规则确定、容易测试，也延续原模型逻辑。真正的“先到先充”需要 frame 内 slot handoff；否则距离近的 UAV 虽然先到，也只能空占整个 frame，无法获得所设想的效率收益。因此该机制留给后续事件版本。

## 4. 一个 upper frame 的精确时序

1. 上层 observation 使用上一个 frame 完成 release/promote 后的 zone 状态。
2. 上层给出 charging request。
3. Scheduler 生成不可变的 `FramePlan`。
4. Scene 固定本 frame 各 UAV 的角色。
5. Serving UAV 执行最多 10 个 low steps；sensor arrival 每步都发生。
6. 任一 serving UAV 在 low step 后电量耗尽，则 episode 在该 low-step 边界终止。
7. 第 10 步后检查所有 serving UAV 是否保有返航电量；不足等同 DEAD。
8. 已接受 charging/waiting 但返航失败的 UAV，也在第 10 个 low reward 中收到 death penalty。
9. 成功返航者扣 return energy、位置设为 airship；charging UAV 再增加 charging energy。
10. 充满者 release，waiting 中低电量者补入空出的 charging slot。
11. 记录本 frame 指标，再生成 next upper observation。

其中第 6 步目前是在一分钟离散边界检测，不模拟“第 37 秒耗尽后只飞一部分”的连续事件。这是明确保留的第一版近似。

## 5. 一个 low step 的精确时序

1. 读取当前 serving IDs，attention policy 输出形状 `(N_serving, 2)` 的位移；
2. 位移按向量范数限制到 `speed × max_move_time = 180 m`；
3. 目标点超地图时裁到边界，同时记录每架 UAV 的 OOB；
4. 根据实际执行距离计算 move time；
5. `service_time = 60 s - move_time`；
6. 扣除 `move_power × move_time + hover_power × service_time`；
7. 所有 sensor 产生一分钟 Poisson arrivals；
8. 在覆盖范围内按最近 UAV 确定唯一 owner；
9. full-clear：有 owner 的 backlog 清零，其他保留；
10. 更新 last visit、reward、评估指标和时钟。

## 6. backlog 口径

为了不再混用 `max_before/max_after`，同时保留如下明确字段：

- `system_max_pre_service`：本步 arrivals 后、收集前的系统最大 backlog；
- `system_max_post_service`：本步收集后的系统最大 backlog；
- `covered_max_pre_service`：本步所有被覆盖 sensor 在清空前的最大值；
- `per_uav_owned_max_pre_service[i]`：按最近 UAV 归属后，UAV i 负责的 sensor 中清空前最大值。

下层 reward 当前严格按提出的结构：

```text
small_weight * sum(per_uav_owned_max_pre_service)
- system_max_post_service
- OOB penalty per UAV
- final return/death penalty
```

这些权重都在 YAML 中，之后可改而不触碰状态演化。

## 7. eval 数据

每个 episode 的 `EpisodeTrace` 保存：

- backlog 最大值时间序列：冷启动后初值 + 每个 low step 收集后值；
- backlog 平均值时间序列：同样的时间点；
- 每个 sensor 被覆盖次数；
- 每个 sensor 被实际收走的 packet 总量；
- 空间 coverage grid：每步某像素被任意 UAV 覆盖则 `+1`；
- UAV presence grid：每个 UAV 每步终点所在像素 `+1`；
- `frames: list[FrameRecord]`。

每个 frame 下每架 UAV 的 record 保存：角色、frame 末 status、连续轨迹、起止电量、返航时间/能耗、charging/waiting 时间、收集总量和该 UAV 的平均 owned-max。

聚合 heatmap 按 episode 保存即可。逐 step 的 owner 和 backlog 数组在 `LowStepResult` 中可供在线分析，但第一版不自动全部写盘，避免训练时产生巨大文件。

## 8. 仍然开放的接口

以下内容尚未被当前设计锁死：

- 上层是 centralized DQN、MAPPO 还是其他 multi-agent observation；
- attention 网络的 padding/mask 和 replay transition 格式；
- upper reward 的最终权重与是否使用多个 frame 的责任窗口；
- rate-based service / TDMA / SINR；
- first-arrival 的 frame 内事件调度；
- 连续时间电量耗尽；
- 3-D 最优返航或 lookup-table energy model；
- episode trace 的最终 NPZ/JSON 文件格式。

这些部分都通过 policy/model/scheduler/recorder 边界与当前核心隔开，可以在确认后逐个加入。
