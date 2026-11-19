# 代码逐段 Review 进度

更新时间：2026-10-05

## 记录规则（用户确认的流程）

- 以函数/方法为最小验收单位；长函数按代码段记录，所有段确认后再标为函数已确认。
- class 只用于分组，不因某个方法确认就将整个 class 标为已确认。
- 用户说“继续”，代表认可上一轮已讲解的本体代码范围；不是形式化正确性证明。
- 本体确认不传递给外部调用、返回对象或属性实现。外部函数需明确展开审查，并得到用户确认后单独登记。
- 简单外部函数可在主线中顺便审查，但必须清楚说明切换到哪个函数。
- 区分：未讲解 / 已讲解待确认 / 本体已确认但依赖未验收 / 含依赖已确认。
- 旧 Note.md、历史测试通过、助手自行读过源码，均不自动等价于本轮用户确认。
- 后续修改了已确认代码，应将受影响函数/代码段标为待复核，并记录变更；无关部分不重复审查。
- 每次推进后更新本文件。运行代码不因记录进度而自动修改。

## 当前断点

- 用户明确 Scene 文件本体已读完；外部依赖与 plan 字段来源尚未完全验收。
- Scene 帧末低电量晋升已理解，但与用户期望 FIFO 有差异；记录为后续改动，不视为该规则符合目标。
- 四个状态 ID helper 已确认。
- FramePlan 字段定义及 serving_ids 属性已由用户认可；字段生成正确性仍未验收。
- 按用户新要求改为调用链驱动审查：从一个完整 frame 的 UpperEnv.step 入手，不先展开训练循环。
- UpperEnv.step 的 make_plan → begin_upper_frame → run_frame → 条件结算本体已由用户确认；后半段 reward/记录/终止尚未逐段审查。
- scheduler 构造、make_plan 入口检查、返航估算及 low_battery_order 本体已确认（低电量规则与 FIFO 目标的差异仍待改）；assigned / ~alive 补充解释已随用户继续确认。
- charging 保留、remaining 请求掩码、waiting 优先候选和新 charging 分配本体已确认；低电量规则与 FIFO 目标差异仍待处理。
- make_plan 的 waiting 位置保留/分配、写入 assigned 本体已确认；retained_waiting 多余排序/截断仍未修改。
- 本轮：returning 掩码、成本归零、arrival SOC 重算、失败条件、C/W 时长、FramePlan 快照打包已讲解待确认。
- 下一段：返回 UpperEnv.step 调用链，完整审查 LowFrameRunner.run_frame；能量模型仍待验收。
- rl/observations.py 的转换函数仍暂缓；能量模型尚未审查。

## Scene 主线

文件：`src/jv_uav/scene.py`

| 方法/代码段 | 状态 | 已确认范围 / 尚未确认部分 |
| --- | --- | --- |
| `__init__`：读取参数、创建属性 | 本体已确认，依赖未验收 | 参数引用、物理量与数组维度；不代表配置校验与地图生成已验收 |
| `__init__`：RNG、状态数组、调用 reset | 本体已确认，依赖见下表 | 已说明重复创建 RNG 不改变当前初始化采样 |
| `reset` | 本体已确认 | reset_rng 两种行为、冷启动、初始位置/SOC/状态、时钟；训练器如何调用尚未审查 |
| `observe_lower` | 本体已确认，依赖未验收 | serving 行映射、原始 x/y/SOC、全部 last-visit 的副本；serving_ids helper 已确认 |
| `observe_upper` | 本体已确认，依赖未验收 | 原始位置/SOC/one-hot、C/W 数量；不等同于 MAPPO 输入；占用 helper 已确认 |
| `begin_upper_frame` | 本体已确认，依赖未验收 | frame 起点检查、仅对 alive UAV 应用 assigned_status；计划如何生成尚未审查 |
| `evolve_low_step`：动作形状、位置副本、径向步长限制 | 本体已确认 | 动作单位米；尚未验收 LowEnv 的上游动作转换 |
| `evolve_low_step`：终点边界处理、位移记录、距离/时长 | 本体已确认 | 坐标截断终点而非沿原方向碰边；按执行距离计时 |
| `evolve_low_step`：耗电调用、死亡标记 | 本体已确认，能量依赖未验收 | 整步结算；先记录终点再判断耗尽；死亡 UAV 不收集 |
| `evolve_low_step`：Poisson arrivals、服务资格、pre_service | 本体已确认 | 本步 arrivals 可被本步收集；其他存活 UAV 完成本步 |
| `evolve_low_step`：collect 调用、backlog/last-visit 更新 | 本体已确认，收集模型待确认 | 接受当前 full-clear 语义；后续解耦见 REFACTOR_NOTES.md |
| `evolve_low_step`：越界展开、时钟、LowStepResult、返回 | 本体已确认，结果类型待系统审查 | 记录索引 0–9；返回前 step_in_frame 加一；reward 在外层计算 |
| `mark_unable_to_return_dead` | 本体已确认，依赖未验收 | 用户本轮“继续”确认；return_trip 和 runner 调用时机仍待审查；serving_ids 已确认 |
| `complete_upper_frame`：入口、返航成功/失败结算 | 本体已确认，依赖未验收 | 到 successful UAV 位置更新为 airship；计划字段计算与外层调用条件未验收 |
| `complete_upper_frame`：充电、释放/候补、计数器与返回 | 本体已审查；FIFO 差异待改、外部依赖待验收 | 充电上限、释放阈值、按电量/ID晋升、晋升不追补本帧充电；charging_energy_fraction 和 plan.charging_time_sec 待验收 |
| `serving_ids` / `waiting_ids` / `charging_ids` / `dead_ids` | 已确认 | 按状态过滤并返回递增 int64 ID 数组，无状态修改 |

## 外部依赖：不继承调用方的确认状态

| 文件 / 函数或类型 | 状态 | 说明 |
| --- | --- | --- |
| `config.py`：section / validate_config / load_config | 暂缓，未验收 | 早前做过概览，用户要求优先读核心代码 |
| `sensor_generator.py`：generate_sensor_positions 及内部函数 | 未逐函数审查 | 仅说明调用时机和输出形状 |
| `models.py`：SimpleEnergyModel 的各方法 | 未逐函数审查 | 包括 serving_energy_fraction / return_trip / charging_energy_fraction / _joules_to_fraction |
| `models.py`：FullClearNearestService.collect | 已讲解部分核心代码，未单独确认 | 上一轮展示了距离、最近归属和收集量；未按用户要求单独确认外部函数，不标为通过；后续补全并确认 |
| `models.py`：LowestArrivalBatteryScheduler.__init__ / make_plan 前半段 | 本体已确认，assigned 补充解释待确认 | 构造、动作检查、返航估算、low_battery_order；能量依赖未验收，FIFO 差异待改 |
| `models.py`：make_plan charging 分配 | 本体已确认；FIFO 差异待改 | retained_charging、remaining、waiting/other 候选、new_charging、charging_ids |
| `models.py`：make_plan waiting 分配和 assigned 写入 | 本体已确认，已知冗余待处理 | 保留已有 waiter、分配剩余 waiting 位、未接纳请求保持 serving；retained_waiting 排序/截断存在与原 retained_charging 相同的冗余和异常处理问题，未修改 |
| `models.py`：ChargingScheduler Protocol | 未逐段讲解 | 接口声明 |
| `types.py`：FramePlan | 定义及 serving_ids 属性已确认 | 字段含义已列；字段如何计算仍待 make_plan 后半段 |
| `types.py`：ReturnTrip / LowStepResult 等 | 结构用途已介绍，未逐项验收 | 后续补齐 |
| `low_env.py`：observation / step / apply_final_return_penalty、LowReward | 未逐函数验收 | 部分代码用于解释调用链，仍需主线检查 |
| `low_runner.py`：LowFrameRunner.run_frame | 未逐函数验收 | 何时做最终返航检查、终止与回调 |
| `upper_env.py`：UpperEnv.step | 前半段本体已确认，依赖待验收 | plan 生成、应用、low runner 调用、completed_all_steps 条件结算；后半段待逐段审查 |
| `upper_env.py`：其余 UpperEnv 方法 / UpperReward | 未逐函数验收 | 历史实现和测试不等于本轮逐段确认 |
| `rl/observations.py`：lower_arrays / upper_observation | 明确暂缓 | 用户要求稍后检查；仅介绍过输入含义与调用位置 |
| `rl/observations.py`：RunningMeanStd / copy_lower | 未审查 | 归一化统计更新、复制与恢复 |
| `rl/replay.py`、trainer.py、networks.py、sac.py、mappo.py、transfer.py、exploration.py | 未逐函数验收 | 后续按实际调用顺序展开 |
| factory.py / joint_runner.py、metrics.py / render.py、RL 日志与溯源、运行脚本、tests | 未逐函数验收 | 后续独立审查；tests 为证据，不替代语义确认 |

## 已发现、暂不改动的事项

- cfg 是引用，修改字典不一定同步到已复制为属性的数值。
- reset 非全部 serving 时，waiting 数量没有在该函数校验容量；当前全部 serving 不触发。
- evolve 的动作形状校验不检查 NaN/Inf。
- 边界采用终点坐标截断；死亡采用整步结算，不模拟准确耗尽时刻/位置。
- full-clear 假设同时存在于收集模型和 evolve 外层；已单独记录重构建议。
- observe_upper 的“接口尚未确定”注释是旧注释；当前 MAPPO 从 RL 层单独构造输入。
- 配置防错、旧字段与历史文档歧义在后续配置/文档 review 处理。

## 代码基线

Scene 文件 SHA256（用于提示后续修改，不能代替逐函数 diff）：
`d1cd797f81b2bdfaeda3a8018c6f99c2b8d0e8cf6bc1a6532c3423436ba02c85`

本文件的确认状态基于本次逐段会话；不将以前实现迁移接口、测试通过等历史工作算作用户验收。

### 本轮新增的语义记录

- 帧末 waiting 晋升按当前 SOC 升序、相同时按 ID 升序，不是 FIFO。这是实际实现，若要改先到先服务需另行确认。
- 帧末释放/晋升只改变下一时点的状态，不追补本帧服务/充电时间。
- 充电时长来自计划，正确扣除返航时间与否仍待调度器审查。

### 调用链审查规则补充

用户要求结合实际运行上下文检查：以 UpperEnv.step 的单 frame 为主线，遇到依赖向下展开再返回，追踪状态何时改变、失败分支、结果消费者。单函数本体确认不代替调用链正确性。LowFrameRunner 中 completed_all_steps 目前仅表示执行步数达到配置值；第十步死亡仍可为 True，外层因此仍结算。此条件已定位，runner 完整实现待后续审查。

- make_plan 入口先转 int8 再验证，可能将非整数动作截断后接受；正常策略输出 0/1 不受此问题影响，后续可补原始输入校验。

## 2026-10-04：审查中授权的小修改

- 用户要求删除 retained_charging 的多余排序。已删除对继续充电者的 low_battery_order 与容量切片，所有继续请求的已有 charger 都保留位置。
- 当前 charging 占用超过容量时明确抛出 ValueError，不通过电量排序静默驱逐；即使部分 UAV 本轮请求离开，也先报告已有状态异常。
- 候选新请求与 waiting 的现有排序未改，FIFO 重构仍待后续。
- 用户随后“可以继续了，到remaining了”确认此段修改；不自动视为整个 make_plan 已确认。

### 当前阅读位置补记

- retained_charging 删除排序/截断并增加超容量报错，已由用户确认。
- 本轮按用户指定位置重新从 remaining 开始：请求掩码副本、移除已保留 charger、waiting/other 候选、charging 空位分配，到 remaining[new_charging] = False；讲解待确认。
- 下一段为 retained_waiting 及 waiting 空位分配。

### remaining 与候选队列的替代关系

- 用户本轮“继续”确认上一轮 remaining 至 remaining[new_charging] = False 的代码讲解范围。
- 分配 charging 后 dock_candidates[len(new_charging):] 的 ID 集合与 np.flatnonzero(remaining) 一致；顺序可能不同。改用队列时仍要保留已有 waiter 的位置优先级、组内排序与容量限制。当前仅解释，未修改。
- 本轮讲解到 assigned[waiting_ids] 写入；下一步审查 returning 与返航时间/失败标志。

## 2026-10-05：make_plan 末段

- 已讲解待确认：最终 C/W 掩码、非返航者成本归零、到达 SOC 重算、返航失败标志、充电/等待时间、FramePlan 构造。
- returning 也包括已在 C/W 区的 UAV，不代表都发生物理返航；其成本此前已设零。
- 失败条件是到达 SOC < 0 或返航时间 > frame_sec；未使用 return_reserve_frac。严格边界允许到达 SOC 为零或恰好用满整帧。
- 对服务者，最终 arrival_battery_frac 等于当前 SOC，不是其假设返航裕量，也不预测本帧服务后的电量。
- 失败仍可有正的计划 C/W 时间；make_plan 不做实际移动/扣电/死亡处理，结算方负责屏蔽失败的充电。失败是否及时终止须结合上层和 runner 后续审查。
- 帧起点状态/位置/电量均 copy 保存，Scene 尚未被本函数修改。时间为计划量，可能因满电或提前终止而不等于实际有效时间。
