## srcjv_uav\types.py

### class LowStepResult

#### variables
----------以下生成于scene.evolve_low_step(),包含生成于 service_model 里的
---------本身就在scene里的
- step_in_frame: int                            [当前位于这个frame里的第几个step]
- serving_ids: np.ndarray                       [服务无人机的id list]
- position_before: np.ndarray                   [所有UAV 当前step移动前的位置]
- position_after: np.ndarray                    [所有UAV 发生移动后的位置]
- requested_displacement: np.ndarray            [所有UAV request的位移 (限制最大距离后)]
- executed_displacement: np.ndarray             [所有UAV executed的位移(限制地图尺寸后)]
- move_distance_m: np.ndarray                   [实际移动的长度]
- move_time_sec: np.ndarray                     [实际移动时间]
- service_time_sec: np.ndarray                  [可用于服务的时间]
- energy_used_frac: np.ndarray                  [用了多少能量]
- arrivals: np.ndarray                          [实际包到达的list]
- packets_pre_service: np.ndarray               [本 step 产生 arrivals 后, 假设尚未执行收集时的 backlog list]
- packets_post_service: np.ndarray              [这回合服务后sensor的backlog]
    ---------- 生成在service_model里的 --------------
- sensor_owner: np.ndarray                      [每个sensor分配给哪个uav list]
- collected_per_sensor: np.ndarray              [每个sensor收到的数据 list]
- collected_by_uav: np.ndarray                  [每个UAV收到的sum packet list]
- per_uav_owned_max_pre_service: np.ndarray     [每个UAV收到的最大packet list]
    -------------------------------------------------
- covered_max_pre_service: float                [本 step 被 UAV 覆盖的 sensors 中，收集前最大的 backlog。]
- system_max_pre_service: float                 [所有 sensors 中，服务前最大的 backlog。]
- system_max_post_service: float                [完成收集后，所有 sensors 中最大的剩余 backlog。]
- system_mean_post_service: float               [完成收集后，所有 sensors 的平均剩余 backlog。]
- oob_mask: np.ndarray                          [UAV是否出界的list]
- oob_overflow_distance_m: np.ndarray           [UAV出界的距离]
- dead_during_step: np.ndarray                  [UAV在frame中dead的id list]
-------------以下生成于
- return_unsafe_ids:
    deferred_return_failure_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    reward: float = 0.0
    reward_terms: dict[str, float] = field(default_factory=dict)


## src\jv_uav\scene.py

### class Scene

描述： 整个系统的场景演化发生的地方
- variables

- functions
    - reset()
        - cold_start: 冷启动 sensors buffer里的packets
        - last_visit_sec reset
        - uav 放回 airship
        - uav 电量变满
        - uav 都进入serving状态
        - 时钟置 0s
        - 上层 step置 0s
        - 下层step数量 = step 0
        - 当前step 在frame 中的位置 = 0
    
    - observe_lower(): [global] obs: [[serving_ids], [serving_pos], [serving_batterys], [sensor_last_visit]]  
    - observe_upper(): [global] obs: [[uav_pos], [uav_battery], [uav_status], [charging_occupancy], [waiting_occupancy]]

    - begin_upper_frame()
    
    - evolve_low_step()
    - mark_unable_to_return_dead()
        - 根据energy model算出 所有服务uav返航需要消耗的能量，如果这个能量小于 他们当前的能量+ 一个预留的能量for safe，就判定为un safe
        - output: unsafe UAV list

    - complete_upper_frame()

    - serving_ids()
    - waiting_ids()
    - charging_ids()
    - dead_ids()








## src\jv_uav\models.py

### class FullClearNearestService

描述： 执行一回合(60s- mov_time) 收集数据的过程 [假设full clear]
- variables
    - num_uavs (无人机数量)
    - radius (覆盖半径)

- functions (methods)
    - collect()
        - 描述： 模拟full clear 的过程 
        - inputs: sensor_pos (传感器的位置), packets (sensor backlog 的 copy), uav_pos (uav 的位置), service_ids (服务无人机组)
        - outputs: (class) FullClearResult
            - owner
            - collectedd_per_sensor
            - collected_by_uav
            - per_uav_max
        - 逻辑：
            1. 生成一个owenr 数组，得到 cover情况： e.g. [1, 0, 5, -1, -1]，表示sensor覆盖情况为： 0，1，2 分别被 UAV1, UAV0, UAV5，覆盖， 而 sensor 3,4 未覆盖，未覆盖。
            2. sensor 有 UAV 服务（owner >= 0）就记录它的全部 packets，否则记为 0，保存在 collected_per_sensor里。
            3. 对于所有uav id, 如果id 是当前sensor的 owner， 就 加到 collected_by_uav 里. 最终  collected_by_uav 记录了uav本回合收集的所有packets大小, per_uav_max 则记录当前回合uav收集到最大的packets length.
            4. 返回 FullClearResult


### class LowReward

计算reward:
    reward = covered_term + system_term + oob_term + death_term 



### class LowEnv

- variables
    - scene
    - energy model
    - metrics
    - cfg
    - reward_model

- functions
    - observation() 调用 scene.observe_lower()
    - step()
    - apply_final_retrun_penalty()
        - input: result, unsafe_serving_ids, deferred_return_failure_ids
        - 计算：result.reward, result.reward_terms, 保存result.return_unsafe_ids, result.deferred_return_failure_ids
     




## src\jv_uav\low_runner.py

### class LowFrameRunner

- variables
    - low_env [class LowEnv]
    - low_policy [class LowPolicy]
    - energy_model [class EnergyModel]

    - scene: low_env.scene

- functions
    - run_frame()
        - inputs: plan [class FramePlan]
        - outputs: LowFrameResult


