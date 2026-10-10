# VLFM前沿决策重建可行性（2026-10-10）

## 结论与首个实测

现有记录足以在不启动模型、模拟器、视觉服务的情况下，重放原策略的障碍图、价值图和前沿选择。试验在服务器原始代码 `584ed56008754fde7997d904983607def8328322` 上，以 HM3Dv1 `4ok3usBNeis.basis__0__c638aa99f8e5` 的前40步缓存为输入，step12–39共28次explore决策的nav_goal全部与原记录逐位相等，最大误差0。此为输入链可行性测试，不是成功率实验，也不证明机器人低层动作已重现。

## 可重放与不可直接重放

- RGB PNG、归一化float32 depth NPY、sensor NPZ、每次ITM输入图片/请求/响应均有保存。
- `habitat_policies.py:173–237`使用GPS(x,-y)、compass和常量camera_height构造策略本身的camera-to-episodic变换，先更新ObstacleMap；`itm_policy.py:251–261`随后更新ValueMap；旧模型不需重推理。
- `BaseITMPolicy._get_best_frontier`的原代码可直接执行，必须保留last_frontier、last_value和acyclic历史，不能逐步独立调用。
- `steps.jsonl.policy_info.nav_goal`可逐位验证explore时选中的点。target_detected/console Mode仅用来确定本步是否进行frontier选择，这些模式属于历史条件输入，不属于本重建独立验证范围。
- 没有低层PointNav hidden state/action logits，所以本工具不声称动作重新计算；改变前沿后的新路线、成功率必须真正闭环重跑。
- 没有真实世界逐帧高度/完整姿态；旧策略本身也是常量相机高度。重建其2D内部图不需要这些字段，但多层往返判定与几何真值审计需要额外采集。不能把楼梯的投影交叠当成重复探索。

## 精确日志hook

1. `itm_policy.py:92`排序后：所有原始/排序后frontier、分数、index对应关系；V3另记target/explore原始两通道值及全局channel switch。
2. 同文件101–124：上一前沿是否仍在/0.5m近邻、前值/当前值、keep-last分支。
3. 同文件129–147：实际执行的cyclic检查和选中分支，未访问的候选不能当作通过检查。
4. 同文件66–68：empty或零哨兵导致NO_FRONTIER STOP，区别于目标STOP和预算结束。
5. `ObstacleMap.update_map`结束：保存`_map`、`_navigable_map`、`explored_area`、frontiers原数组；`ValueMap.update_map`结束：保存原始`_value_map`和confidence `_map`。历史保存的map是RGB可视化，不是这些原始数组。
6. `room_online_env.py:10–19`已能在worker读取sim.get_agent_state及sensor_states，复用为旁路采集真实xyz/quaternion，不传入策略、不调用新模型。

注意explored_area并非单调：`obstacle_map.py:126–146`按障碍/连通域清理。因此新增面积应并报当前mask增减及累计union首次覆盖，不能直接以总面积差代替信息收益。

## 原代码风险（只记录，不修基线）

- `acyclic_enforcer.py`的StateAction只有`__hash__`没有`__eq__`；每次check创建新对象，等值内容不会按内容相等命中set。必须对原实现做单元复验并报告，不应偷偷修复后冒充原VLFM。
- all-cyclic fallback打印closest，但实际取输入frontiers最大距离index，然后以该index索引sorted_pts；这也是记录原行为，禁止在诊断时顺手修复。
- 原episode.json的object_category未知、start_position/rotation为空，因为VectorEnv暴露的是简化EpisodeInfo。完整身份必须连接不可变content shard原loaded index核对，不能把该JSON hash当完整任务hash。

## 单key真实重跑

原suite只有scene选择和每scene数量，resume只支持严格已完成前缀。推荐在保持原content不变时、Habitat加载完整episode集合后按scene+loaded episode_id筛选/seek，额外核对目标、start_position、start_rotation、content SHA与源行SHA；不要先裁剪JSON导致Habitat episode_id重编号。固定单env与test_episode_count=1，独立输出目录。

## CPU工具边界

`replay_frontier_probe.py`使用原map实现与AST提取的原选择方法；按step/call严格消耗缓存ITM，核对每次ITM图片与原RGB逐像素相等，保存所有输入与代码hash。优先用冻结8案例manifest，先失败、再失败对照和成功对照。输出steps/decisions/inputs/result与可选lossless map delta。该工具属于“历史决策重建”，不属于第三层干预因果验证。
