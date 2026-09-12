# Codex 任务 B：LAVA 简单双来源负采样消融

## 任务与基线

请直接在 Kaixi66/LAVA 工作区实现本说明，不是只给建议。先检查 HEAD、git status 和已有改动，不要覆盖用户未提交的工作，不要重置仓库。

参照提交：`7bcfc1e2c480e50e3707ac18c8a8a13521667bda`；基线配置：`configs/robotwin_lava_v63.yaml`。

新增独立的 `episode_balanced` negative 模式：目标为每个 anchor 4 个同 episode 时间错配片段 + 4 个其他 episode 的随机片段，全部同 scale。该模式关闭人工 order negative，取消 local/far 分桶和全量 batch negatives。

本任务只改 negative 的采样、数据传递、候选集合与必要的 loss 接线，不改 normalization、score、FiLM 或主模型。B 单独运行时仍用 V6.3 的 `ema_rms_soft + dot`。如果任务 A 已实现，B 必须兼容 A 的 scorer/normalizer，但不能强制开启它们。

这是简洁的采样假设，不保证所有 negative 在操作语义上都不同。

## 1. 正路径与时间约定不变

沿用 `_sample_lava_interval` 选合法正路径：

```text
policy 当前观测的 episode 内帧号 = t_obs
采样子区间的相对起点 = start
正路径绝对起点 p = t_obs + start
scale = L，路径有 L+1 帧
正路径帧号 p, ..., p+L
```

不改变主 policy DataLoader、正区间起点分布、scale 分布、task/episode balancing、LAVA 采样概率和 action hidden 对齐。负候选不足不能反向触发正区间重采样。

## 2. 同 episode 候选：最多 4 个

在 anchor 所在的同一物理 episode 内，合法负起点 n 必须满足：

```text
0 <= n <= total_frames - L - 1
abs(n - p) >= 2 * L
```

均匀、无放回抽最多 4 个合法起点；每段读取 n,...,n+L 的真实帧。

注意：

- 排除距离相对正片段起点 p，而不是 policy 当前观测 t_obs。
- 不再使用 `min(4L,32)` 的 local 半径，不划 local/far 两桶。
- `2L` 是本次消融选定的 heuristic，不宣称是阶段边界。
- 只要求这些候选与正片段满足间隔，不额外要求负候选彼此间隔 2L，不添加新的筛选规则。
- 合法起点少于 4 个就全取；一个也没有就返回空列表。不能重复采样凑满，不能放松间隔，不能丢掉正路径。

例：T=64、p=24、L=16 时，同 episode 合法负候选为零，但正路径仍保留，可使用跨 episode 候选训练。

## 3. 跨 episode 候选：最多 4 个

从当前 batch 中已经抽到并编码了 DINO 特征的其他合法正路径中选择：

```text
candidate scale == anchor scale
candidate episode_uid != anchor episode_uid
```

在符合条件的独特路径中均匀、无放回抽最多 4 个。以 `(episode_uid, absolute_start, scale)` 去重，避免同一路径重复贡献分母。

`episode_uid` 必须标识整个数据集中唯一的物理 episode，可使用规范化后的 HDF5 路径；不能只用可能在不同 task/clean/randomized 下重复的 `episode0`，也不能把 task_name 当 episode ID。

同 task 的其他 episode 允许作为候选；不刻意偏向同 task 或跨 task。不做 embedding hardest mining、不做 action 距离、DTW、聚类或权重筛选。同一个其他 episode 中不同的片段可自然出现，不新增每 episode 配额。

少于 4 个就取现有数量；不能把同 episode 的 batch 路径冒充跨 episode 候选，不新增全数据集搜索或额外图像读取来凑满，不重复填充。

## 4. 候选数不足与归属

每个 anchor 最终有：

```text
1 positive + 0..4 same_episode + 0..4 cross_episode
```

4+4 是候选充足时的目标，不靠重复或权重制造固定 50/50。真实数量要记录。

不得在此基础上再追加旧 local、旧 far、全部 batch candidates 或 order candidate，否则不是这个消融。

单个 anchor 完全没有合法 negative 时，只跳过它的 LAVA 对比项，不重采样数据、不删除该 policy 样本的 flow/future loss。该正路径仍可供别的 anchor 当候选。LAVA loss 对至少有一个 negative 的 anchor 取平均；整个 batch 都无可用对比时，返回与现有梯度路径相连的零值。

区分 sampled/encoded/scored anchor 数量，记录无候选原因；不要为了通过 health monitor 伪造 sample_count 或关闭监测。可按其接口最小扩展，让合理缺候选与管线出错可区分。

如果 normalization 是 EMA，仍按本次采到的合法正路径更新参考一次，候选不参与参考更新；不要因为某个 anchor 无 negative 再改变 normalization 的语义。

## 5. FiLM 与打分保持一致

对 anchor i，其 positive、同 episode、跨 episode 候选全部使用 i 的 policy 当前观测 context 编码。

```text
world_candidate_for_row_i = Phi(candidate_frames | context_i)
```

不使用候选所属 episode 的当前帧，不使用候选第一帧冒充 anchor context。每条 candidate 的所有帧使用同一 context。

跨 episode 可复用冻结 DINO 特征，但不能直接复用按候选自己 context 生成的最终 world signature。保留按 row context 重新解码的原则。

随机候选选择必须发生在 checkpoint 重计算函数外，forward/backward 使用同一候选索引。旧 EMA 的 forward 参考快照规则也保留。

打分调用现有 scorer；若 A 已实现，则调用选中的 `dot` 或 `neg_l2`。B 不自行修改 score 或温度。

每个有效候选直接作为一个独立 logit 进入同一个 cross entropy 分母：

```text
scores = [score(a, positive), score(a, n1), ..., score(a, nk)]
loss_i = cross_entropy(scores / temperature, positive_index)
```

不做按来源的 log-mean-exp、不做 family count averaging、不增设同/跨 episode loss 权重。不增加独立 order loss。无候选的位置使用显式 mask，不把零张量当真实 negative。

## 6. 数据与接口最小改动

重点检查：

- `dataloader/dataset.py`：新模式的采样；传递 episode_uid、正路径绝对起点和同 episode 负路径列表；`collate_fn` 支持每个 anchor 零到多个负路径。
- `models/model_runner.py`：复用原 DINO 提取，编码并传递新负路径列表与 metadata；不强制新模式提供旧的 local/far 字段。
- `models/vla_model_fm.py`：新候选模式、mask 和 loss 组装；将原先 query_film 只允许 mixed_batch 的检查最小扩展到支持新模式。
- `train.py`/factory/config：只做必要的开关、元信息与诊断接线。

episode_uid、绝对帧号等元信息只供采样和核验，不能输入神经网络。零到多个候选的 ragged list 或 padding+mask 均可，不需要新增采样框架、队列、memory bank 或外部服务。

旧 mixed_batch 及其他已有模式保留原行为，包括原来的数据加载与严格校验；仅新模式允许新的可变候选语义，不要全局取消校验。

## 7. 配置

建议使用以下字段，旧配置缺失时走原分支：

```yaml
training:
  lava_negative_mode: episode_balanced
  lava_num_same_episode_negatives: 4
  lava_num_cross_episode_negatives: 4
  lava_negative_exclusion_multiplier: 2
  lava_order_negative: false
```

新模式首版要求 order_negative=false、action_similarity_weighting=false；配置不一致应报错，不悄悄增加候选。新配置不要以旧 local/far 半径控制此模式。

生成独立完整配置：

- `configs/ablations/robotwin_lava_v63_negatives.yaml`：以原 V6.3 为基线，只切上述新 negative 方案，继续 ema_rms_soft + dot。
- `configs/ablations/robotwin_lava_v63_order_off.yaml`：原 V6.3 只关闭 order，其他保持 mixed_batch。作为区分“去掉 order”和“换采样”收益的低成本控制配置，不要求自动启动实验。

保留原 batch size、学习率、温度 0.07、lambda 0.01、warmup、尺度、比例、FiLM、归一化、score 与训练时长。run/checkpoint 标识独立，不能覆盖旧结果。

B 是“负样本方案整体消融”，同时改变了来源配比、总数量、时间排除范围、order 项和缺候选的处理；不要把 B 相对 V6.3 的收益只归因于某一条规则。

## 8. 必须验证

1. 同 episode 每段恰有 L+1 帧，不越界，`abs(n-p)>=2L`，无放回。覆盖 L=1/2/4/8/16、边界和候选不足。
2. `interval_start>0` 时，排除范围以 p=t_obs+start 为基准；FiLM condition 仍为 t_obs 的观测。
3. 同 task 不同 episode 可用于跨 episode；不同 task 下都叫 episode0 仍必须辨认为不同 episode；同 episode 的 batch 样本不得混入 cross 候选。
4. 候选够时为 4+4；不够时少取，无重复扩增；缺 same 或 cross 不丢合法正路径。
5. 没有 order/旧 local/旧 far/未选 batch 候选偷偷进入分母。手工 CE 与实现一致。
6. 同一 row 的所有 world 候选使用 row context，checkpoint 重计算没有重抽样。
7. 如果归一化为 EMA，依然只使用正路径更新一次，全部候选共享这次 forward 的参考快照。
8. 候选可用率、实际 same/cross 数量、无候选 anchor 数可读；health monitor 不被绕过。
9. 固定输入下旧模式回归测试通过；新模式有有限 loss/梯度，DINO 冻结，推理路径不变。
10. A 尚未实现时 B 能独立训练；A 存在时能组合且不覆盖 A 的 scorer/normalizer 配置。

输出改动文件、采样定义、配置差异、测试命令与真实结果。不编造通过；不提交集群任务、不启动长期训练、不覆盖 checkpoint。
