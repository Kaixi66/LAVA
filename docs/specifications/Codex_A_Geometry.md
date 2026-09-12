# Codex 任务 A：LAVA 几何读出与距离打分消融

## 任务与基线

请直接在 Kaixi66/LAVA 工作区实现本说明，不是只给建议。先检查 HEAD、git status 和已有改动，不要覆盖用户未提交的工作，不要重置仓库。

本说明对照的提交是 `7bcfc1e2c480e50e3707ac18c8a8a13521667bda`，基线配置是 `configs/robotwin_lava_v63.yaml`。若 HEAD 更新，先核对相关接口再移植下面的设计，不要求退回该提交。

只实现两项可独立开启的变化：

- 新增 `graded_soft`：一阶、二阶使用同一个路径尺度，按阶缩放。
- 新增 `neg_l2`：使用负平方欧氏距离作为匹配分数。

这是研究假设，不是已经验证有效的方法。不要顺便改 negative 构造、FiLM、action projector 或训练日程。任务 B 是独立负采样任务；A 单独运行时必须保留原负采样和原 order 开关。

## 1. 新归一化的确切定义

令 D 是原始一阶 LogSig，A 是原始反对称二阶 LogSig 的上三角坐标，沿用现有 `_raw_logsignature_levels`。不要展开成完整反对称矩阵后重新计范数，否则会引入额外的两倍因子。

对每条路径独立计算：

```text
rho = 1.0
s = (rho**4 + ||D||_2**4 + ||A||_2**2)**(1/4)
z = concat(D / s, A / s**2)
```

“共用一把尺子”指一条路径内两阶共用 s，并且两端使用同一个函数和 rho；不是 action/world 共用一个从二者联合估计的数值 s，也不是全 batch 共用一个数值 s。每条 action path、每条按 anchor 条件化后的 world candidate 分别算自己的 s。

`rho` 是固定正数，不训练、不做 EMA、默认 1.0；配置中必须验证有限且大于零。它是选定的 latent 尺度，并非已经知道的物理噪声阈值。

参考实现：

```python
# D: [..., d]，A: [..., d * (d - 1) // 2]
# 本函数在关闭 autocast 的 FP32 区域运行。
delta = D.float()
area = A.float()
if depth == 1:
    area = torch.zeros_like(area)  # 深度 1 的尺度不能依赖被丢弃的二阶信息
elif scale == 1:
    area = torch.zeros_like(area)  # 单条线段的结构性零二阶

d2 = delta.square().sum(dim=-1, keepdim=True)
a2 = area.square().sum(dim=-1, keepdim=True)
s2 = (rho**4 + d2.square() + a2).sqrt()  # s**2
s = s2.sqrt()
if depth == 1:
    z = delta / s
else:
    z = torch.cat((delta / s, area / s2), dim=-1)
```

要求：

- s 保留梯度，不 detach；原始 D、A 的构造完全不变。
- 不做 per-level unit normalization，不做 EMA，不做最终 F.normalize，不额外除 sqrt(2)，不增加 level-2 权重。
- 深度 2 保留当前输出维数。128 维增量对应 8256 维读出；L=1 的二阶补零，不删除这些坐标。深度 1 按现有接口只输出一阶。
- 首版仅支持 `state_delta`、无 time channel。对新模式下不支持的组合明确报错，不影响旧模式。
- 不把路径级 s 反馈到逐帧编码、FiLM 或 action projector；它只是原始 LogSig 后的读出。
- 不宣称各窗口独立归一化后的读出仍能直接满足原始 Chen 组合公式。原始 residual 的 telescoping 性质仍应保持。

## 2. 独立的打分开关

新增 `model.lava.signature_score`，允许 `dot`（默认，原行为）和 `neg_l2`。

```python
# 支持配对和广播后的 anchor-candidate 计算。
# a/w 是归一化后的 signature，而不是动作数值或帧特征。
if score_type == 'neg_l2':
    score = -(a.float() - w.float()).square().sum(dim=-1)
else:
    score = existing_dot_product(a, w)
```

注意必须是坐标求和，不是 mean；不要用普通 L2 的 sqrt，不是 cosine，不再先单位化，也不要悄悄乘 0.5。

整个匹配目标继续用现有 `cross_entropy(logits / temperature, labels)`。所有进入同一个损失的 positive、batch、local、far、order 分数必须统一用选择的 scorer，不能只改 positive 或 batch。原先的 family reduction、mask 和候选数完全不动。order 分数若存在，先用新 scorer，再按旧规则聚合。

点积旧分支尽量原样保留，以保证关闭新开关时旧行为不变。匹配相关诊断也应使用相同的分数；距离模式下字段不能被解释成 cosine。纯表示诊断可保留原定义，但要明确标注。

## 3. FiLM 与 EMA 的边界

继续使用当前观测的 context，整条路径固定，所有候选使用 ROW anchor 的 context。

`graded_soft` 没有 EMA 状态，也不更新旧 calibrator；保留旧 normalizer 的实现与 checkpoint 兼容性。`ema_rms_soft + neg_l2` 则仍使用旧 EMA，并保留本次 forward 的参考快照与 checkpoint 重计算规则。

不要为了支持新 normalizer 全局使用 `strict=False`。这些消融配置默认从头训练；跨 normalizer 的旧 checkpoint 权重迁移不属于本任务。

## 4. 配置与范围

新增字段示例，旧配置缺失时默认保持原行为：

```yaml
model:
  lava:
    signature_normalization:
      type: graded_soft
      rho: 1.0
    signature_score: neg_l2
```

请从 V6.3 复制生成独立可运行配置，不假设未实现的 YAML 继承功能：

- `configs/ablations/robotwin_lava_v63_norm_only.yaml`：graded_soft + dot。
- `configs/ablations/robotwin_lava_v63_score_only.yaml`：ema_rms_soft + neg_l2。
- `configs/ablations/robotwin_lava_v63_geometry.yaml`：graded_soft + neg_l2。

所有配置保留 V6.3 的 FiLM、8×16 queries、depth 2、无 time channel、负采样、order 开关、温度 0.07、lambda 0.01、尺度、采样比例、warmup 和训练时长。修改必要的 run/checkpoint 标识，不能覆盖旧输出。

重要解释：负平方距离与点积的分数量级不同。由
`-||a-w||² = 2a·w - ||a||² - ||w||²`
可知，固定一行时距离分数的点积系数是 2。因此固定温度的 score-only 对照是“打分规则整体变化”，不只是在测试候选范数惩罚。文档注明这一点；可额外提供 score-only、temperature=0.14 的温度系数对照，但不要自动替换主配置的 0.07，也不要启动搜索。

## 5. 修改位置

优先检查：

- `models/vla_model_fm.py`：normalizer 类型检查/构造，`compute_lava_loss` 的 signature build 与所有打分位置。
- `models/model_runner.py`：ModelFactory 配置透传。
- `train.py`：只有参数透传、已有诊断接线确实需要时才修改。
- 新测试文件及上述消融配置。

不要重写 dataloader，不改负样本采样，不加新网络、噪声估计器、置信度 head、学习式温度或新 loss。当前候选集是现有 negative；A 不依赖 B。

## 6. 必须验证的行为

1. 关闭新选项，固定权重、输入、候选和随机状态时，旧输出、loss 与梯度保持兼容。
2. 原始零路径读出为零，前向和反向有限；尺度由 rho 确定，不受其他样本或调用次数影响。
3. 固定一阶，二阶从 1e-2 缩到 1e-6 时，二阶读出仍变弱；不被单独拉成单位范数。
4. 深度 2/L=1 维数仍是 8256，二阶全零；深度 1 的读出不依赖传入的 A。
5. 相同输入 a=w 时 neg_l2 最高为 0；任意不同的 w 得分更低。构造“方向稍差但范数更大”的候选，验证不会打败完全相等的正确表示。
6. 配对与 anchor-candidate 批量 scorer 与手工求和一致，覆盖长度 1/2/4/8/16 与 BF16 autocast。
7. 条件化 candidate checkpoint 开/关的 loss 和梯度一致；新 normalizer 不引入可变统计状态。旧 EMA+新 score 的参考快照不漂移。
8. action projector、FiLM/world encoder 能收到有限梯度，DINO 仍冻结；推理接口与执行路径不变。

输出：改动文件、关键公式、各配置的实际差异、测试命令与真实结果。缺环境就注明未跑项，不编造通过。不要自动提交 Slurm、启动长期训练或覆盖 checkpoint。
