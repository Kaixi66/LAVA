# LAVA: Learning Action Semantics from Multi-Scale World Evolution

LAVA is a research implementation that extends
[LiLa-WAM](https://github.com/teee000/LiLa-WAM) with an auxiliary training
objective for aligning action representations with multi-scale visual world
evolution. The flow-matching and future-feature objectives from LiLa-WAM are
retained. LAVA is used only during training and adds no inference-time branch.

> This repository is an experimental research fork, not the official
> LiLa-WAM repository. Results are still being validated; no benchmark claim is
> made here yet.

## Method

For a sampled interval of length $L$, LAVA V6 extracts frozen DINOv3 patch
features for all $L+1$ frames. One shared Q-Former encodes each frame into
8 ordered queries, each projected to 16 dimensions and flattened:

$$
Y_i = \Phi(Z_i) \in \mathbb{R}^{128}, \qquad r_i^W = Y_{i+1} - Y_i.
$$

The shared action MLP projects final-normalized hidden states immediately
before the action head into 128 dimensions. World increment $r_i^W$ aligns
with $h_{i+1}$; $h_0$ is never used by LAVA.

Both residual paths use a global differentiable depth-2 log-signature without
a time channel. Raw first-level world signatures telescope to the state
endpoint difference. Per-level normalization follows signature construction.
The default `mixed_batch` objective uses
every other same-scale world path in the batch as an individual negative,
including same-task paths. Self-pairs and different-scale paths are excluded.
Each batch candidate enters the InfoNCE denominator directly, with no family
count averaging and no action-similarity weighting.

Each anchor also retains a same-episode local path, a far path, and an order
candidate. The order candidate is the log-mean-exp of a contiguous block swap
and a full derangement, enabled only for $L\geq4$. Local, far, and order
candidates are also unweighted. The legacy `mixed` objective remains available
for reproducing older experiments; it is not the default training objective.

The default restores the V2/V3 block-wise normalization: first- and
second-level LogSignatures are normalized independently, concatenated, and
normalized once more. Every non-degenerate depth-2 path therefore assigns
equal representation energy to endpoint displacement and temporal order.

The training objective is

$$
\mathcal L = \mathcal L_{\mathrm{flow}} + 0.5\,\mathcal L_{\mathrm{future}} + \lambda_{\mathrm{LAVA}}(s)\,\mathcal L_{\mathrm{LAVA}}.
$$

where $\lambda_{\mathrm{LAVA}}$ linearly warms from 0 to 0.01 over the first
5% of optimizer steps.

## Default LAVA configuration

The defaults live in [`configs/robotwin_all.yaml`](configs/robotwin_all.yaml):

```yaml
model:
  lava:
    enabled: true
    action_target_layer: final
    dino_target_layer: -4
    world_encoding: state_delta
    time_channel: false
    residual_dim: 128
    qformer:
      hidden_dim: 256
      num_queries: 8
      query_dim: 16
      num_layers: 2
      num_heads: 4
    logsig_depth: 2
    signature_normalization:
      type: per_level_unit

training:
  lambda_lava: 0.01
  lava_temperature: 0.07
  lava_scales: [1, 2, 4, 8, 16]
  lava_sample_ratio: 0.125
  lava_scale_sampling: batch_uniform
  lava_sampling_balance: task_episode
  lava_negative_mode: mixed_batch
  lava_negative_window_multiplier: 4
  lava_negative_window_max: 32
  lava_warmup_ratio: 0.05
  lava_order_negative: true
  lava_grad_diagnostics_interval: 200
  lava_action_similarity_weighting: false
  lava_action_component_calibration: false
```

`lava_sampling_balance: task_episode` keeps the base flow/future DataLoader
frame-uniform while equalizing the expected number of LAVA paths across tasks
and then across episodes within each task. The provided RoboTwin Slurm scripts
enable this mode explicitly and write an actual per-task sampling audit after
every epoch.

`mixed_batch` mode uses `lava_sample_ratio: 0.125`, a local search radius
$R(L)=\min(4L,32)$, and a far path outside that radius. Batch candidates
reuse positive paths already encoded for the batch; order corruptions reuse
the positive residual path and therefore require no extra DINO forward.

The fixed method choices are:

- frozen DINOv3 targets from layer `-4`;
- final-normalized hidden states before the action head;
- latent state differences `Phi(Z[t+1]) - Phi(Z[t])`;
- 8 ordered 16-dimensional queries and no time channel;
- one-way Action-to-World contrast with individual batch candidates;
- a shared action projector across positions and scales;
- no LAVA execution during policy inference.

The active RoboTwin scripts use run name
`robotwin10_lava_v6_1h100nvl`. They retain the V5.1 sampling
ratio (0.125), batch-uniform scales, task/episode balancing, final action tap,
and per-level normalization. CSV diagnostics add `Batch_Candidate_Count`,
`Same_Task_Batch_Candidate_Count`, `Negative_Candidate_Count`, `Batch_Margin`,
`Batch_Acc`, and hardest batch/same-task fractions. The negative candidate count
counts the pooled order candidate once. Existing cross-task family metrics
remain descriptive diagnostics; they do not define the new batch loss.

## Installation

Python 3.10 is recommended.

```bash
conda create -n lava python=3.10 -y
conda activate lava

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.0.0rc0 omegaconf accelerate h5py pytest
```

LAVA uses the frozen `dinov3-vitl16-pretrain-lvd1689m` encoder. Update these
configuration fields before training:

| Field | Description |
|---|---|
| `dataset.dataset_dir` | Processed RoboTwin dataset root |
| `dataset.task_cond_dir` | Precomputed VTT/task-condition directory |
| `model.vision_encoder.checkpoint_path` | Local DINOv3 checkpoint directory |

The processed 50-task RoboTwin dataset and LiLa-WAM checkpoints are described
in the [upstream repository](https://github.com/teee000/LiLa-WAM). This
repository does not include raw datasets, DINO weights, training runs, or model
checkpoints.

## RoboTwin task sets

`dataset.task_set` accepts:

- `"50"`: all RoboTwin tasks;
- `"10"`: the 10-task development subset, capped at 50 clean and 200
  randomized demonstrations per task.

The subset utility is available at
[`utils/build_robotwin_10_subset.py`](utils/build_robotwin_10_subset.py).

## Training

Stage 1 uses 12 epochs with a peak learning rate of `2e-4`:

```bash
python train.py --config configs/robotwin_all.yaml --set \
  training.epochs=12 \
  training.learning_rate=2e-4
```

Stage 2 initializes model weights from the final Stage-1 checkpoint but creates
a fresh optimizer and scheduler. Do not use `--resume` for the stage switch:

```bash
python train.py --config configs/robotwin_all.yaml \
  --init_from /path/to/stage1_epoch_12.pt \
  --set training.epochs=4 training.learning_rate=4e-5 \
        training.lava_warmup_ratio=0.0
```

Use `--resume` only to recover an interruption within the same stage; it
restores model, optimizer, scheduler, epoch, and LAVA warmup progress.

CML-specific single-H100-NVL Slurm templates and an `afterok` two-stage
launcher are under [`scripts/slurm`](scripts/slurm).

## Monitoring

The optimizer-step CSV records base losses, LAVA behavior, representation
health, per-scale metrics, flow-timestep bins, gradients, throughput, and GPU
memory. The most important diagnostics are:

```text
Loss_Flow, Loss_Future, Loss_LAVA, Lambda_LAVA
Loss_Base, Weighted_LAVA, LAVA_Base_Ratio
Pos_Sim, Negative_Sim, Shuffle_Sim, Order_Margin, Retrieval_Acc
Temporal_Neg_Sim, Temporal_Margin, Temporal_Acc
Order_Acc, Candidate_Acc, Positive_Temporal_World_Sim
Cross_Task/Far/Block_Swap/Derangement_Sim, Margin, Acc
Hardest_Cross_Task/Local/Far/Order_Fraction
Negative_Distance_Over_L, Far_Negative_Distance_Over_L, Dropped_Pair_Rate
Same_Task_Neg_Sim, Cross_Task_Neg_Sim, Task_Shortcut_Gap
Loss_S1/S2/S4/S8/S16
Pos_Sim_S1/S2/S4/S8/S16
Order_Margin_S1/S2/S4/S8/S16
Action/World_LogSig_L1/L2_Raw_Norm, Action/World_LogSig_L2_L1_Ratio
Action/World_LogSig_L1/L2_Calibrated_Norm, L2_Energy_Fraction
Loss_LAVA/Pos_Sim/Candidate_Acc/Local_Margin/Order_Margin_Executed/Tail
Grad_Norm, Grad_Norm_LAVA_Branch
Grad_Cos_Shared, Grad_Norm_Base/LAVA_Shared, Weighted_Grad_Ratio
Arm_State/Arm_Change/Gripper_State/Gripper_Change/Raw_Combined/Combined_Action_Distance
Calibrated_Arm_State/Arm_Change/Gripper_State/Gripper_Change_Action_Distance
Action_Component_Beta_*, Action_Distance_*_Contribution_Fraction
CrossTask/Local/Far_Neg_Weight_Mean, Effective_Negative_Mass
Raw/Weighted_CrossTask/Local/Far_Margin/Acc
```

`Order_Margin` uses the count-balanced order family containing a contiguous
block swap and a full derangement. Per-corruption metrics remain available in
the CSV so the structured and destructive order tests can be studied
separately.
Scales 1 and 2 have no order negative. Shared-gradient diagnostics run every 200
optimizer steps; their CSV fields are `nan` on non-diagnostic steps.

## Tests

Run the focused LAVA tests with:

```bash
pytest -q tests/test_lava.py
```

They cover interval bounds, the one-frame action/world offset, depth-2
log-signature dimensionality and order sensitivity, mixed-precision InfoNCE
backpropagation, frozen-DINO behavior, and the inference fast path.

## Evaluation

The inherited RoboTwin evaluation entry point is documented in
[`README_EVAL.md`](README_EVAL.md). LAVA modules are present in LAVA
checkpoints but are not called during policy inference.

## Acknowledgements

LAVA is built directly on LiLa-WAM. Please cite the original work when using
this repository:

```bibtex
@article{yang2026lila,
  title={LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robotic Manipulation},
  author={Yang, Fan and Su, Yuting and Wang, Xiaobo and You, Yuncheng and Fan, Fugui and Wu, Yuting and Wu, Minghui and Zhao, Chenxu and Ning, JiaHong and Jing, Peiguang},
  journal={arXiv preprint arXiv:2608.03701},
  year={2026}
}
```

The upstream paper and project page are available from the
[official LiLa-WAM repository](https://github.com/teee000/LiLa-WAM).

### LAVA V6

`robotwin10_lava_v6_1h100nvl` encodes each frozen DINO frame with the same
Q-Former (8 ordered queries, hidden size 256, two layers, four heads). A shared
256→16 projection per query produces a flattened 128-dimensional state `Y_t`.
The world increments are `Y_{t+1} - Y_t`; projection and subtraction use FP32
under BF16 training. The action projector maps final action hidden states
`h_{t+1}` to 128 dimensions. No query mean pooling or time channel is used.
The global depth-2 LogSignature contains 8,256 coordinates, including
cross-query areas. Its raw first level telescopes to the endpoint difference;
per-level normalization is applied only after signature construction.

V5.2 negatives are retained: unweighted individual same-scale batch candidates
(including same-task paths), paired local/far paths, and pooled block-swap /
derangement negatives for L≥4. Lambda=0.01 and sampling ratio=0.125 are unchanged.
`state_endpoint_error`, `order_signature_distance`, and
`order_equivalent_fraction` (normalized signature distance <1e-5) monitor the
state identity and degenerate order negatives without filtering candidates.
Old configs default to feature-delta encoding, mean pooling, and a time channel.

The requested run is Stage 1 for 12 epochs on one H100 NVL, followed by the
Stage-1 epoch-12 evaluation on one L40S using an `afterok` dependency.

### Dataset validity fix (2026-09-09)

The original V5.2 job 7479255 (54,288 logged steps) and V6 job 7479456
(27,144 logged steps) never received LAVA supervision. With action weighting
disabled, negative action paths were not requested, but `__getitem__` still
read them. Its broad exception handler recursively resampled and silently
discarded every LAVA example. Their checkpoints and evaluations are invalid
as LAVA results; originals are preserved under
`runs/LAVA/invalid_runs/20260909_103505` in the workspace, outside auto-resume paths.

Negative actions are now read only when action weighting is enabled. Dataset
errors and missing requested evolution frames raise immediately with episode
context. `LAVAHealthMonitor` verifies batch-to-loss sample counts, finite loss,
and autograd connectivity; it stops after 100 consecutive unsupervised batches
(configurable through `training.lava_max_empty_batches`). HDF5 and multiworker
regressions cover weighted/unweighted local, mixed, and mixed-batch paths at
all five temporal scales. Production-shaped GPU validation additionally runs
real data through frozen DINO, the full policy and LAVA backward before restart.
