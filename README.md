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

For a sampled interval of length $L$, LAVA extracts frozen DINOv3 patch
features from the same episode and constructs visual changes

$$
\Delta Z_i = Z_{i+1} - Z_i.
$$

A learnable-query World Residual encoder compresses every patch-level change to
a 32-dimensional residual. On the action side, LAVA V5 uses the final-normalized
hidden state immediately before the linear action head and a shared MLP
projector. The alignment
is offset by one step:

$$
\Delta Z_i \longleftrightarrow h_{i+1},
$$

so $h_0$ is never used by LAVA.

Both residual paths receive a normalized time channel and are pooled with a
differentiable depth-2 log-signature. LAVA V5 keeps four count-balanced negative
families: cross-task same-scale paths from the current batch, a same-episode
far path, a same-episode local path, and order corruptions. The order family
contains a contiguous block swap and a full derangement; for $L=2$ their only
unique permutation is included once.

Cross-task, local, and far negatives are softened only when their detached,
ground-truth action paths are similar. The descriptor combines normalized arm
joint deltas with raw gripper state and gripper change. Order corruptions always
retain weight 1. Each temporal scale has a family-balanced EMA distance scale:
cross/local/far medians are computed separately and then averaged equally.

The raw first- and second-level LogSignatures are calibrated by detached
action/world-, level-, and scale-specific EMA RMS statistics. They are then
concatenated with a fixed second-level weight of 0.5 and normalized once. This
preserves weak-vs-strong order structure instead of forcing each path's two
levels to unit norm independently.

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
    residual_dim: 32
    qformer:
      hidden_dim: 256
      num_queries: 1
      num_layers: 2
      num_heads: 4
    logsig_depth: 2
    signature_normalization:
      type: ema_calibrated
      ema_momentum: 0.99
      level2_weight: 0.5

training:
  lambda_lava: 0.01
  lava_temperature: 0.07
  lava_scales: [1, 2, 4, 8, 16]
  lava_sample_ratio: 0.125
  lava_scale_sampling: batch_uniform
  lava_sampling_balance: task_episode
  lava_negative_mode: mixed
  lava_negative_window_multiplier: 4
  lava_negative_window_max: 32
  lava_warmup_ratio: 0.05
  lava_order_negative: true
  lava_grad_diagnostics_interval: 200
  lava_action_similarity_weighting: true
  lava_action_similarity_min_weight: 0.1
  lava_action_similarity_beta_momentum: 0.99
  lava_action_similarity_beta_multiplier: 1.0
  lava_action_gripper_indices: [6, 13]
  lava_action_gripper_state_weight: 0.5
  lava_action_gripper_change_weight: 0.5
```

`lava_sampling_balance: task_episode` keeps the base flow/future DataLoader
frame-uniform while equalizing the expected number of LAVA paths across tasks
and then across episodes within each task. The provided RoboTwin Slurm scripts
enable this mode explicitly and write an actual per-task sampling audit after
every epoch.

`mixed` mode uses `lava_sample_ratio: 0.125`, a local search radius
$R(L)=\min(4L,32)$, and a far path outside that radius. Cross-task candidates
reuse positive paths already encoded for the batch; order corruptions reuse
the positive residual path and therefore require no extra DINO forward.

The fixed method choices are:

- frozen DINOv3 targets from layer `-4`;
- final-normalized hidden states before the action head;
- visual differences `Z[t+1] - Z[t]`;
- normalized time augmentation;
- one-way Action-to-World mixed-family candidate contrast;
- a shared action projector across positions and scales;
- no LAVA execution during policy inference.

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
Arm/Gripper_State/Gripper_Change/Combined_Action_Distance
CrossTask/Local/Far_Neg_Weight_Mean, Effective_Negative_Mass
Raw/Weighted_CrossTask/Local/Far_Margin/Acc
```

`Order_Margin` uses the count-balanced order family containing a contiguous
block swap and a full derangement. Per-corruption metrics remain available in
the CSV so the structured and destructive order tests can be studied
separately.
Scale 1 has no order negative. Shared-gradient diagnostics run every 200
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
