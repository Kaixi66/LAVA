# LAVA V6.3

Experiment: `robotwin10_lava_v6.3_1h100nvl`.

V6.3 removes the fixed time coordinate and changes signature normalization to
avoid amplifying isolated near-zero first- or second-level features. It
retains the current-observation FiLM architecture introduced in V6.2. Time
removal and normalization replacement form one combined experiment, so its
SR comparison cannot isolate their individual effects.

## Representation and candidates

- Eight ordered queries, 16 output dimensions each: 128-dimensional states.
- Encode frames with a shared encoder, then subtract adjacent states. For a
  fixed anchor context c, `r_k = Phi(Z_k; c) - Phi(Z_(k-1); c)`, so increments
  sum to `Phi(Z_L; c) - Phi(Z_0; c)`.
- FiLM uses pooled DINO patches from the current policy observation. Every
  positive, local, far and batch candidate for a row uses the row's context.
  Candidate-row activation checkpointing limits retained encoder activations.
- Scales are `[1,2,4,8,16]`. Spatial depth-2 LogSig has
  `128 + 128*127/2 = 8256` coordinates, compared with V6.2's 8385 with time.
- L=1 has an effective first level and a zero-padded second level. L>=2
  retains both levels.
- Same-scale batch negatives are independent candidates, including same-task
  paths. Batch exponential mass is summed. Local/far candidates and the
  pooled block-swap/full-derangement order family at L>=4 are unchanged.
  Action similarity weighting is disabled.

## Normalization

Mode: `ema_rms_soft`, implemented by `EMASoftLogSignatureNormalizer`.

For each modality (action/world), signature level and temporal scale, use:

```text
batch_energy = mean_positive_paths(sum_coordinates(S**2))
v = 0.99 * v + 0.01 * batch_energy
normalized_S = S / sqrt(sum_coordinates(S**2) + stop_gradient(v))
```

Squared-energy buffers start at 1.0 and use the moving-average update from
the first batch, including an all-zero first batch. Buffers and normalization
arithmetic remain FP32 when the model is cast to BF16. Updates use positive
paths only, once per loss forward. All candidates capture the same reference
statistics; backward checkpoint recomputation neither updates them nor reads
later forward updates. EMA buffers are saved in the model state dict.

```text
K = 1 if L == 1 else 2
signature = concat(normalized_D, normalized_A) / sqrt(K)
```

The structural L=1 second level is excluded from EMA updates and scoring.
There is no final unit/cosine normalization and no additional second-level
weight. Scoring uses signature dot products, temperature 0.07 and
lambda_LAVA 0.01. Weak signatures can stay weak. Raw auxiliary loss values
are not directly comparable to the previous normalization objective.

## Training and evaluation

Use [the V6.3 configuration](../configs/robotwin_lava_v63.yaml). It records the
production RoboTwin 10-task subset settings: seed42, batch128, 12 epochs
(54,288 optimizer steps for that dataset), LR 2e-4 to 5e-5, LAVA ratio 0.125,
task/episode balancing and 5% LAVA warmup. Dataset, conditioning, statistics
and DINO paths must point to the assets in the local workspace.

From the repository root in the training environment:

```bash
python train.py \
  --config configs/robotwin_lava_v63.yaml \
  --norm_stats_path /fs/cml-projects/WAM/data/robotwin_200_10_assets/stat-local-200-10.json \
  --save_dir /fs/cml-projects/WAM/runs/LAVA/robotwin/stage1/robotwin10_lava_v6.3_1h100nvl
```

This starts a new training run without loading V6.2 weights. Production
scheduling records submitted on 2026-09-12 were:

| Job | Resource | Dependency |
|---|---|---|
| V6.3 training 7508179 | 1 H100 NVL | afterany:7497586 (V6.2) |
| V6.3 evaluation 7508180 | 1 L40S | afterok:7508179 |

Evaluation selects epoch12 and runs 50 episodes for each of the 10 tasks
with video recording. Exact cluster submission scripts are archived at
[training](../scripts/slurm/archive/v63/train_robotwin_lava_v63_stage1_1_h100_nvl.sbatch)
and [evaluation](../scripts/slurm/archive/v63/eval_robotwin10_lava_v63_1_l40s.sbatch).
They reference the original CML workspace, Slurm accounts, environments and
pre-created immutable `provenance/source` snapshot; they do not create those
assets on a fresh checkout. The source manifest is archived alongside them
for identifying the original run. It also inventories auxiliary files copied
into that snapshot which are not required for RoboTwin V6.3 training.

## Validation

The focused regression suite passed 81 tests before production submission:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python -m pytest -q tests/test_lava_soft_normalization.py \
  tests/test_lava_film.py tests/test_lava_batch_mean.py tests/test_lava.py
```

Coverage includes static/tiny signatures at every scale, BF16 behavior,
FP32 EMA preservation, checkpoint round trips, frozen references during
activation recomputation, FiLM context mapping and negative aggregation.
The shared implementation includes an optional batch-mean reduction tested
for compatibility; V6.3 uses the default sum reduction.

Real-data validation job7508130 completed 200 full-model steps with batch128,
16 data workers and lambda_LAVA=0.01 from the first step (a stress check;
production retains its warmup). All 200 steps had active LAVA and FiLM
gradients; peak allocated memory was approximately 24.0 GiB on L40S.

Replay job7508131 evaluated the same 64 real L=2 batches (1,068 anchors per
checkpoint), including the previously identified degenerate local/far cases.
With fixed V6.2 epoch2 weights, the maximum LAVA branch gradient decreased
from 5.853 under the original objective to 0.124 under the V6.3 objective.
This used a newly initialized, updating EMA and no optimizer steps. These
bounded checks target the observed normalization mechanism; they do not
establish full-run stability, order-negative validity or an SR improvement.
