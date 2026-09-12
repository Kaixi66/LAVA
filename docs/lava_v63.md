# Revised LAVA V6.3

Experiment: `robotwin10_lava_v6.3_1h100nvl`. This revision combines the provided
[geometry specification](specifications/Codex_A_Geometry.md) and
[negative specification](specifications/Codex_B_Negatives.md). The user
explicitly requested replacing the original V6.3 training and publishing the
revision. It starts from scratch, preserving old results in an archive.

Baseline commit: `7bcfc1e2c480e50e3707ac18c8a8a13521667bda`. Its complete config
is retained at `configs/ablations/robotwin_lava_v63_original.yaml`.
Old normalizers, dot scoring and mixed-batch behavior remain available.
Checkpoint loading stays strict; cross-normalizer weight migration is not
implemented.

## Geometry and matching

Keep 8 ordered queries × 16 dimensions, current-observation FiLM, the action
projector and shared encode-before-subtract world residuals. Every frame in
a path uses the row anchor's fixed context c:

```text
r_k = Phi(Z_k; c) - Phi(Z_(k-1); c)
sum(r_k) = Phi(Z_L; c) - Phi(Z_0; c)
```

The new `graded_soft` readout uses raw first-level D and the existing upper
triangle of the antisymmetric second-level A, without double-counting area:

```text
s = (rho^4 + ||D||_2^4 + ||A||_2^2)^(1/4)
z = concat(D / s, A / s^2)
rho = 1.0
```

Each action path and world candidate computes its own differentiable s.
Both modalities use the same function and fixed finite positive rho. The
readout runs in FP32 with autocast disabled. There is no EMA, detached scale,
per-level or final unit normalization, sqrt(2) divisor, or extra area weight.
Rho specifies a latent scale, not a measured physical noise threshold.

The first implementation requires state-delta encoding without time. Depth2
has 8256 coordinates. At L=1, area is zero-padded. Depth1 returns only D/s
and its scale ignores A. Residual telescoping remains valid; independently
normalized windows do not directly obey the raw Chen composition identity.

The independent `signature_score` switch defaults to `dot`. `neg_l2` uses:

```text
score(a, w) = -sum_coordinates((a.float() - w.float())^2)
loss = cross_entropy(scores / temperature, positive_index)
```

It applies to all positive, batch, local, far and order matching before any
existing family reductions. Matching similarity/margin diagnostics use the
selected scorer and are not cosine values in distance mode. Pure
representation dot-product diagnostics in the legacy branch retain their
original definitions.

Temperature stays 0.07. Because `-||a-w||^2 = 2*a.w - ||a||^2 - ||w||^2`,
the score-only ablation changes the dot coefficient as well as the
candidate-norm term. It tests the scoring rule as a whole, not an isolated
norm penalty. No temperature search is launched.

## Episode-balanced candidates

The existing positive interval sampler is retained. With observation t_obs,
relative start start and scale L, the positive begins at p=t_obs+start and
contains L+1 frames.

- Same episode: uniformly sample up to four distinct legal starts n with
  `0 <= n <= T-L-1` and `abs(n-p) >= 2L`. There are no local/far buckets,
  relaxed exclusions, replacement or extra spacing between negatives.
- Cross episode: uniformly select up to four unique same-scale positive
  paths already in the batch, excluding the anchor's physical episode.
  Deduplicate `(episode_uid, absolute_start, scale)`. Canonical HDF5 paths
  identify episodes, including their task/clean/randomized directory.
  Other episodes of the same task are allowed.
- The denominator has exactly one positive plus 0..4 same-episode and 0..4
  cross-episode independent logits. There are no order, old local/far,
  unselected batch candidates, family averages or source weights.
- Shortages reduce candidate counts. An anchor with no negative contributes
  no contrastive term; the loss averages only eligible anchors. If the entire
  batch lacks candidates, return a connected zero. Every positive remains
  encoded and can be another row's candidate. All policy flow/future losses
  remain active.

All candidates use the ROW anchor's current observation context for every
frame. Cross candidates reuse frozen DINO features and are decoded again
under that context. Random selection happens outside activation checkpoint
recomputation. Episode IDs and absolute starts never enter the network.

B-only EMA normalization still updates once from all sampled positives,
including skipped rows, and shares the captured reference across candidates.
Diagnostics distinguish sampled, encoded and scored anchors, missing-source
reasons, actual same/cross counts and availability. Health monitoring checks
accounting and autograd connectivity.

This changes the negative scheme as a whole: source mix, total count,
temporal exclusion, order and shortage handling. It does not guarantee
semantic negativity or isolate which rule influences SR.

## Complete configurations

Every YAML below is complete, with no implicit inheritance. Paths are under
`configs/ablations/`.

| File | Normalizer | Score | Negatives | Order |
|---|---|---|---|---|
| `robotwin_lava_v63_original.yaml` | ema_rms_soft | dot | mixed_batch | on |
| `robotwin_lava_v63_norm_only.yaml` | graded_soft | dot | mixed_batch | on |
| `robotwin_lava_v63_score_only.yaml` | ema_rms_soft | neg_l2 | mixed_batch | on |
| `robotwin_lava_v63_geometry.yaml` | graded_soft | neg_l2 | mixed_batch | on |
| `robotwin_lava_v63_negatives.yaml` | ema_rms_soft | dot | episode_balanced | off |
| `robotwin_lava_v63_order_off.yaml` | ema_rms_soft | dot | mixed_batch | off |
| `robotwin_lava_v63_combined.yaml` | graded_soft | neg_l2 | episode_balanced | off |

Ablations have distinct checkpoint tags. The original config retains its
historical tag for identification; choose a fresh output directory when
rerunning it. `configs/robotwin_lava_v63.yaml` equals the combined config
except for the original experiment tag. Only that combined production
experiment is submitted, not an ablation sweep.

All retain seed42, batch128, 12 epochs, LR 2e-4 to 5e-5, temperature0.07,
lambda0.01, 5% LAVA warmup, scales1/2/4/8/16, LAVA ratio0.125 and existing
task/episode balance. Dataset, statistics and DINO paths need local assets.

```bash
python train.py --config configs/robotwin_lava_v63.yaml \
  --norm_stats_path /fs/cml-projects/WAM/data/robotwin_200_10_assets/stat-local-200-10.json \
  --save_dir /fs/cml-projects/WAM/runs/LAVA/robotwin/stage1/robotwin10_lava_v6.3_1h100nvl
```

## Validation and deployment

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python -m pytest -q tests/test_lava_geometry.py \
  tests/test_lava_episode_balanced.py tests/test_lava_soft_normalization.py \
  tests/test_lava_film.py tests/test_lava_batch_mean.py tests/test_lava.py \
  tests/test_lava_dataset_pipeline.py
```

Tests cover published-code loss/gradient compatibility, exact graded formula,
zero/tiny paths, weak area, depth1/2, distance ranking, BF16, row context,
checkpoint recomputation, strict state loading, HDF5/collate metadata,
exact CE counts, shortages and unchanged policy flow/future losses.
DINO extraction remains frozen and inference code is unchanged.

A 200-step real-data short run at full lambda, batch128 and 16 workers is
required before production submission. Actual test, short-training and
scheduling results are recorded in
[validation/lava_v63_revised.json](validation/lava_v63_revised.json).
These bounded checks do not establish SR benefit or full-run stability.

Original train7508179 and eval7508180 were cancelled at the user's request.
Its valid results and immutable source are preserved at
`runs/LAVA/superseded/v63_ema_rms_soft_7508179_20260912/robotwin10_lava_v6.3_1h100nvl`.
The replacement uses a fresh directory at the original run path. Batchmean
7497812 follows the replacement; a new L40S eval requires successful training.

Original scripts remain at `scripts/slurm/archive/v63/`. Revised scripts and
source hashes are archived at `scripts/slurm/archive/v63_revised/`. They
reference CML-specific accounts, environments and a pre-created immutable
source snapshot; they do not provision a fresh checkout's external assets.
