# V6.6 projected third-order LogSignature

V6.6 starts from V6.5 action8. It keeps the global 129-dimensional, depth-2
LogSignature, current-frame FiLM, time channel, graded-soft normalization,
negative squared L2 score, and episode-balanced 4+4 same-scale negatives.

For each action or world increment, a fixed shared 128-to-31 orthogonal
projection mixes all ordered query coordinates. The time increment is appended
unchanged, producing a 32-dimensional path. The exact degree-3 log-signature
tensor of that projected path is appended to the existing 8,385-dimensional
readout. The tensor has 32^3 = 32,768 coordinates; it is a redundant embedding
of the 10,912-dimensional free-Lie level, chosen to avoid basis conversion.
The resulting score vector has 41,153 coordinates. The fixed Hadamard
projection is stored as an int8 model buffer, so BF16 model conversion does
not quantize its orthogonal matrix. It adds no trainable parameters.

The existing graded-soft path scale `s` is retained. The added coordinates
are `8 * log3(projected_path) / s^3`. The factor 8 was selected before the
production run: factor 1 had negligible initial action energy (~0.004%).
This is a fixed experimental scale, not tuned on task success rates.

Training CSV monitoring adds third-order raw/calibrated norms, action/world
energy fractions, fractions by scale, positive/negative score contributions,
and candidate accuracy with and without the added third-order coordinates.
The existing flow, LAVA branch gradient, FiLM gradient, GPU memory, update
time, and task/scale monitors remain available.

Validation: `tests/test_lava_projected_third.py` checks the exact two-segment
BCH formula, zero third level for a one-segment path, finite backpropagation,
and the full candidate objective. A 10-step real RoboTwin batch run at full
LAVA lambda used the production source on A100 (job 7580191): completed 10/10,
peak allocated memory 20.6 GB, finite losses and gradients. Its 10-batch
candidate-accuracy gain averaged +0.008; this is only a pipeline check, not
evidence of policy improvement. Full training uses seed 42 and 12 epochs.
