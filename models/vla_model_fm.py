import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Standard Transformer SinCos positional encoding.
    Returns: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)


# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VisualFeatureAdapter(nn.Module):
    """Perceiver-style adapter: learnable queries cross-attend to vision tokens, output (B, num_queries, hidden_dim)."""
    def __init__(self, feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        self.input_norm = nn.LayerNorm(feat_dim)
        self.feature_proj = nn.Linear(feat_dim, hidden_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, features):
        """
        features: (B, N, feat_dim)
        Returns: (B, num_queries, hidden_dim)
        """
        B = features.shape[0]
        memory = self.input_norm(features)
        memory = self.feature_proj(memory)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.transformer_decoder(tgt, memory)
        return out


class MultiLayerConcatFusion(nn.Module):
    """
    Multi-layer DINO feature fusion (concat mode):
      - Each layer is normalized by its own LayerNorm first (feature norms differ
        a lot across DINOv3 layers; shallow layers would be drowned out otherwise)
      - Concatenate token-wise along the feature dim -> (B, N, L*feat_dim)
      - Linear / MLP projection down to -> (B, N, out_dim)
    The output is then fed into a single VisualFeatureAdapter.
    """
    def __init__(self, feat_dim, num_layers, out_dim, proj_type="linear", pre_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        if pre_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        in_dim = feat_dim * num_layers
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported concat proj_type: {proj_type}")

    def forward(self, feats_list):
        """feats_list: List[(B, N, feat_dim)] (all layers must share the same N)"""
        assert len(feats_list) == self.num_layers, \
            f"MultiLayerConcatFusion expects {self.num_layers} layers, got {len(feats_list)}"
        if self.pre_norm:
            feats_list = [ln(f) for ln, f in zip(self.layer_norms, feats_list)]
        x = torch.cat(feats_list, dim=-1)   # (B, N, L*feat_dim)
        return self.proj(x)                 # (B, N, out_dim)


class FutureFeatureDecoder(nn.Module):
    """
    Future-frame feature prediction decoder (Transformer):
      - Learnable query tokens as Q
      - cond tokens produced by the action head as KV (cross-attention)
      - Output projected to the target feature dim (= DINO hidden_size), supervised
        against future-frame patch features with a cosine loss

    Number of queries = number of future-frame patch tokens (dense prediction,
    one query per patch position).
    """
    def __init__(self, num_queries, hidden_dim, out_dim,
                 num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.out_dim = out_dim

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # Normalize cond tokens before they enter cross-attention
        self.kv_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_tokens):
        """
        cond_tokens: (B, M, hidden_dim) — observation tokens evolved by the DiT
        Returns: (B, num_queries, out_dim) predicted future-frame patch features
        """
        B = cond_tokens.shape[0]
        memory = self.kv_norm(cond_tokens)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.decoder(tgt, memory)
        out = self.out_proj(self.out_norm(out))
        return out


class WorldResidualEncoder(nn.Module):
    """Encode DINO patches; legacy pooled residuals or ordered per-frame state queries."""

    def __init__(self, feat_dim, residual_dim, hidden_dim=256,
                 num_queries=1, num_layers=2, num_heads=4, query_dim=None,
                 use_query_film=False):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        if query_dim is not None and residual_dim != num_queries * query_dim:
            raise ValueError("State latent dimension must equal num_queries * query_dim")
        self.input_norm = nn.LayerNorm(feat_dim)
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.query_embed = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, residual_dim if query_dim is None else query_dim)
        self.use_query_film = bool(use_query_film)
        if self.use_query_film:
            # Module construction happens on CPU. Do not shift initialization
            # of the existing action projector when enabling a zero-init FiLM.
            with torch.random.fork_rng(devices=[]):
                self.context_norm = nn.LayerNorm(feat_dim)
                self.film = nn.Linear(feat_dim, 2 * hidden_dim)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)

    def forward(self, feature_differences, context=None):
        """Input is (B, patches, dino_dim): frame states in V6, differences in legacy mode."""
        batch_size = feature_differences.shape[0]
        memory = self.input_proj(self.input_norm(feature_differences))
        queries = self.query_embed.expand(batch_size, -1, -1)
        if self.use_query_film:
            if context is None or context.shape != (batch_size, feature_differences.shape[-1]):
                raise ValueError("query_film requires one current-frame context per encoded frame")
            gamma, beta = self.film(self.context_norm(context)).chunk(2, dim=-1)
            queries = queries * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        elif context is not None:
            raise ValueError("Context supplied to an unconditional world encoder")
        residual_queries = self.decoder(queries, memory)
        state_queries = self.output_norm(residual_queries)
        if self.query_dim is None:
            # Legacy checkpoint architecture: mean queries before projection.
            return self.output_proj(state_queries.mean(dim=1))
        # Keep ordered query slots. Project and subtract in FP32 even during
        # BF16 training: small state changes should not vanish in output rounding.
        with torch.autocast(device_type=state_queries.device.type, enabled=False):
            states = F.linear(state_queries.float(), self.output_proj.weight.float(),
                              self.output_proj.bias.float())
        return states.flatten(1)



def sample_full_shuffle_permutation(length, device=None):
    """Sample an order-negative permutation that moves every position."""
    if length < 2:
        raise ValueError(f"Full shuffle requires length >= 2, got {length}")
    if length == 2:
        return torch.tensor([1, 0], device=device, dtype=torch.long)

    # Sample on CPU so rejection does not introduce repeated GPU sync points.
    identity = torch.arange(length)
    while True:
        permutation = torch.randperm(length)
        if torch.all(permutation != identity):
            return permutation.to(device=device)


def sample_contiguous_block_swap_permutation(length, device=None):
    """Swap two adjacent, non-empty contiguous blocks while preserving block order."""
    if length < 2:
        raise ValueError(f"Block swap requires length >= 2, got {length}")
    if length == 2:
        return torch.tensor([1, 0], device=device, dtype=torch.long)

    # [0:a] [a:b] [b:c] [c:L] -> [0:a] [b:c] [a:b] [c:L]
    cuts = torch.randperm(length + 1)[:3].sort().values.tolist()
    while cuts[0] == cuts[1] or cuts[1] == cuts[2]:
        cuts = torch.randperm(length + 1)[:3].sort().values.tolist()
    a, b, c = cuts
    if a == b or b == c:
        raise RuntimeError("Failed to sample two non-empty blocks")
    permutation = torch.cat((
        torch.arange(0, a),
        torch.arange(b, c),
        torch.arange(a, b),
        torch.arange(c, length),
    ))
    if torch.equal(permutation, torch.arange(length)):
        raise RuntimeError("Contiguous block swap unexpectedly produced identity")
    return permutation.to(device=device, dtype=torch.long)


def _raw_logsignature_levels(path_increments):
    """Return the unnormalized depth-1 and antisymmetric depth-2 levels."""
    increments = path_increments.float()
    level_one = increments.sum(dim=0)
    prefix = increments.cumsum(dim=0) - increments
    area = 0.5 * torch.einsum("li,lj->ij", prefix, increments)
    area = area - area.transpose(0, 1)
    upper = torch.triu_indices(
        area.shape[0], area.shape[1], offset=1, device=area.device)
    level_two = area[upper[0], upper[1]]
    return level_one, level_two


def projected_logsignature_level_three(path_increments, projection):
    """Exact degree-3 log-signature tensor after a fixed spatial projection.

    The tensor coordinates are redundant (D**3 rather than the free-Lie
    dimension), but retain every third-order cross-coordinate interaction.
    The final path coordinate is the unprojected time channel.
    """
    if path_increments.ndim != 2 or path_increments.shape[1] != projection.shape[0] + 1:
        raise ValueError("Projected level three expects spatial increments plus time")
    with torch.autocast(device_type=path_increments.device.type, enabled=False):
        spatial_projection = projection.float()
        if projection.dtype == torch.int8:
            spatial_projection = spatial_projection / math.sqrt(projection.shape[0])
        spatial = path_increments[:, :-1].float() @ spatial_projection
        increments = torch.cat((spatial, path_increments[:, -1:].float()), dim=-1)
        d = increments.shape[-1]
        one = increments.new_zeros(d)
        two = increments.new_zeros(d, d)
        three = increments.new_zeros(d, d, d)
        for x in increments.unbind(0):
            xx = torch.outer(x, x)
            three = (three + torch.einsum("ij,k->ijk", two, x)
                     + 0.5 * torch.einsum("i,jk->ijk", one, xx)
                     + torch.einsum("i,jk->ijk", x, xx) / 6.0)
            two = two + torch.outer(one, x) + 0.5 * xx
            one = one + x
        log_three = (three
                     - 0.5 * (torch.einsum("i,jk->ijk", one, two)
                              + torch.einsum("ij,k->ijk", two, one))
                     + torch.einsum("i,j,k->ijk", one, one, one) / 3.0)
        if increments.shape[0] == 1:
            log_three = torch.zeros_like(log_three)
        return log_three.reshape(-1)


def graded_soft_logsignature(level_one, level_two, scale, depth=2, rho=1.0):
    """One differentiable path scale, with homogeneous scaling by level."""
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("graded_soft rho must be finite and positive")
    if depth not in (1, 2):
        raise ValueError("LogSig depth must be 1 or 2")
    with torch.autocast(device_type=level_one.device.type, enabled=False):
        delta, area = level_one.float(), level_two.float()
        if depth == 1 or int(scale) == 1:
            area = torch.zeros_like(area)
        d2 = delta.square().sum(dim=-1, keepdim=True)
        a2 = area.square().sum(dim=-1, keepdim=True)
        s2 = (rho ** 4 + d2.square() + a2).sqrt()
        one, two = delta / s2.sqrt(), area / s2
        signature = one if depth == 1 else torch.cat((one, two), dim=-1)
        one_energy = one.detach().square().sum(dim=-1)
        two_energy = two.detach().square().sum(dim=-1)
        return signature, {
            "level1_calibrated_norm": one_energy.sqrt(),
            "level2_calibrated_norm": two_energy.sqrt(),
            "level2_energy_fraction": two_energy / (one_energy + two_energy).clamp_min(1e-12),
            "level1_ema_rms": one.new_tensor(float("nan")),
            "level2_ema_rms": one.new_tensor(float("nan")),
        }


def lava_signature_score(action, world, score_type="dot"):
    """Paired/broadcast signature score; neg_l2 sums coordinates in FP32."""
    if score_type == "dot":
        return (action * world).sum(dim=-1)
    if score_type != "neg_l2":
        raise ValueError("signature_score must be dot or neg_l2")
    with torch.autocast(device_type=action.device.type, enabled=False):
        return -(action.float() - world.float()).square().sum(dim=-1)


def sample_cross_episode_indices(episode_uids, absolute_starts, scales, count=4):
    """Sample unique same-scale paths, outside activation recomputation."""
    if not 0 <= count <= 4:
        raise ValueError("episode_balanced requires 0..4 cross-episode negatives")
    if not (len(episode_uids) == len(absolute_starts) == len(scales)):
        raise ValueError("Episode metadata lengths must match positive paths")
    if any(not isinstance(uid, str) or not uid for uid in episode_uids):
        raise ValueError("episode_uid must be a nonempty globally unique identifier")
    rows = []
    for uid, scale in zip(episode_uids, scales):
        unique = {}
        for j, (other, start, other_scale) in enumerate(zip(episode_uids, absolute_starts, scales)):
            if other != uid and int(other_scale) == int(scale):
                unique.setdefault((other, int(start), int(other_scale)), j)
        candidates = list(unique.values())
        permutation = torch.randperm(len(candidates)).tolist()
        rows.append([candidates[j] for j in permutation[:count]])
    return rows


def normalized_logsignature(path_increments, depth=2, eps=1e-6,
                            return_raw_norms=False):
    """Closed-form depth-1/2 log-signature of a piecewise-linear path.

    The first and second levels are normalized separately, concatenated, and
    normalized once more. This implementation is differentiable and avoids a
    dependency on signatory. ``path_increments`` has shape (L, D).
    """
    if depth not in (1, 2):
        raise ValueError(f"LAVA supports logsig_depth 1 or 2, got {depth}")
    if path_increments.ndim != 2 or path_increments.shape[0] < 1:
        raise ValueError(
            f"path_increments must have shape (L,D) with L >= 1, got {tuple(path_increments.shape)}")

    raw_level_one, raw_level_two = _raw_logsignature_levels(path_increments)
    level_one = F.normalize(raw_level_one, dim=0, eps=eps)
    if depth == 1:
        signature = level_one
    else:
        level_two = F.normalize(raw_level_two, dim=0, eps=eps)
        signature = F.normalize(
            torch.cat((level_one, level_two), dim=0), dim=0, eps=eps)

    if not return_raw_norms:
        return signature
    raw_norms = {
        "level1_raw_norm": raw_level_one.detach().norm(),
        "level2_raw_norm": raw_level_two.detach().norm(),
    }
    return signature, raw_norms


class EMALogSignatureCalibrator(nn.Module):
    """Population-scale calibration for variable-length depth-1/2 LogSigs.

    Unlike per-sample level normalization, this preserves whether an individual
    path has weak or strong second-order structure. Statistics are modality- and
    scale-specific buffers, so they are checkpointed and inherited by Stage 2.
    """

    _MODALITIES = {"action": 0, "world": 1}

    def __init__(self, scales=(1, 2, 4, 8, 16), momentum=0.99,
                 level2_weight=0.5, eps=1e-6):
        super().__init__()
        scales = tuple(int(scale) for scale in scales)
        if not scales or len(set(scales)) != len(scales):
            raise ValueError(f"Calibration scales must be unique and non-empty: {scales}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"EMA momentum must be in [0,1), got {momentum}")
        if level2_weight < 0:
            raise ValueError(f"Level-2 weight must be non-negative, got {level2_weight}")
        self.scales = scales
        self.scale_to_index = {scale: index for index, scale in enumerate(scales)}
        self.momentum = float(momentum)
        self.level2_weight = float(level2_weight)
        self.eps = float(eps)
        # [modality(action/world), level(1/2), scale]
        self.register_buffer(
            "ema_squared_norm", torch.ones(2, 2, len(scales), dtype=torch.float32))
        self.register_buffer(
            "ema_initialized", torch.zeros(2, 2, len(scales), dtype=torch.bool))

    def _indices(self, modality, scale):
        if modality not in self._MODALITIES:
            raise ValueError(f"Unknown LogSig modality: {modality}")
        scale = int(scale)
        if scale not in self.scale_to_index:
            raise ValueError(
                f"Scale {scale} is absent from calibrator scales {self.scales}")
        return self._MODALITIES[modality], self.scale_to_index[scale]

    @torch.no_grad()
    def update(self, modality, level_ones, level_twos, scales):
        """Update detached FP32 block-energy statistics from positive paths."""
        if not (len(level_ones) == len(level_twos) == len(scales)):
            raise ValueError("LogSig calibration inputs must have matching lengths")
        modality_index = self._MODALITIES[modality]
        for scale in sorted(set(int(value) for value in scales)):
            scale_index = self.scale_to_index[scale]
            selected = [index for index, value in enumerate(scales)
                        if int(value) == scale]
            level_values = (
                torch.stack([level_ones[index].detach().float() for index in selected]),
                torch.stack([level_twos[index].detach().float() for index in selected]),
            )
            for level_index, values in enumerate(level_values):
                if level_index == 1 and scale < 2:
                    continue
                batch_squared_norm = values.square().sum(dim=-1).mean()
                target = self.ema_squared_norm[modality_index, level_index, scale_index]
                initialized = self.ema_initialized[
                    modality_index, level_index, scale_index]
                ema_value = (
                    target * self.momentum
                    + batch_squared_norm * (1.0 - self.momentum))
                target.copy_(torch.where(
                    initialized, ema_value,
                    batch_squared_norm.clamp_min(self.eps ** 2)))
                initialized.fill_(True)

    def forward(self, level_one, level_two, modality, scale, depth=2):
        modality_index, scale_index = self._indices(modality, scale)
        level_one_rms = self.ema_squared_norm[
            modality_index, 0, scale_index].detach().sqrt().clamp_min(self.eps)
        calibrated_one = level_one.float() / level_one_rms
        if depth == 1:
            return F.normalize(calibrated_one, dim=0, eps=self.eps), {
                "level1_calibrated_norm": calibrated_one.detach().norm(),
                "level2_calibrated_norm": calibrated_one.new_tensor(0.0),
                "level2_energy_fraction": calibrated_one.new_tensor(0.0),
                "level1_ema_rms": level_one_rms.detach(),
                "level2_ema_rms": calibrated_one.new_tensor(float("nan")),
            }

        level_two_rms = self.ema_squared_norm[
            modality_index, 1, scale_index].detach().sqrt().clamp_min(self.eps)
        calibrated_two = (
            level_two.float() / level_two_rms * self.level2_weight)
        joined = torch.cat((calibrated_one, calibrated_two), dim=0)
        energy_one = calibrated_one.square().sum()
        energy_two = calibrated_two.square().sum()
        energy_fraction = energy_two / (energy_one + energy_two).clamp_min(self.eps ** 2)
        return F.normalize(joined, dim=0, eps=self.eps), {
            "level1_calibrated_norm": calibrated_one.detach().norm(),
            "level2_calibrated_norm": calibrated_two.detach().norm(),
            "level2_energy_fraction": energy_fraction.detach(),
            "level1_ema_rms": level_one_rms.detach(),
            "level2_ema_rms": level_two_rms.detach(),
        }


class EMASoftLogSignatureNormalizer(EMALogSignatureCalibrator):
    """V6.3: positive-population RMS with smooth, bounded level features.

    A zero first batch decays the unit initial reference instead of replacing
    it with epsilon. Individual zero paths retain finite derivatives. There is
    deliberately no final unit normalization of the concatenated signature.
    """

    def __init__(self, scales=(1, 2, 4, 8, 16), momentum=0.99):
        super().__init__(scales=scales, momentum=momentum, level2_weight=1.0)

    def _apply(self, fn, recurse=True):
        # model.to(bfloat16) must not quantize accumulated population energy.
        original_energy = self.ema_squared_norm
        result = super()._apply(fn, recurse=recurse)
        self.ema_squared_norm = original_energy.to(
            device=self.ema_squared_norm.device, dtype=torch.float32)
        return result

    @torch.no_grad()
    def update(self, modality, level_ones, level_twos, scales):
        if not (len(level_ones) == len(level_twos) == len(scales)):
            raise ValueError("Calibration inputs must have matching lengths")
        for scale in sorted(set(int(value) for value in scales)):
            modality_index, scale_index = self._indices(modality, scale)
            selected = [i for i, value in enumerate(scales) if int(value) == scale]
            groups = (level_ones,) if scale == 1 else (level_ones, level_twos)
            with torch.autocast(device_type=self.ema_squared_norm.device.type,
                                enabled=False):
                energies = torch.stack([
                    torch.stack([group[i].detach().float() for i in selected])
                    .square().sum(dim=-1).mean() for group in groups])
                reference = self.ema_squared_norm[modality_index, :len(groups), scale_index]
                updated = reference * self.momentum + energies * (1.0 - self.momentum)
                if not bool((torch.isfinite(updated) & (updated > 0)).all()):
                    raise FloatingPointError(
                        f"Degenerate EMA LogSig energy: modality={modality}, scale={scale}")
                reference.copy_(updated)
                self.ema_initialized[modality_index, :len(groups), scale_index] = True

    def forward(self, level_one, level_two, modality, scale, depth=2, reference=None):
        if depth not in (1, 2):
            raise ValueError(f"LAVA supports logsig_depth 1 or 2, got {depth}")
        modality_index, scale_index = self._indices(modality, scale)
        with torch.autocast(device_type=level_one.device.type, enabled=False):
            # clone isolates this graph's reference even if another forward
            # updates the buffer before its backward pass.
            population = self.ema_squared_norm if reference is None else reference
            energy = population[modality_index, :, scale_index].detach().clone()
            one = level_one.float()
            one = one / (one.square().sum() + energy[0]).sqrt()
            two = torch.zeros_like(level_two, dtype=torch.float32)
            two_rms = one.new_tensor(float("nan"))
            if depth == 2 and int(scale) >= 2:
                two = level_two.float()
                two = two / (two.square().sum() + energy[1]).sqrt()
                two_rms = energy[1].sqrt()
            if depth == 1:
                signature = one
            else:
                signature = torch.cat((one, two), dim=0)
                if int(scale) >= 2:
                    signature = signature / (2.0 ** 0.5)
            one_energy = one.detach().square().sum()
            two_energy = two.detach().square().sum()
            return signature, {
                "level1_calibrated_norm": one.detach().norm(),
                "level2_calibrated_norm": two.detach().norm(),
                "level2_energy_fraction": (
                    two_energy / (one_energy + two_energy).clamp_min(self.eps ** 2)),
                "level1_ema_rms": energy[0].sqrt(),
                "level2_ema_rms": two_rms,
            }


class EMAActionDistanceCalibrator(nn.Module):
    """Family-balanced, per-scale action-distance calibration for V5."""

    def __init__(self, scales=(1, 2, 4, 8, 16), momentum=0.99,
                 multiplier=1.0, eps=1e-8):
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.scale_to_index = {scale: index for index, scale in enumerate(self.scales)}
        self.momentum = float(momentum)
        self.multiplier = float(multiplier)
        self.eps = float(eps)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("Action-distance EMA momentum must be in [0,1)")
        if self.multiplier <= 0.0:
            raise ValueError("Action-distance beta multiplier must be positive")
        self.register_buffer("ema_beta", torch.ones(len(self.scales), dtype=torch.float32))
        self.register_buffer("ema_initialized", torch.zeros(len(self.scales), dtype=torch.bool))

    @torch.no_grad()
    def update(self, scale, family_distances):
        """Equal-weight available family medians, independent of family counts."""
        index = self.scale_to_index[int(scale)]
        medians = []
        for distances in family_distances:
            if distances is not None and distances.numel():
                medians.append(distances.detach().float().median())
        if not medians:
            return None
        aggregate = torch.stack(medians).mean().clamp_min(self.eps)
        if self.ema_initialized[index]:
            aggregate = (self.ema_beta[index] * self.momentum
                         + aggregate * (1.0 - self.momentum))
        self.ema_beta[index].copy_(aggregate)
        self.ema_initialized[index].fill_(True)
        return aggregate

    def beta(self, scale):
        index = self.scale_to_index[int(scale)]
        return (self.ema_beta[index].detach() * self.multiplier).clamp_min(self.eps)


class EMAActionComponentCalibrator(nn.Module):
    """Per-scale calibration that puts action descriptor components on comparable scales."""

    component_names = (
        "arm_state", "arm_change", "gripper_state", "gripper_change")

    def __init__(self, scales=(1, 2, 4, 8, 16), momentum=0.99, eps=1e-8):
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.scale_to_index = {scale: index for index, scale in enumerate(self.scales)}
        self.momentum = float(momentum)
        self.eps = float(eps)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("Action-component EMA momentum must be in [0,1)")
        shape = (len(self.scales), len(self.component_names))
        self.register_buffer("ema_beta", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("ema_initialized", torch.zeros(shape, dtype=torch.bool))

    @torch.no_grad()
    def update(self, scale, family_components):
        """Equal-weight family medians for every descriptor component.

        Each family tensor has shape ``(..., 4)``. A zero-only component stays
        uninitialized until it first carries a non-zero signal, preventing an
        epsilon initialization from amplifying a later non-zero batch.
        """
        scale_index = self.scale_to_index[int(scale)]
        for component_index in range(len(self.component_names)):
            medians = []
            for values in family_components:
                if values is None or not values.numel():
                    continue
                component = values[..., component_index].detach().float()
                component = component[torch.isfinite(component)]
                if component.numel():
                    medians.append(component.median())
            if not medians:
                continue
            aggregate = torch.stack(medians).mean()
            if not torch.isfinite(aggregate) or aggregate <= self.eps:
                continue
            initialized = self.ema_initialized[scale_index, component_index]
            target = self.ema_beta[scale_index, component_index]
            updated = (target * self.momentum
                       + aggregate * (1.0 - self.momentum))
            target.copy_(torch.where(initialized, updated, aggregate))
            initialized.fill_(True)
        return self.beta(scale)

    def beta(self, scale):
        scale_index = self.scale_to_index[int(scale)]
        values = self.ema_beta[scale_index].detach()
        initialized = self.ema_initialized[scale_index]
        # An unseen/zero-only component contributes its raw (normally zero)
        # value instead of receiving an arbitrary large rescaling.
        return torch.where(initialized, values.clamp_min(self.eps),
                           torch.ones_like(values))


def action_path_distance(left_normalized, left_raw, right_normalized, right_raw,
                         gripper_indices=(6, 13), arm_state_weight=0.5,
                         arm_change_weight=0.5, gripper_state_weight=0.5,
                         gripper_change_weight=0.5):
    """Detached RoboTwin action descriptor distance for two equal-length paths."""
    if left_normalized.shape != right_normalized.shape or left_raw.shape != right_raw.shape:
        raise ValueError("Action descriptor paths must have matching shapes")
    gripper_indices = tuple(int(index) for index in gripper_indices)
    action_dim = left_normalized.shape[-1]
    if any(index < 0 or index >= action_dim for index in gripper_indices):
        raise ValueError(f"Invalid gripper indices {gripper_indices} for action_dim={action_dim}")
    arm_indices = [index for index in range(action_dim) if index not in gripper_indices]
    left_arm = left_normalized[..., arm_indices].detach().float()
    right_arm = right_normalized[..., arm_indices].detach().float()
    left_arm_delta = torch.diff(left_arm, dim=0)
    right_arm_delta = torch.diff(right_arm, dim=0)
    left_grip = left_raw[..., list(gripper_indices)].detach().float()
    right_grip = right_raw[..., list(gripper_indices)].detach().float()
    arm_state = F.mse_loss(left_arm, right_arm)
    arm_change = F.mse_loss(left_arm_delta, right_arm_delta)
    grip_state = F.mse_loss(left_grip, right_grip)
    grip_change = F.mse_loss(torch.diff(left_grip, dim=0), torch.diff(right_grip, dim=0))
    combined = (float(arm_state_weight) * arm_state
                + float(arm_change_weight) * arm_change
                + float(gripper_state_weight) * grip_state
                + float(gripper_change_weight) * grip_change)
    return arm_state, arm_change, grip_state, grip_change, combined


def action_similarity_weight(distance, beta, min_weight=0.1):
    """V5 false-negative gate; a true negative asymptotically keeps weight 1."""
    return (float(min_weight) + (1.0 - float(min_weight))
            * (1.0 - torch.exp(-distance / beta.clamp_min(1e-8))))


def weighted_count_balanced_family_logit(values, weights, temperature):
    """Weighted log-sum-exp with count (not weight-sum) family balancing."""
    valid = torch.isfinite(values)
    counts = valid.sum(dim=1)
    scaled = (values / temperature
              + weights.clamp_min(1e-12).log()).masked_fill(~valid, -torch.inf)
    result = temperature * (
        torch.logsumexp(scaled, dim=1)
        - counts.clamp_min(1).float().log())
    return result.masked_fill(counts == 0, -torch.inf), counts


def future_feature_cosine_loss(pred, target):
    """
    Per-token cosine loss = mean(1 - cos_sim).
    pred / target: (B, M, D). Computed in float32 for numerical stability
    under bf16 training.
    """
    assert pred.shape == target.shape, \
        f"future-feat pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)} " \
        f"(check patch count: num_queries should equal the future-frame DINO patch token count)"
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    cos = (pred * target).sum(dim=-1)        # (B, M)
    return (1.0 - cos).mean()


class DiTBlock(nn.Module):
    """
    DiT block with Self-Attention + MLP, conditioned via adaLN-zero on time embedding.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, attn_mask=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb_t).chunk(6, dim=1)

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn1(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class VLAModel(nn.Module):
    """
    VLA model (DINOv3 version, no language input).

    Architecture:
    1. Action / Proprio / Register / multi-layer DINO tokens are concatenated
       and processed jointly by a DiT
    2. Bi-directional self-attention (Flow Matching predicts the whole chunk in parallel)
    3. State injection: placeholder zeros are replaced once at a given layer
       (avoids signal inflation from repeated accumulation)
    4. Multi-layer DINO features: per-layer independent adapters, or a single
       adapter after concat fusion
    5. Optional future-frame feature prediction: an auxiliary decoder predicts
       future DINO patch features from the evolved observation tokens
    """
    def __init__(self,
                 action_dim=14,
                 proprio_dim=16,
                 hidden_dim=512,
                 num_heads=4,
                 depth=12,
                 action_len=32,
                 proprio_len=1,
                 num_registers=2,
                 dino_feat_dims=(1024,),     # feat_dim of each extracted layer; len = number of layers
                 vlm_num_queries=64,         # number of queries per adapter
                 adapter_depth=2,
                 state_inject_start=0,
                 task_cond_dim=None,         # task condition vector dim (DINOv3 CLS difference feature, e.g. 1024); None = disabled
                 # --- Multi-layer feature fusion mode ---
                 fusion_mode="per_layer",    # "per_layer" (independent adapter per layer) | "concat" (concat + project, single adapter)
                 concat_proj_type="linear",  # projection type in concat mode: "linear" | "mlp"
                 concat_pre_norm=True,       # apply per-layer LayerNorm before concat in concat mode
                 concat_out_dim=None,        # projection target dim in concat mode; None -> = dino_feat_dims[0]
                 # --- Future feature prediction ---
                 use_future_feat=False,
                 future_feat_num_queries=None,        # = future-frame patch token count (dense prediction)
                 future_feat_out_dim=None,            # = DINO hidden_size; None -> = dino_feat_dims[0]
                 future_feat_depth=2,
                 future_feat_heads=4,
                 # --- LAVA (training-only auxiliary supervision) ---
                 use_lava=False,
                 lava_dino_feat_dim=None,
                 lava_residual_dim=32,
                 lava_world_encoding="feature_delta",
                 lava_time_channel=True,
                 lava_world_condition="none",
                 lava_query_dim=16,
                 lava_qformer_hidden_dim=256,
                 lava_qformer_num_queries=1,
                 lava_qformer_num_layers=2,
                 lava_qformer_num_heads=4,
                 lava_logsig_depth=2,
                 lava_projected_third_order=False,
                 lava_third_spatial_dim=31,
                 lava_third_weight=1.0,
                 lava_action_target_layer="final",
                 lava_signature_normalization="ema_calibrated",
                 lava_scales=(1, 2, 4, 8, 16),
                 lava_signature_ema_momentum=0.99,
                 lava_signature_level2_weight=0.5,
                 lava_signature_rho=1.0,
                 lava_signature_score="dot",
                 lava_action_similarity_weighting=False,
                 lava_batch_negative_reduction="sum",
                 lava_num_cross_episode_negatives=4,
                 lava_reuse_world_signatures=False,
                 lava_action_component_calibration=False,
                 lava_action_similarity_min_weight=0.1,
                 lava_action_similarity_beta_momentum=0.99,
                 lava_action_similarity_beta_multiplier=1.0,
                 lava_action_gripper_indices=(6, 13),
                 lava_action_arm_state_weight=0.5,
                 lava_action_arm_change_weight=0.5,
                 lava_action_gripper_state_weight=0.5,
                 lava_action_gripper_change_weight=0.5,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.num_registers = num_registers
        self.state_inject_start = state_inject_start
        self.num_dino_layers = len(dino_feat_dims)
        self.use_task_cond = task_cond_dim is not None

        self.fusion_mode = fusion_mode
        assert fusion_mode in ("per_layer", "concat"), f"Unknown fusion_mode: {fusion_mode}"

        self.use_future_feat = use_future_feat
        self.use_lava = use_lava
        self.lava_world_encoding = str(lava_world_encoding)
        if self.lava_world_encoding not in {"feature_delta", "state_delta"}:
            raise ValueError("lava_world_encoding must be feature_delta or state_delta")
        self.lava_time_channel = bool(lava_time_channel)
        self.lava_world_condition = str(lava_world_condition)
        if self.lava_world_condition not in {"none", "query_film"}:
            raise ValueError("lava_world_condition must be none or query_film")
        if self.lava_world_condition == "query_film" and self.lava_world_encoding != "state_delta":
            raise ValueError("query_film requires state_delta world encoding")
        self.lava_residual_dim = lava_residual_dim
        self.lava_logsig_depth = lava_logsig_depth
        self.lava_projected_third_order = bool(lava_projected_third_order)
        self.lava_third_spatial_dim = int(lava_third_spatial_dim)
        self.lava_third_weight = float(lava_third_weight)
        if self.lava_projected_third_order:
            if (self.lava_logsig_depth != 2 or not self.lava_time_channel
                    or self.lava_world_encoding != "state_delta"):
                raise ValueError("projected third order requires depth-2 state_delta with time channel")
            if not 1 <= self.lava_third_spatial_dim <= lava_residual_dim:
                raise ValueError("Invalid projected third-order spatial dimension")
            if not math.isfinite(self.lava_third_weight) or self.lava_third_weight <= 0:
                raise ValueError("Projected third-order weight must be finite and positive")
            if lava_residual_dim & (lava_residual_dim - 1):
                raise ValueError("Fixed third-order projection requires power-of-two residual dimension")
            generator = torch.Generator(device="cpu").manual_seed(66408)
            hadamard = torch.ones(1, 1, dtype=torch.int8)
            while hadamard.shape[0] < lava_residual_dim:
                hadamard = torch.cat((torch.cat((hadamard, hadamard), dim=1),
                                      torch.cat((hadamard, -hadamard), dim=1)), dim=0)
            row_order = torch.randperm(lava_residual_dim, generator=generator)
            column_order = torch.randperm(lava_residual_dim, generator=generator)
            projection = hadamard[row_order][:, column_order[:self.lava_third_spatial_dim]].contiguous()
            self.register_buffer("lava_third_projection", projection, persistent=True)
        self.lava_signature_normalization = str(
            lava_signature_normalization).lower()
        self.lava_signature_rho = float(lava_signature_rho)
        self.lava_signature_score = str(lava_signature_score)
        if self.lava_signature_score not in {"dot", "neg_l2"}:
            raise ValueError("signature_score must be dot or neg_l2")
        if self.lava_signature_normalization == "graded_soft":
            if self.lava_world_encoding != "state_delta":
                raise ValueError("graded_soft requires state_delta")
            if not math.isfinite(self.lava_signature_rho) or self.lava_signature_rho <= 0:
                raise ValueError("graded_soft rho must be finite and positive")
        if self.lava_signature_normalization not in {
                "per_level_unit", "ema_calibrated", "ema_rms_soft", "graded_soft"}:
            raise ValueError(
                "lava_signature_normalization must be per_level_unit, "
                f"ema_calibrated, ema_rms_soft or graded_soft, got {self.lava_signature_normalization}")
        self.lava_action_similarity_weighting = bool(
            lava_action_similarity_weighting)
        self.lava_batch_negative_reduction = str(lava_batch_negative_reduction)
        self.lava_num_cross_episode_negatives = int(lava_num_cross_episode_negatives)
        if not 0 <= self.lava_num_cross_episode_negatives <= 4:
            raise ValueError("episode_balanced requires 0..4 cross-episode negatives")
        # Without query FiLM a candidate's world signature is anchor-independent, so
        # episode_balanced can reuse positive signatures for cross candidates and
        # encode all same-episode paths in one batch (numerically the same objective).
        self.lava_reuse_world_signatures = bool(lava_reuse_world_signatures)
        if self.lava_reuse_world_signatures and self.lava_world_condition == "query_film":
            raise ValueError("lava_reuse_world_signatures requires world_condition disabled")
        if self.lava_batch_negative_reduction not in {"sum", "mean"}:
            raise ValueError("lava_batch_negative_reduction must be sum or mean")
        self.lava_action_component_calibration = bool(
            lava_action_component_calibration)
        self.lava_action_similarity_min_weight = float(
            lava_action_similarity_min_weight)
        self.lava_action_gripper_indices = tuple(
            int(index) for index in lava_action_gripper_indices)
        self.lava_action_arm_state_weight = float(
            lava_action_arm_state_weight)
        self.lava_action_arm_change_weight = float(
            lava_action_arm_change_weight)
        self.lava_action_gripper_state_weight = float(
            lava_action_gripper_state_weight)
        self.lava_action_gripper_change_weight = float(
            lava_action_gripper_change_weight)
        if any(weight < 0.0 for weight in (
                self.lava_action_arm_state_weight,
                self.lava_action_arm_change_weight,
                self.lava_action_gripper_state_weight,
                self.lava_action_gripper_change_weight)):
            raise ValueError("Action descriptor weights must be non-negative")
        if not 0.0 <= self.lava_action_similarity_min_weight <= 1.0:
            raise ValueError("lava_action_similarity_min_weight must be in [0,1]")
        if (self.lava_action_component_calibration
                and not self.lava_action_similarity_weighting):
            raise ValueError(
                "Action-component calibration requires action-similarity weighting")
        if isinstance(lava_action_target_layer, str):
            normalized_target = lava_action_target_layer.strip().lower()
            if normalized_target == "final":
                self.lava_action_target_layer = "final"
            elif normalized_target.isdigit():
                self.lava_action_target_layer = int(normalized_target)
            else:
                raise ValueError(
                    "lava_action_target_layer must be 'final' or a 1-indexed "
                    f"block number, got {lava_action_target_layer!r}")
        else:
            self.lava_action_target_layer = int(lava_action_target_layer)
        if (self.lava_action_target_layer != "final"
                and not 1 <= self.lava_action_target_layer <= depth):
            raise ValueError(
                f"lava_action_target_layer must be in [1,{depth}] or 'final', "
                f"got {self.lava_action_target_layer}")

        # --- 1. Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- 2. Projections & PosEmb ---
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)

        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))

        # Type embeddings: action, proprio + DINO
        # per_layer: one type emb per DINO layer; concat: a single fused DINO token set -> one type emb
        self.type_emb_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_proprio = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        num_dino_type_emb = 1 if fusion_mode == "concat" else self.num_dino_layers
        self.type_emb_dino_layers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(num_dino_type_emb)
        ])

        nn.init.normal_(self.type_emb_action, std=0.02)
        nn.init.normal_(self.type_emb_proprio, std=0.02)
        for emb in self.type_emb_dino_layers:
            nn.init.normal_(emb, std=0.02)

        # --- 3. Register Tokens ---
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- 3b. Task Condition Token (DINOv3 CLS difference feature) ---
        # Project task_cond_dim -> hidden_dim and append as one extra token
        if self.use_task_cond:
            self.task_cond_proj = nn.Sequential(
                nn.LayerNorm(task_cond_dim),
                nn.Linear(task_cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.type_emb_task_cond = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.type_emb_task_cond, std=0.02)

        # --- 4. DINO adapters ---
        if fusion_mode == "concat":
            # Concat + project -> single adapter
            fusion_out_dim = concat_out_dim if concat_out_dim is not None else dino_feat_dims[0]
            self.concat_fusion = MultiLayerConcatFusion(
                feat_dim=dino_feat_dims[0],
                num_layers=self.num_dino_layers,
                out_dim=fusion_out_dim,
                proj_type=concat_proj_type,
                pre_norm=concat_pre_norm,
            )
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=fusion_out_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
            ])
        else:
            # per_layer: one independent adapter per layer (NOT shared across layers)
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=feat_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
                for feat_dim in dino_feat_dims
            ])

        # --- 5. Core Transformer Blocks ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # --- 6. Output Head ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # --- 7. Future Feature Prediction Decoder (optional) ---
        if self.use_future_feat:
            assert future_feat_num_queries is not None, \
                "use_future_feat=True requires future_feat_num_queries (= future-frame patch token count)"
            ff_out_dim = future_feat_out_dim if future_feat_out_dim is not None else dino_feat_dims[0]
            self.future_feat_decoder = FutureFeatureDecoder(
                num_queries=future_feat_num_queries,
                hidden_dim=hidden_dim,
                out_dim=ff_out_dim,
                num_heads=future_feat_heads,
                num_layers=future_feat_depth,
            )

        # --- 8. LAVA encoders (never called by inference) ---
        if self.use_lava:
            if lava_dino_feat_dim is None:
                raise ValueError("use_lava=True requires lava_dino_feat_dim")
            self.lava_world_encoder = WorldResidualEncoder(
                feat_dim=lava_dino_feat_dim,
                residual_dim=lava_residual_dim,
                query_dim=(lava_query_dim if self.lava_world_encoding == "state_delta" else None),
                hidden_dim=lava_qformer_hidden_dim,
                num_queries=lava_qformer_num_queries,
                num_layers=lava_qformer_num_layers,
                num_heads=lava_qformer_num_heads,
                use_query_film=(self.lava_world_condition == "query_film"),
            )
            # Shared across action positions and temporal scales by design.
            self.lava_action_projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, lava_residual_dim),
            )
            if self.lava_signature_normalization == "ema_calibrated":
                self.lava_signature_calibrator = EMALogSignatureCalibrator(
                    scales=lava_scales,
                    momentum=lava_signature_ema_momentum,
                    level2_weight=lava_signature_level2_weight,
                )
            elif self.lava_signature_normalization == "ema_rms_soft":
                self.lava_signature_calibrator = EMASoftLogSignatureNormalizer(
                    scales=lava_scales, momentum=lava_signature_ema_momentum)
            else:
                # V2/V3 behavior: each non-zero LogSig level is normalized
                # independently before concatenation. Keeping no calibrator
                # buffers also preserves strict loading of legacy checkpoints.
                self.lava_signature_calibrator = None
            # Conditional registration is deliberate: V4 state dicts do not
            # acquire V5-only keys and therefore still strict-load when disabled.
            if self.lava_action_similarity_weighting:
                self.lava_action_distance_calibrator = EMAActionDistanceCalibrator(
                    scales=lava_scales,
                    momentum=lava_action_similarity_beta_momentum,
                    multiplier=lava_action_similarity_beta_multiplier,
                )
                if self.lava_action_component_calibration:
                    self.lava_action_component_calibrator = (
                        EMAActionComponentCalibrator(
                            scales=lava_scales,
                            momentum=lava_action_similarity_beta_momentum,
                        ))

    def forward(self,
                t,
                noisy_actions,
                qpos_history=None,
                dino_features_list=None,    # List[Tensor(B, N_l, feat_dim_l)], ordered as feat_layers
                task_cond=None,             # (B, task_cond_dim) task condition vector
                ):
        """
        Returns:
            final_pred: (B, action_len, action_dim) predicted velocity field
            cond_tokens: (B, M, hidden_dim) observation tokens evolved by the DiT
                (all tokens after the action tokens); used as conditioning (KV)
                for the future-feature decoder
        """
        t_emb = self.time_mlp(t)

        # 1. Action / Proprio tokens
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action

        x_proprio_real = self.proprio_proj(qpos_history) + \
                         self.proprio_pos_emb[:, :qpos_history.shape[1], :] + \
                         self.type_emb_proprio

        # State injection: zero placeholder first, replaced once at the given layer
        x_proprio = torch.zeros_like(x_proprio_real)

        tokens_list = [x_action, x_proprio]
        proprio_start = x_action.shape[1]
        proprio_end = proprio_start + x_proprio.shape[1]

        # 2. Register Tokens
        if self.num_registers > 0:
            regs = self.register_tokens.expand(noisy_actions.shape[0], -1, -1)
            tokens_list.append(regs)

        # 2b. Task Condition Token (projected and appended as one extra token)
        if self.use_task_cond:
            assert task_cond is not None, "use_task_cond=True but task_cond was not passed to forward"
            tc = self.task_cond_proj(task_cond).unsqueeze(1)      # (B, 1, hidden_dim)
            tc = tc + self.type_emb_task_cond
            tokens_list.append(tc)

        # 3. Multi-layer DINO Feature Tokens
        #    per_layer: independent adapter per layer; concat: concat + project -> single adapter
        if dino_features_list is not None:
            assert len(dino_features_list) == self.num_dino_layers, \
                f"Expected {self.num_dino_layers} DINO feature layers, got {len(dino_features_list)}"
            if self.fusion_mode == "concat":
                fused = self.concat_fusion(dino_features_list)       # (B, N, fusion_out_dim)
                cond = self.dino_adapters[0](fused)                  # (B, Q, hidden_dim)
                tokens_list.append(cond + self.type_emb_dino_layers[0])
            else:
                for layer_idx, feats in enumerate(dino_features_list):
                    cond = self.dino_adapters[layer_idx](feats)
                    tokens_list.append(cond + self.type_emb_dino_layers[layer_idx])

        # 4. Concat all tokens
        x = torch.cat(tokens_list, dim=1)

        # 5. Transformer Blocks
        state_injected = False
        lava_action_hidden = None

        for i, block in enumerate(self.blocks):
            if i == self.state_inject_start and not state_injected:
                x[:, proprio_start:proprio_end, :] = x_proprio_real
                state_injected = True

            x = block(x, t_emb)
            if (self.use_lava
                    and self.lava_action_target_layer == i + 1):
                # A view is sufficient: the following blocks create new tensors,
                # while LAVA gradients still terminate at this exact block.
                lava_action_hidden = x[:, :self.action_len, :]

        # 6. Output Head
        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)

        if self.use_lava and self.lava_action_target_layer == "final":
            lava_action_hidden = x_action_out
        if self.use_lava and lava_action_hidden is None:
            raise RuntimeError(
                f"Failed to capture LAVA action layer {self.lava_action_target_layer}")

        # Evolved observation tokens (all tokens after the action tokens),
        # used as cond tokens for future-feature prediction
        cond_tokens = x[:, self.action_len:, :]

        return {
            "final_pred": final_pred,
            "cond_tokens": cond_tokens,
            # Training-only auxiliary consumers use this tap. The policy output
            # always continues through every block, final_norm, and output_proj.
            "action_hidden": (lava_action_hidden
                              if lava_action_hidden is not None else x_action_out),
            "final_action_hidden": x_action_out,
        }

    def _episode_balanced_objective(
            self, action_signatures, world_signatures, positive_paths,
            same_paths, episode_uids, absolute_starts, lengths, contexts,
            time_channels, build_world_signature, action_levels, world_levels,
            action_calibration, world_calibration, action_residuals, world_residuals,
            interval_starts, execution_horizon, endpoint_error, soft_reference, temperature,
            anchor_task_names=None, same_distance_ratios=None,
            cross_scale_extensions=None, cross_scale_prefix_valid=None):
        """Independent 0..4 same-episode + 0..4 cross-episode candidate logits.

        Optional cross-scale candidates share the anchor's start but not its duration:
        extend = frames [t, t+2L] (positive DINO frames + the L extension frames),
        prefix = the positive's first L/2 learned increments. Each is signed at its own
        scale (time channel 1/L') and appended after the cross-episode columns.
        """
        count = len(positive_paths)
        if same_paths is None or len(same_paths) != count:
            raise ValueError("episode_balanced requires a ragged same-episode list per anchor")
        if episode_uids is None or absolute_starts is None:
            raise ValueError("episode_balanced requires episode_uid and absolute start metadata")
        if any(len(group) > 4 for group in same_paths):
            raise ValueError("Too many same-episode negatives")
        # Sampling occurs exactly once, before any checkpointed computation.
        cross_indices = sample_cross_episode_indices(
            episode_uids, absolute_starts, lengths, self.lava_num_cross_episode_negatives)
        action = torch.stack(action_signatures)
        world = torch.stack(world_signatures)
        positive_scores = lava_signature_score(action, world, self.lava_signature_score)
        rows, same_counts, cross_counts = [], [], []
        base_rows = []
        base_width = self.lava_residual_dim + int(self.lava_time_channel)
        base_width += base_width * (base_width - 1) // 2
        if self.lava_projected_third_order:
            base_positive_scores = lava_signature_score(
                action[:, :base_width], world[:, :base_width], self.lava_signature_score)
        state_mode = self.lava_world_encoding == "state_delta"
        reuse = self.lava_reuse_world_signatures
        if reuse:
            if contexts is not None:
                raise ValueError("Reused world signatures are anchor-independent; FiLM must be off")
            # One batched encoding of every same-episode path; cross candidates below
            # reuse the already-encoded, differentiable positive signatures.
            owners = [i for i, group in enumerate(same_paths) for _ in group]
            same_flat = [path for group in same_paths for path in group]
            if any(path.shape[0] != lengths[i] + int(state_mode) for i, path in zip(owners, same_flat)):
                raise ValueError("episode_balanced candidate length does not match anchor scale")
            same_signatures = [[] for _ in same_paths]
            if same_flat:
                states = self.lava_world_encoder(torch.cat([
                    path.to(device=action.device, dtype=positive_paths[0].dtype) for path in same_flat]))
                for i, path in zip(owners, states.split([path.shape[0] for path in same_flat])):
                    same_signatures[i].append(build_world_signature(
                        path[1:] - path[:-1] if state_mode else path, time_channels[i], lengths[i]))
        extend_signatures, prefix_signatures = [None] * count, [None] * count
        if cross_scale_extensions is not None or cross_scale_prefix_valid is not None:
            if not (reuse and state_mode):
                raise ValueError("cross-scale negatives require state_delta and reused world signatures")

            def time_channel(steps):
                return torch.full((steps, int(self.lava_time_channel)), 1.0 / steps,
                                  device=action.device, dtype=world_residuals[0].dtype)
            if cross_scale_extensions is not None:
                if len(cross_scale_extensions) != count:
                    raise ValueError("cross-scale extensions must align with LAVA anchors")
                owners = [i for i, path in enumerate(cross_scale_extensions) if path is not None]
                if owners:
                    full = [torch.cat((positive_paths[i], cross_scale_extensions[i].to(
                        device=positive_paths[i].device, dtype=positive_paths[i].dtype))) for i in owners]
                    if any(path.shape[0] != 2 * lengths[i] + 1 for i, path in zip(owners, full)):
                        raise ValueError("cross-scale extension must add exactly L frames")
                    states = self.lava_world_encoder(torch.cat(full))
                    for i, path in zip(owners, states.split([path.shape[0] for path in full])):
                        extend_signatures[i] = build_world_signature(
                            path[1:] - path[:-1], time_channel(2 * lengths[i]), 2 * lengths[i])
            if cross_scale_prefix_valid is not None:
                if len(cross_scale_prefix_valid) != count:
                    raise ValueError("cross-scale prefix flags must align with LAVA anchors")
                for i, ok in enumerate(cross_scale_prefix_valid):
                    if ok and lengths[i] >= 2:
                        half = lengths[i] // 2
                        prefix_signatures[i] = build_world_signature(
                            world_residuals[i][:half], time_channel(half), half)
        extend_counts, prefix_counts = [], []
        for i, scale in enumerate(lengths):
            candidates = list(same_paths[i]) + [positive_paths[j] for j in cross_indices[i]]
            same_counts.append(len(same_paths[i]))
            cross_counts.append(len(cross_indices[i]))
            extras = [sig for sig in (extend_signatures[i], prefix_signatures[i]) if sig is not None]
            extend_counts.append(int(extend_signatures[i] is not None))
            prefix_counts.append(int(prefix_signatures[i] is not None))
            if not candidates and not extras:
                rows.append(positive_scores[i:i + 1])
                if self.lava_projected_third_order:
                    base_rows.append(base_positive_scores[i:i + 1])
                continue
            path_length = scale + int(state_mode)
            if any(path.shape[0] != path_length for path in candidates):
                raise ValueError("episode_balanced candidate length does not match anchor scale")
            if reuse:
                signatures = torch.stack(same_signatures[i] + [world[j] for j in cross_indices[i]] + extras)
            else:
                frames = torch.cat([path.to(device=action.device, dtype=positive_paths[i].dtype)
                                    for path in candidates])
                context = contexts[i] if contexts is not None else action.new_empty(0)

                def encode_row(frames, context, path_length=path_length, scale=scale,
                               time=time_channels[i], conditioned=contexts is not None):
                    states = (self.lava_world_encoder(frames, context=context[None].expand(len(frames), -1))
                              if conditioned else self.lava_world_encoder(frames))
                    return torch.stack([build_world_signature(
                        path[1:] - path[:-1] if state_mode else path, time, scale)
                        for path in states.split(path_length)])

                signatures = (checkpoint(encode_row, frames, context, use_reentrant=False,
                                         preserve_rng_state=False)
                              if torch.is_grad_enabled() else encode_row(frames, context))
            negative_scores = lava_signature_score(action[i], signatures, self.lava_signature_score)
            rows.append(torch.cat((positive_scores[i:i + 1], negative_scores)))
            if self.lava_projected_third_order:
                base_negative_scores = lava_signature_score(
                    action[i, :base_width], signatures[:, :base_width], self.lava_signature_score)
                base_rows.append(torch.cat((base_positive_scores[i:i + 1], base_negative_scores)))
        width = max(map(len, rows))
        logits = torch.stack([F.pad(row, (0, width - len(row)), value=-torch.inf) for row in rows])
        same = torch.tensor(same_counts, device=action.device)
        cross = torch.tensor(cross_counts, device=action.device)
        extend = torch.tensor(extend_counts, device=action.device)
        prefix = torch.tensor(prefix_counts, device=action.device)
        valid = (same + cross + extend + prefix) > 0
        losses = torch.full_like(positive_scores, float("nan"))
        accuracy = torch.full_like(positive_scores, float("nan"))
        if valid.any():
            selected_losses = F.cross_entropy(logits[valid] / temperature,
                torch.zeros(int(valid.sum()), device=action.device, dtype=torch.long), reduction="none")
            loss = selected_losses.mean()
            losses = losses.index_copy(0, valid.nonzero().flatten(), selected_losses)
            accuracy[valid] = (logits[valid].argmax(1) == 0).float()
        else:
            # Keep the actual encoded positives connected, including FiLM and
            # the action projector. Policy flow/future losses are untouched.
            loss = (action.sum() + world.sum()) * 0.0

        with torch.no_grad():
            def mean(values, mask=None):
                keep = torch.isfinite(values)
                if mask is not None:
                    keep &= mask
                return values[keep].float().mean().item() if keep.any() else float("nan")

            all_positions = torch.cat([torch.arange(int(start) + 1, int(start) + scale + 1,
                device=action.device) for start, scale in zip(interval_starts.tolist(), lengths)])
            diagnostics = {
                "episode_balanced_mode": 1,
                "lava_sample_count": count,  # encoded positive paths, unchanged legacy meaning
                "lava_sampled_anchor_count": count,
                "lava_encoded_anchor_count": count,
                "lava_scored_anchor_count": int(valid.sum()),
                "lava_no_negative_anchor_count": int((~valid).sum()),
                "lava_no_same_episode_count": int((same == 0).sum()),
                "lava_no_cross_episode_count": int((cross == 0).sum()),
                "same_episode_negative_count": same.float().mean().item(),
                "cross_episode_negative_count": cross.float().mean().item(),
                "same_episode_negative_min": int(same.min()),
                "same_episode_negative_max": int(same.max()),
                "cross_episode_negative_min": int(cross.min()),
                "cross_episode_negative_max": int(cross.max()),
                "negative_candidate_count": (same + cross).float().mean().item(),
                "negative_candidate_availability": valid.float().mean().item(),
                "lava_order_negative_count": 0,
                "order_candidate_count": 0.0,
                "state_endpoint_error": endpoint_error,
                "pos_sim": positive_scores.mean().item(),
                "negative_sim": mean(logits[:, 1:]),
                "retrieval_acc": mean(accuracy), "candidate_acc": mean(accuracy),
                "lava_position_mean": all_positions.float().mean().item(),
                "lava_position_min": int(all_positions.min()),
                "lava_position_max": int(all_positions.max()),
                "lava_executed_path_ratio": ((interval_starts + torch.tensor(lengths, device=action.device))
                                             < execution_horizon).float().mean().item(),
            }
            if self.lava_projected_third_order:
                base_logits = torch.stack([
                    F.pad(row, (0, width - len(row)), value=-torch.inf) for row in base_rows])
                base_accuracy = (base_logits[valid].argmax(1) == 0).float()
                diagnostics.update({
                    "third_base_candidate_acc": base_accuracy.mean().item() if valid.any() else float("nan"),
                    "third_candidate_acc_gain": (accuracy[valid].mean() - base_accuracy.mean()).item()
                    if valid.any() else float("nan"),
                    "third_positive_score_contribution": (positive_scores - base_positive_scores).mean().item(),
                    "third_negative_score_contribution": mean(logits[:, 1:] - base_logits[:, 1:]),
                    "third_action_raw_norm": mean(torch.stack([
                        stats["level3_raw_norm"] for stats in action_calibration])),
                    "third_world_raw_norm": mean(torch.stack([
                        stats["level3_raw_norm"] for stats in world_calibration])),
                    "third_action_calibrated_norm": mean(torch.stack([
                        stats["level3_calibrated_norm"] for stats in action_calibration])),
                    "third_world_calibrated_norm": mean(torch.stack([
                        stats["level3_calibrated_norm"] for stats in world_calibration])),
                    "third_action_energy_fraction": mean(torch.stack([
                        stats["level3_energy_fraction"] for stats in action_calibration])),
                    "third_world_energy_fraction": mean(torch.stack([
                        stats["level3_energy_fraction"] for stats in world_calibration])),
                })
            for modality, raw, calibrated, residuals in (
                    ("action", action_levels, action_calibration, action_residuals),
                    ("world", world_levels, world_calibration, world_residuals)):
                norms = [torch.stack([level[k].norm() for level in raw]) for k in (0, 1)]
                for k in (0, 1):
                    diagnostics[f"{modality}_logsig_l{k + 1}_raw_norm"] = norms[k].mean().item()
                    diagnostics[f"{modality}_logsig_l{k + 1}_calibrated_norm"] = mean(torch.stack([
                        stats[f"level{k + 1}_calibrated_norm"] for stats in calibrated]))
                energy = torch.stack([stats['level2_energy_fraction'] for stats in calibrated])
                eligible = torch.tensor(lengths, device=action.device) >= 2
                diagnostics[f"{modality}_logsig_l2_energy_fraction"] = mean(energy, eligible)
                diagnostics[f"{modality}_logsig_l2_l1_ratio"] = mean(norms[1] / norms[0].clamp_min(1e-12), eligible)
                flat = torch.cat(residuals)
                diagnostics[f"{modality}_residual_norm"] = flat.float().norm(dim=-1).mean().item()
                diagnostics[f"{modality}_residual_std"] = flat.float().std(unbiased=False).item()
            for scale in (1, 2, 4, 8, 16):
                mask = torch.tensor(lengths, device=action.device) == scale
                diagnostics[f"loss_s{scale}"] = mean(losses, mask)
                diagnostics[f"pos_sim_s{scale}"] = mean(positive_scores, mask)
                diagnostics[f"candidate_acc_s{scale}"] = mean(accuracy, mask)
                if self.lava_projected_third_order:
                    action_third_energy = torch.stack([
                        stats["level3_energy_fraction"] for stats in action_calibration])
                    world_third_energy = torch.stack([
                        stats["level3_energy_fraction"] for stats in world_calibration])
                    diagnostics[f"third_action_energy_fraction_s{scale}"] = mean(action_third_energy, mask)
                    diagnostics[f"third_world_energy_fraction_s{scale}"] = mean(world_third_energy, mask)
            # Family-resolved competition: softmax shares at the logit level, not
            # parameter-gradient fractions. Uniform exchangeable candidates give
            # mass fraction = same/(same+cross) and hardest-is-same = same/(same+cross).
            columns = torch.arange(width, device=action.device)[None]
            same_mask = (columns >= 1) & (columns < 1 + same[:, None])
            cross_mask = (columns >= 1 + same[:, None]) & (columns < 1 + (same + cross)[:, None])
            scores = logits.float()
            probability = torch.softmax(scores / temperature, dim=1)
            positive = scores[:, 0]
            missing = torch.full_like(scores, -torch.inf)
            max_same = torch.where(same_mask, scores, missing).max(1).values
            max_cross = torch.where(cross_mask, scores, missing).max(1).values
            p_same = (probability * same_mask).sum(1)
            p_cross = (probability * cross_mask).sum(1)
            has_same, has_cross = same > 0, cross > 0
            both = has_same & has_cross
            lengths_tensor = torch.tensor(lengths, device=action.device)

            def share(numerator, denominator, mask):
                total = denominator[mask].sum()
                return (numerator[mask].sum() / total).item() if mask.any() and total > 0 else float("nan")

            def per_candidate(mask):
                return ((probability * mask).sum() / mask.sum()).item() if mask.any() else float("nan")

            diagnostics.update({
                "positive_probability": mean(probability[:, 0], valid),
                "same_episode_probability": mean(p_same, has_same),
                "cross_episode_probability": mean(p_cross, has_cross),
                "same_episode_probability_per_candidate": per_candidate(same_mask),
                "cross_episode_probability_per_candidate": per_candidate(cross_mask),
                "same_episode_mass_fraction": share(p_same, p_same + p_cross, both),
                "same_episode_mass_uniform": share(same.float(), (same + cross).float(), both),
                "hardest_is_same_episode": mean((max_same > max_cross).float(), both),
                "same_episode_acc": mean((positive > max_same).float(), has_same),
                "cross_episode_acc": mean((positive > max_cross).float(), has_cross),
                "same_episode_margin": mean(positive - max_same, has_same),
                "cross_episode_margin": mean(positive - max_cross, has_cross),
                # Paths passed through the world Q-Former this step (positives included).
                "episode_balanced_world_path_count": count + int(same.sum()) + (
                    0 if self.lava_reuse_world_signatures else int(cross.sum())),
            })
            for scale in (1, 2, 4, 8, 16):
                mask = lengths_tensor == scale
                diagnostics[f"same_episode_mass_fraction_s{scale}"] = share(p_same, p_same + p_cross, both & mask)
                diagnostics[f"hardest_is_same_episode_s{scale}"] = mean((max_same > max_cross).float(), both & mask)
                diagnostics[f"same_episode_acc_s{scale}"] = mean((positive > max_same).float(), has_same & mask)
                diagnostics[f"cross_episode_acc_s{scale}"] = mean((positive > max_cross).float(), has_cross & mask)
            if same_distance_ratios is not None:
                # Per same-episode candidate accuracy by |dt|/L band: near [2,4), mid [4,8), far >=8.
                for name, lo, hi in (("near", 2, 4), ("mid", 4, 8), ("far", 8, math.inf)):
                    hits = [float(positive[i] > scores[i, 1 + k])
                            for i, ratios in enumerate(same_distance_ratios)
                            for k, ratio in enumerate(ratios[:int(same[i])]) if lo <= ratio < hi]
                    diagnostics[f"same_episode_acc_{name}"] = sum(hits) / len(hits) if hits else float("nan")
                    diagnostics[f"same_episode_band_fraction_{name}"] = len(hits) / max(int(same.sum()), 1)
            if cross_scale_extensions is not None or cross_scale_prefix_valid is not None:
                rows_index = torch.arange(count, device=action.device)
                for name, has, column in (("extend", extend > 0, 1 + same + cross),
                                          ("prefix", prefix > 0, 1 + same + cross + extend)):
                    variant = scores[rows_index, column.clamp_max(width - 1)]
                    diagnostics[f"cross_scale_{name}_availability"] = has.float().mean().item()
                    diagnostics[f"cross_scale_{name}_acc"] = mean((positive > variant).float(), has)
                    diagnostics[f"cross_scale_{name}_hardest"] = mean((variant > max_cross).float(), has & has_cross)
                    diagnostics[f"cross_scale_{name}_probability"] = mean(
                        probability[rows_index, column.clamp_max(width - 1)], has)
                    for scale in (1, 2, 4, 8, 16):
                        diagnostics[f"cross_scale_{name}_acc_s{scale}"] = mean(
                            (positive > variant).float(), has & (lengths_tensor == scale))
            if anchor_task_names is not None:
                # Same-task cross-episode candidates need dynamics; cross-task ones can be
                # separated by task/scene identity alone.
                same_task_mask = torch.zeros_like(cross_mask)
                for i, row in enumerate(cross_indices):
                    for k, j in enumerate(row):
                        if anchor_task_names[j] == anchor_task_names[i]:
                            same_task_mask[i, 1 + int(same[i]) + k] = True
                cross_task_mask = cross_mask & ~same_task_mask
                max_same_task = torch.where(same_task_mask, scores, missing).max(1).values
                max_cross_task = torch.where(cross_task_mask, scores, missing).max(1).values
                diagnostics.update({
                    "same_task_cross_fraction": (same_task_mask.sum() / cross_mask.sum().clamp_min(1)).item(),
                    "same_task_cross_acc": mean((positive > max_same_task).float(), same_task_mask.any(1)),
                    "cross_task_cross_acc": mean((positive > max_cross_task).float(), cross_task_mask.any(1)),
                    "same_task_cross_probability_per_candidate": per_candidate(same_task_mask),
                    "cross_task_cross_probability_per_candidate": per_candidate(cross_task_mask),
                })
            for label, mask in (("executed", interval_starts + torch.tensor(lengths, device=action.device) < execution_horizon),
                                ("tail", interval_starts + torch.tensor(lengths, device=action.device) >= execution_horizon)):
                diagnostics[f"loss_lava_{label}"] = mean(losses, mask)
                diagnostics[f"candidate_acc_{label}"] = mean(accuracy, mask)
            if contexts is not None:
                gamma, beta = self.lava_world_encoder.film(
                    self.lava_world_encoder.context_norm(contexts)).chunk(2, dim=-1)
                diagnostics.update(film_gamma_rms=gamma.float().square().mean().sqrt().item(),
                                   film_beta_rms=beta.float().square().mean().sqrt().item(),
                                   film_context_count=count)
            if soft_reference is not None:
                mask = torch.ones_like(soft_reference, dtype=torch.bool)
                first = self.lava_signature_calibrator.scale_to_index.get(1)
                if first is not None:
                    mask[:, 1, first] = False
                rms = soft_reference[mask].sqrt()
                diagnostics.update(signature_ema_rms_min=rms.min().item(), signature_ema_rms_max=rms.max().item())
        return loss, diagnostics

    def compute_lava_loss(self, action_hidden, world_feature_differences,
                          batch_indices, interval_starts, interval_scales,
                          temperature=0.07, order_negative=True,
                          flow_timesteps=None, task_names=None,
                          action_execution_horizon=16,
                          temporal_negative_feature_differences=None,
                          far_negative_feature_differences=None,
                          negative_mode="batch",
                          normalized_actions=None, raw_actions=None,
                          temporal_negative_normalized_actions=None,
                          temporal_negative_raw_actions=None,
                          far_negative_normalized_actions=None,
                          far_negative_raw_actions=None,
                          world_feature_paths=None,
                          temporal_negative_feature_paths=None,
                          far_negative_feature_paths=None,
                          lava_context=None,
                          same_episode_negative_features=None,
                          lava_episode_uids=None, lava_absolute_starts=None,
                          same_episode_distance_ratios=None,
                          cross_scale_extension_features=None, cross_scale_prefix_valid=None):
        """Compute one-way action-to-world InfoNCE for sampled intervals."""
        if not self.use_lava:
            raise RuntimeError("compute_lava_loss called while LAVA is disabled")
        # Keep legacy mixed-family checkpoints reproducible. The new mode uses
        # the same local/far sampler but exposes every same-scale batch path as
        # an individual, unweighted candidate (including same-task paths).
        individual_batch_negatives = negative_mode == "mixed_batch"
        paired_batch = negative_mode == "paired_batch"
        paired_diagnostics = {}
        if paired_batch:
            if order_negative or self.lava_action_similarity_weighting or self.lava_world_encoder.use_query_film:
                raise ValueError("paired_batch requires order, action weighting and FiLM disabled")
            negative_mode = "batch"
        episode_balanced = negative_mode == "episode_balanced"
        if episode_balanced:
            if order_negative or self.lava_action_similarity_weighting:
                raise ValueError("episode_balanced requires order and action weighting disabled")
            if self.lava_batch_negative_reduction != "sum":
                raise ValueError("episode_balanced uses independent logits without batch averaging")
        if self.lava_batch_negative_reduction == "mean" and not individual_batch_negatives:
            raise ValueError("batch negative mean reduction requires mixed_batch")
        if individual_batch_negatives:
            if self.lava_action_similarity_weighting:
                raise ValueError("mixed_batch requires action-similarity weighting disabled")
            negative_mode = "mixed"
        state_mode = self.lava_world_encoding == "state_delta"
        if state_mode:
            if world_feature_differences is not None or temporal_negative_feature_differences is not None or far_negative_feature_differences is not None:
                raise ValueError("state_delta requires full frame feature paths, not DINO differences")
            if world_feature_paths is None:
                raise ValueError("state_delta requires world_feature_paths")
            world_inputs = world_feature_paths
            local_inputs = temporal_negative_feature_paths
            far_inputs = far_negative_feature_paths
        else:
            if any(x is not None for x in (world_feature_paths, temporal_negative_feature_paths, far_negative_feature_paths)):
                raise ValueError("feature_delta cannot receive full frame feature paths")
            world_inputs = world_feature_differences
            local_inputs = temporal_negative_feature_differences
            far_inputs = far_negative_feature_differences
        sample_count = len(world_inputs)
        if sample_count == 0:
            zero = action_hidden.sum() * 0.0
            return zero, {
                "pos_sim": 0.0,
                "negative_sim": 0.0,
                "shuffle_sim": 0.0,
                "order_margin": 0.0,
                "retrieval_acc": 0.0,
                "action_pair_sim": 0.0,
                "world_pair_sim": 0.0,
                "same_task_negative_sim": float("nan"),
                "cross_task_negative_sim": float("nan"),
                "task_shortcut_gap": float("nan"),
                "temporal_negative_sim": float("nan"),
                "temporal_margin": float("nan"),
                "temporal_acc": float("nan"),
                "order_acc": float("nan"),
                "candidate_acc": float("nan"),
                "positive_temporal_world_sim": float("nan"),
                "lava_sample_count": 0,
                "lava_order_negative_count": 0,
            }
        if temperature <= 0:
            raise ValueError(f"LAVA temperature must be positive, got {temperature}")
        if self.lava_action_similarity_weighting:
            if normalized_actions is None or raw_actions is None:
                raise ValueError("V5 action weighting requires normalized and raw actions")
            if negative_mode != "mixed":
                raise ValueError("V5 action weighting expects V4 mixed negatives")
            required_paths = (
                temporal_negative_normalized_actions,
                temporal_negative_raw_actions,
                far_negative_normalized_actions,
                far_negative_raw_actions,
            )
            if any(value is None or len(value) != sample_count for value in required_paths):
                raise ValueError("V5 requires one local and far action path per anchor")
        if negative_mode not in {"batch", "episode_local", "mixed", "episode_balanced"}:
            raise ValueError(
                "negative_mode must be 'batch', 'episode_local', or 'mixed', "
                f"got {negative_mode}")
        if negative_mode in {"episode_local", "mixed"}:
            if local_inputs is None:
                raise ValueError(
                    f"{negative_mode} mode requires local-negative feature paths")
            if len(local_inputs) != sample_count:
                raise ValueError(
                    "Every positive LAVA path requires one temporal-negative path")
            if negative_mode == "mixed":
                if far_inputs is None:
                    raise ValueError(
                        "mixed mode requires far-negative feature paths")
                if len(far_inputs) != sample_count:
                    raise ValueError(
                        "Every positive LAVA path requires one far-negative path")
            elif far_inputs is not None:
                raise ValueError(
                    "episode_local mode must not receive far-negative paths")
        elif (local_inputs is not None
              or far_inputs is not None):
            raise ValueError(
                "batch mode must not receive paired negative feature paths")

        batch_indices = batch_indices.to(action_hidden.device, dtype=torch.long)
        interval_starts = interval_starts.to(action_hidden.device, dtype=torch.long)
        interval_scales = interval_scales.to(action_hidden.device, dtype=torch.long)
        if not (sample_count == batch_indices.numel() == interval_starts.numel() == interval_scales.numel()):
            raise ValueError("LAVA batch metadata and feature-difference counts do not match")

        lengths = [int(scale) for scale in interval_scales.tolist()]
        conditioned = self.lava_world_condition == "query_film"
        selected_contexts = None
        if conditioned:
            if not (individual_batch_negatives or episode_balanced):
                raise ValueError("query_film currently requires mixed_batch negatives")
            if lava_context is None or lava_context.ndim != 2 or lava_context.shape[0] != action_hidden.shape[0]:
                raise ValueError("query_film requires current-observation context for the policy batch")
            selected_contexts = lava_context.to(action_hidden.device)[batch_indices]
        groups = [world_inputs]
        if local_inputs is not None:
            groups.append(local_inputs)
        if far_inputs is not None:
            groups.append(far_inputs)
        input_lengths = [length + int(state_mode) for length in lengths]
        paths = []
        for group in groups:
            for path, expected in zip(group, input_lengths):
                if path.shape[0] != expected:
                    raise ValueError(f"LAVA feature path length {path.shape[0]} != {expected}")
                paths.append(path.to(action_hidden.device, dtype=action_hidden.dtype))
        flat_frames = torch.cat(paths, dim=0)
        if conditioned:
            flat_contexts = torch.cat([
                context[None, :].expand(length, -1)
                for _ in groups
                for context, length in zip(selected_contexts, input_lengths)
            ], dim=0)
            encoded = self.lava_world_encoder(flat_frames, context=flat_contexts)
        else:
            encoded = self.lava_world_encoder(flat_frames)
        encoded_paths = list(encoded.split(input_lengths * len(groups), dim=0))
        residual_paths = ([path[1:] - path[:-1] for path in encoded_paths]
                          if state_mode else encoded_paths)
        world_residual_paths = residual_paths[:sample_count]
        temporal_negative_residual_paths = (
            residual_paths[sample_count:2 * sample_count] if local_inputs is not None else None)
        far_negative_residual_paths = (
            residual_paths[2 * sample_count:] if far_inputs is not None else None)
        flat_world_residuals = torch.cat(world_residual_paths)
        # Diagnostics use actual input transitions; no extra trainable encoding.
        positive_inputs = paths[:sample_count]
        flat_world = torch.cat([
            path[1:].float() - path[:-1].float() if state_mode else path
            for path in positive_inputs])
        state_endpoint_error = float("nan")
        if state_mode:
            with torch.no_grad():
                state_endpoint_error = max(
                    (residual.sum(0) - (state[-1] - state[0])).abs().max().item()
                    for residual, state in zip(world_residual_paths, encoded_paths))

        action_residual_paths = []
        for sample_idx, start, scale in zip(
                batch_indices.tolist(), interval_starts.tolist(), lengths):
            # Visual delta c_r = Z_{r+1} - Z_r aligns with action hidden h_{r+1};
            # h_0 intentionally never participates in LAVA.
            hidden_path = action_hidden[sample_idx, start + 1:start + scale + 1]
            if hidden_path.shape[0] != scale:
                raise ValueError(
                    f"LAVA interval [{start}, {start + scale}] exceeds action chunk "
                    f"length {action_hidden.shape[1]}")
            action_residual_paths.append(self.lava_action_projector(hidden_path))

        flat_action_residuals = torch.cat(action_residual_paths, dim=0)

        action_raw_levels = []
        world_raw_levels = []
        action_full_increments = []
        world_full_increments = []
        action_level1_raw_norms = []
        action_level2_raw_norms = []
        world_level1_raw_norms = []
        world_level2_raw_norms = []
        def raw_signature_levels(increments):
            if state_mode:
                # einsum is autocast-eligible even for float() inputs. Keep V6
                # signed areas in FP32 as well as the state differences.
                with torch.autocast(device_type=increments.device.type, enabled=False):
                    return _raw_logsignature_levels(increments)
            return _raw_logsignature_levels(increments)

        time_channels = []
        for action_path, world_path, scale in zip(
                action_residual_paths, world_residual_paths, lengths):
            time_channel = torch.full(
                (scale, int(self.lava_time_channel)), 1.0 / scale, device=action_path.device, dtype=action_path.dtype)
            time_channels.append(time_channel)
            action_increments = torch.cat((action_path, time_channel), dim=-1)
            world_increments = torch.cat((world_path, time_channel), dim=-1)
            action_full_increments.append(action_increments)
            world_full_increments.append(world_increments)
            action_levels = raw_signature_levels(action_increments)
            world_levels = raw_signature_levels(world_increments)
            action_raw_levels.append(action_levels)
            world_raw_levels.append(world_levels)
            action_level1_raw_norms.append(action_levels[0].detach().norm())
            action_level2_raw_norms.append(action_levels[1].detach().norm())
            world_level1_raw_norms.append(world_levels[0].detach().norm())
            world_level2_raw_norms.append(world_levels[1].detach().norm())

        # EMA mode updates only from positive paths. Paired negatives then use
        # exactly the same detached population calibration in this forward.
        if self.training and self.lava_signature_calibrator is not None:
            self.lava_signature_calibrator.update(
                "action", [value[0] for value in action_raw_levels],
                [value[1] for value in action_raw_levels], lengths)
            self.lava_signature_calibrator.update(
                "world", [value[0] for value in world_raw_levels],
                [value[1] for value in world_raw_levels], lengths)

        soft_reference = (
            self.lava_signature_calibrator.ema_squared_norm.detach().clone()
            if isinstance(self.lava_signature_calibrator, EMASoftLogSignatureNormalizer)
            else None)

        def build_signature(levels, modality, scale, increments=None):
            if self.lava_signature_normalization == "graded_soft":
                signature, stats = graded_soft_logsignature(
                    *levels, scale, depth=self.lava_logsig_depth, rho=self.lava_signature_rho)
                if self.lava_projected_third_order:
                    if increments is None:
                        raise ValueError("Projected third order requires path increments")
                    with torch.autocast(device_type=increments.device.type, enabled=False):
                        delta, area = (level.float() for level in levels)
                        s2 = (self.lava_signature_rho ** 4
                              + delta.square().sum().square()
                              + area.square().sum()).sqrt()
                        raw_third = projected_logsignature_level_three(
                            increments, self.lava_third_projection)
                        calibrated_third = self.lava_third_weight * raw_third / s2.pow(1.5)
                        signature = torch.cat((signature, calibrated_third))
                        stats["level3_raw_norm"] = raw_third.detach().norm()
                        stats["level3_calibrated_norm"] = calibrated_third.detach().norm()
                        stats["level3_energy_fraction"] = (
                            calibrated_third.detach().square().sum()
                            / signature.detach().square().sum().clamp_min(1e-12))
                return signature, stats
            if self.lava_projected_third_order:
                raise ValueError("Projected third order currently requires graded_soft")
            if soft_reference is not None:
                return self.lava_signature_calibrator(
                    *levels, modality, scale, depth=self.lava_logsig_depth,
                    reference=soft_reference)
            if self.lava_signature_calibrator is not None:
                return self.lava_signature_calibrator(
                    *levels, modality, scale, depth=self.lava_logsig_depth)

            level_one = F.normalize(levels[0].float(), dim=0, eps=1e-6)
            if self.lava_logsig_depth == 1:
                return level_one, {
                    "level1_calibrated_norm": level_one.detach().norm(),
                    "level2_calibrated_norm": level_one.new_tensor(0.0),
                    "level2_energy_fraction": level_one.new_tensor(0.0),
                    "level1_ema_rms": level_one.new_tensor(float("nan")),
                    "level2_ema_rms": level_one.new_tensor(float("nan")),
                }
            level_two = F.normalize(levels[1].float(), dim=0, eps=1e-6)
            joined = torch.cat((level_one, level_two), dim=0)
            energy_one = level_one.square().sum()
            energy_two = level_two.square().sum()
            signature = F.normalize(joined, dim=0, eps=1e-6)
            return signature, {
                "level1_calibrated_norm": level_one.detach().norm(),
                "level2_calibrated_norm": level_two.detach().norm(),
                "level2_energy_fraction": (
                    energy_two / (energy_one + energy_two).clamp_min(1e-12)),
                "level1_ema_rms": level_one.new_tensor(float("nan")),
                "level2_ema_rms": level_one.new_tensor(float("nan")),
            }

        action_signatures = []
        world_signatures = []
        action_calibration = []
        world_calibration = []
        for action_levels, world_levels, action_increments, world_increments, scale in zip(
                action_raw_levels, world_raw_levels, action_full_increments,
                world_full_increments, lengths):
            action_signature, action_stats = build_signature(
                action_levels, "action", scale, action_increments)
            world_signature, world_stats = build_signature(
                world_levels, "world", scale, world_increments)
            action_signatures.append(action_signature)
            world_signatures.append(world_signature)
            action_calibration.append(action_stats)
            world_calibration.append(world_stats)

        def calibrated_world_signature(path, time_channel, scale):
            increments = torch.cat((path, time_channel), dim=-1)
            levels = raw_signature_levels(increments)
            return build_signature(levels, "world", scale, increments)[0]

        if episode_balanced:
            return self._episode_balanced_objective(
                action_signatures, world_signatures, paths[:sample_count],
                same_episode_negative_features, lava_episode_uids, lava_absolute_starts,
                lengths, selected_contexts, time_channels, calibrated_world_signature,
                action_raw_levels, world_raw_levels, action_calibration, world_calibration,
                action_residual_paths, world_residual_paths, interval_starts,
                action_execution_horizon, state_endpoint_error, soft_reference, temperature,
                anchor_task_names=(None if task_names is None else
                                   [task_names[index] for index in batch_indices.tolist()]),
                same_distance_ratios=same_episode_distance_ratios,
                cross_scale_extensions=cross_scale_extension_features,
                cross_scale_prefix_valid=cross_scale_prefix_valid)

        temporal_negative_signatures = []
        far_negative_signatures = []
        block_swap_signatures = []
        derangement_signatures = []
        block_action_indices = []
        derangement_action_indices = []
        for sample_idx, (world_path, time_channel, scale) in enumerate(zip(
                world_residual_paths, time_channels, lengths)):
            if temporal_negative_residual_paths is not None:
                temporal_negative_signatures.append(calibrated_world_signature(
                    temporal_negative_residual_paths[sample_idx], time_channel, scale))
            if far_negative_residual_paths is not None:
                far_negative_signatures.append(calibrated_world_signature(
                    far_negative_residual_paths[sample_idx], time_channel, scale))
            # V5.1: two-transition paths carry almost no reliable order signal
            # (V5 order accuracy was ~chance at L=2). Keep both order
            # corruptions as hard negatives only where L >= 4.
            if order_negative and scale >= 4:
                block_permutation = sample_contiguous_block_swap_permutation(
                    scale, device=world_path.device)
                derangement_permutation = sample_full_shuffle_permutation(
                    scale, device=world_path.device)
                block_swap_signatures.append(calibrated_world_signature(
                    world_path[block_permutation], time_channel, scale))
                block_action_indices.append(sample_idx)
                # L=2 has only one non-identity permutation; do not duplicate it.
                if scale >= 3:
                    derangement_signatures.append(calibrated_world_signature(
                        world_path[derangement_permutation], time_channel, scale))
                    derangement_action_indices.append(sample_idx)

        with torch.no_grad():
            order_distances = [
                (signature - world_signatures[index]).float().norm()
                for signatures, indices in (
                    (block_swap_signatures, block_action_indices),
                    (derangement_signatures, derangement_action_indices))
                for signature, index in zip(signatures, indices)]
        action_signatures = torch.stack(action_signatures)
        world_signatures = torch.stack(world_signatures)
        device = action_signatures.device
        positive_values = lava_signature_score(
            action_signatures, world_signatures, self.lava_signature_score)
        nan_value = action_signatures.new_tensor(float("nan"))
        same_task_negative_sim = nan_value.clone()
        cross_task_negative_sim = nan_value.clone()
        task_shortcut_gap = nan_value.clone()
        temporal_negative_sim = nan_value.clone()
        temporal_margin = nan_value.clone()
        temporal_accuracy = nan_value.clone()
        candidate_accuracy = nan_value.clone()
        positive_temporal_world_sim = nan_value.clone()
        order_accuracy = nan_value.clone()
        temporal_margin_per_sample = torch.full(
            (sample_count,), torch.nan, device=device, dtype=torch.float32)
        temporal_values = torch.full(
            (sample_count,), torch.nan, device=device,
            dtype=positive_values.dtype)
        candidate_correct_per_sample = torch.full(
            (sample_count,), torch.nan, device=device, dtype=torch.float32)

        if conditioned:
            # Every candidate is interpreted using the ROW anchor's context.
            # Reusing W_j(context_j) would create a condition-ID shortcut.
            # Checkpoint whole rows so N^2 decoding does not retain N copies
            # of Transformer activations. Frozen DINO features are shared.
            candidate_frames = torch.cat(paths[:sample_count], dim=0)

            def encode_candidate_row(frames, context):
                states = self.lava_world_encoder(
                    frames, context=context[None, :].expand(frames.shape[0], -1))
                state_paths = states.split(input_lengths, dim=0)
                return torch.stack([
                    calibrated_world_signature(state[1:] - state[:-1], time, scale)
                    for state, time, scale in zip(state_paths, time_channels, lengths)
                ])

            rows = []
            for context in selected_contexts:
                if torch.is_grad_enabled():
                    row = checkpoint(encode_candidate_row, candidate_frames, context,
                                     use_reentrant=False, preserve_rng_state=False)
                else:
                    row = encode_candidate_row(candidate_frames, context)
                rows.append(row)
            candidate_signatures = torch.stack(rows)
            positive_world_logits = (torch.einsum(
                "id,ijd->ij", action_signatures, candidate_signatures)
                if self.lava_signature_score == "dot" else lava_signature_score(
                    action_signatures[:, None, :], candidate_signatures, self.lava_signature_score))
        else:
            positive_world_logits = (action_signatures @ world_signatures.transpose(0, 1)
                if self.lava_signature_score == "dot" else lava_signature_score(
                    action_signatures[:, None, :], world_signatures[None, :, :], self.lava_signature_score))
        if sample_count > 1:
            off_diagonal = ~torch.eye(
                sample_count, dtype=torch.bool, device=device)
            action_pair_sim = (
                action_signatures @ action_signatures.transpose(0, 1)
            )[off_diagonal].mean()
            world_pair_sim = (
                world_signatures @ world_signatures.transpose(0, 1)
            )[off_diagonal].mean()
        else:
            off_diagonal = torch.zeros(
                (sample_count, sample_count), dtype=torch.bool, device=device)
            action_pair_sim = action_signatures.new_tensor(0.0)
            world_pair_sim = action_signatures.new_tensor(0.0)

        def paired_signature_values(signatures, indices):
            values = torch.full_like(positive_values, -torch.inf)
            if signatures:
                index_tensor = torch.as_tensor(
                    indices, device=device, dtype=torch.long)
                stacked = torch.stack(signatures)
                values[index_tensor] = lava_signature_score(
                    action_signatures[index_tensor], stacked, self.lava_signature_score)
            return values

        def family_logmeanexp(values):
            """Return one count-balanced similarity logit for a negative family."""
            valid = torch.isfinite(values)
            counts = valid.sum(dim=1)
            scaled = (values / temperature).masked_fill(~valid, -torch.inf)
            family = temperature * (
                torch.logsumexp(scaled, dim=1)
                - counts.clamp_min(1).float().log())
            return family.masked_fill(counts == 0, -torch.inf), counts

        block_values = paired_signature_values(
            block_swap_signatures, block_action_indices)
        derangement_values = paired_signature_values(
            derangement_signatures, derangement_action_indices)
        order_candidates = torch.stack((block_values, derangement_values), dim=1)
        order_family_values, order_candidate_counts = family_logmeanexp(
            order_candidates)

        shuffle_sim = action_signatures.new_tensor(0.0)
        order_margin = action_signatures.new_tensor(0.0)
        order_margin_per_sample = torch.full(
            (sample_count,), torch.nan, device=device, dtype=torch.float32)
        order_valid = torch.isfinite(order_family_values)
        if order_valid.any():
            individual_order_values = order_candidates[torch.isfinite(order_candidates)]
            shuffle_sim = individual_order_values.mean()
            order_margin_per_sample[order_valid] = (
                positive_values[order_valid] - order_family_values[order_valid]).float()
            order_margin = order_margin_per_sample[order_valid].mean()
            order_accuracy = (
                positive_values[order_valid] > order_family_values[order_valid]
            ).float().mean()

        selected_task_names = None
        same_task = torch.zeros_like(off_diagonal)
        cross_task = torch.zeros_like(off_diagonal)
        same_scale = interval_scales[:, None] == interval_scales[None, :]
        if task_names is not None and sample_count > 1:
            if len(task_names) != action_hidden.shape[0]:
                raise ValueError(
                    "task_names must have one entry per action-hidden batch sample")
            selected_task_names = [
                task_names[index] for index in batch_indices.tolist()]
            same_task = torch.tensor([
                [left == right for right in selected_task_names]
                for left in selected_task_names
            ], device=device, dtype=torch.bool) & off_diagonal
            cross_task = (~torch.tensor([
                [left == right for right in selected_task_names]
                for left in selected_task_names
            ], device=device, dtype=torch.bool)) & off_diagonal
            if same_task.any():
                same_task_negative_sim = positive_world_logits[same_task].mean()
            if cross_task.any():
                cross_task_negative_sim = positive_world_logits[cross_task].mean()
            if (torch.isfinite(same_task_negative_sim)
                    and torch.isfinite(cross_task_negative_sim)):
                task_shortcut_gap = same_task_negative_sim - cross_task_negative_sim

        cross_same_scale_mask = cross_task & same_scale
        batch_negative_mask = off_diagonal & same_scale
        batch_negative_values = positive_world_logits.masked_fill(
            ~batch_negative_mask, -torch.inf)
        if individual_batch_negatives and self.lava_batch_negative_reduction == "mean":
            # Change only batch mass: sum_j exp(s_ij/T) -> mean_j exp(s_ij/T).
            # Keep the same candidates (including same-task paths), positives,
            # local/far/order terms, and random-number consumption.
            batch_counts = batch_negative_mask.sum(dim=1).clamp_min(1)
            batch_negative_values = (
                batch_negative_values - temperature * batch_counts.float().log()[:, None])
        cross_task_family_values, cross_task_candidate_counts = family_logmeanexp(
            positive_world_logits.masked_fill(~cross_same_scale_mask, -torch.inf))
        raw_cross_task_family_values = cross_task_family_values
        cross_task_valid = torch.isfinite(cross_task_family_values)
        cross_task_margin_per_sample = torch.full(
            (sample_count,), torch.nan, device=device, dtype=torch.float32)
        cross_task_margin_per_sample[cross_task_valid] = (
            positive_values[cross_task_valid]
            - cross_task_family_values[cross_task_valid]).float()

        far_values = torch.full_like(positive_values, -torch.inf)
        far_margin_per_sample = torch.full(
            (sample_count,), torch.nan, device=device, dtype=torch.float32)
        positive_far_world_sim = nan_value.clone()

        if negative_mode in {"episode_local", "mixed"}:
            temporal_negative_signatures = torch.stack(temporal_negative_signatures)
            temporal_values = lava_signature_score(
                action_signatures, temporal_negative_signatures, self.lava_signature_score)
            temporal_negative_sim = temporal_values.mean()
            temporal_margin_per_sample = (positive_values - temporal_values).float()
            temporal_margin = temporal_margin_per_sample.mean()
            temporal_accuracy = (positive_values > temporal_values).float().mean()
            positive_temporal_world_sim = (
                world_signatures * temporal_negative_signatures).sum(dim=-1).mean()
            if negative_mode == "mixed":
                far_negative_signatures = torch.stack(far_negative_signatures)
                far_values = lava_signature_score(
                    action_signatures, far_negative_signatures, self.lava_signature_score)
                far_margin_per_sample = (positive_values - far_values).float()
                positive_far_world_sim = (
                    world_signatures * far_negative_signatures).sum(dim=-1).mean()

            weighted_temporal_values = temporal_values
            weighted_far_values = far_values
            raw_logits = None
            action_similarity_stats = {}
            action_similarity_audit = []
            if self.lava_action_similarity_weighting:
                positive_norm_paths = []
                positive_raw_paths = []
                for batch_index, start, scale in zip(
                        batch_indices.tolist(), interval_starts.tolist(), lengths):
                    positive_norm_paths.append(
                        normalized_actions[batch_index, start:start + scale + 1])
                    positive_raw_paths.append(
                        raw_actions[batch_index, start:start + scale + 1])

                # Action descriptors and their EMA calibrator are deliberately
                # FP32. Under CUDA autocast the similarity logits are BF16, so
                # inheriting their dtype here makes the FP32 gate assignment
                # fail (and would unnecessarily quantize small distances).
                cross_distances = torch.full(
                    positive_world_logits.shape, torch.nan,
                    device=device, dtype=torch.float32)
                local_distances = torch.full(
                    positive_values.shape, torch.nan,
                    device=device, dtype=torch.float32)
                far_distances = torch.full(
                    positive_values.shape, torch.nan,
                    device=device, dtype=torch.float32)
                cross_components = {
                    key: torch.full(
                        positive_world_logits.shape, torch.nan,
                        device=device, dtype=torch.float32)
                    for key in ("arm_state", "arm_change", "gripper_state",
                                "gripper_change")}
                local_components = {
                    key: torch.full(
                        positive_values.shape, torch.nan,
                        device=device, dtype=torch.float32)
                    for key in ("arm_state", "arm_change", "gripper_state",
                                "gripper_change")}
                far_components = {
                    key: torch.full(
                        positive_values.shape, torch.nan,
                        device=device, dtype=torch.float32)
                    for key in ("arm_state", "arm_change", "gripper_state",
                                "gripper_change")}
                cross_weights = torch.ones(
                    positive_world_logits.shape, device=device, dtype=torch.float32)
                local_weights = torch.ones(
                    positive_values.shape, device=device, dtype=torch.float32)
                far_weights = torch.ones(
                    positive_values.shape, device=device, dtype=torch.float32)
                raw_component_values = {
                    "arm_state": [], "arm_change": [], "gripper_state": [],
                    "gripper_change": [], "combined": []}

                def measure(left_index, right_normalized, right_raw):
                    values = action_path_distance(
                        positive_norm_paths[left_index], positive_raw_paths[left_index],
                        right_normalized, right_raw,
                        gripper_indices=self.lava_action_gripper_indices,
                        arm_state_weight=self.lava_action_arm_state_weight,
                        arm_change_weight=self.lava_action_arm_change_weight,
                        gripper_state_weight=self.lava_action_gripper_state_weight,
                        gripper_change_weight=self.lava_action_gripper_change_weight)
                    for key, value in zip(raw_component_values, values):
                        raw_component_values[key].append(value.detach().float())
                    return tuple(value.to(device=device) for value in values)

                for left in range(sample_count):
                    for right in range(sample_count):
                        if cross_same_scale_mask[left, right]:
                            measured = measure(
                                left, positive_norm_paths[right], positive_raw_paths[right])
                            cross_distances[left, right] = measured[-1]
                            for key, value in zip(cross_components, measured[:-1]):
                                cross_components[key][left, right] = value
                    measured = measure(
                        left, temporal_negative_normalized_actions[left],
                        temporal_negative_raw_actions[left])
                    local_distances[left] = measured[-1]
                    for key, value in zip(local_components, measured[:-1]):
                        local_components[key][left] = value
                    measured = measure(
                        left, far_negative_normalized_actions[left],
                        far_negative_raw_actions[left])
                    far_distances[left] = measured[-1]
                    for key, value in zip(far_components, measured[:-1]):
                        far_components[key][left] = value

                raw_cross_distances = cross_distances.clone()
                raw_local_distances = local_distances.clone()
                raw_far_distances = far_distances.clone()
                calibrated_cross_components = {
                    key: value.clone() for key, value in cross_components.items()}
                calibrated_local_components = {
                    key: value.clone() for key, value in local_components.items()}
                calibrated_far_components = {
                    key: value.clone() for key, value in far_components.items()}

                per_scale_medians = {}
                for scale in sorted(set(lengths)):
                    scale_mask = interval_scales == scale
                    cross_mask = scale_mask[:, None] & cross_same_scale_mask

                    if self.lava_action_component_calibration:
                        def stacked_components(components, mask):
                            return torch.stack([
                                components[key][mask]
                                for key in ("arm_state", "arm_change",
                                            "gripper_state", "gripper_change")
                            ], dim=-1)

                        component_families = (
                            stacked_components(cross_components, cross_mask),
                            stacked_components(local_components, scale_mask),
                            stacked_components(far_components, scale_mask),
                        )
                        if self.training:
                            component_beta = (
                                self.lava_action_component_calibrator.update(
                                    scale, component_families))
                        else:
                            component_beta = (
                                self.lava_action_component_calibrator.beta(scale))

                        def calibrate_family(raw_components, calibrated_components,
                                             mask):
                            for component_index, key in enumerate(
                                    ("arm_state", "arm_change", "gripper_state",
                                     "gripper_change")):
                                calibrated_components[key][mask] = (
                                    raw_components[key][mask]
                                    / component_beta[component_index])

                        calibrate_family(
                            cross_components, calibrated_cross_components, cross_mask)
                        calibrate_family(
                            local_components, calibrated_local_components, scale_mask)
                        calibrate_family(
                            far_components, calibrated_far_components, scale_mask)

                        def combined_distance(components, mask):
                            return (
                                self.lava_action_arm_state_weight
                                * components["arm_state"][mask]
                                + self.lava_action_arm_change_weight
                                * components["arm_change"][mask]
                                + self.lava_action_gripper_state_weight
                                * components["gripper_state"][mask]
                                + self.lava_action_gripper_change_weight
                                * components["gripper_change"][mask])

                        cross_distances[cross_mask] = combined_distance(
                            calibrated_cross_components, cross_mask)
                        local_distances[scale_mask] = combined_distance(
                            calibrated_local_components, scale_mask)
                        far_distances[scale_mask] = combined_distance(
                            calibrated_far_components, scale_mask)
                        for component_index, key in enumerate(
                                ("arm_state", "arm_change", "gripper_state",
                                 "gripper_change")):
                            action_similarity_stats[
                                f"action_component_beta_{key}_s{scale}"] = (
                                    component_beta[component_index].item())

                    cross_values = cross_distances[cross_mask]
                    cross_values = cross_values[torch.isfinite(cross_values)]
                    local_values = local_distances[scale_mask]
                    far_distance_values = far_distances[scale_mask]
                    families = (cross_values, local_values, far_distance_values)
                    medians = [values.median() if values.numel() else None
                               for values in families]
                    per_scale_medians[scale] = medians
                    aggregate = (self.lava_action_distance_calibrator.update(
                        scale, families) if self.training else None)
                    available = [value for value in medians if value is not None]
                    if aggregate is None:
                        aggregate = (torch.stack(available).mean()
                                     if available else nan_value)
                    beta = self.lava_action_distance_calibrator.beta(scale)

                    def gate(distance):
                        return action_similarity_weight(
                            distance, beta,
                            self.lava_action_similarity_min_weight)

                    cross_weights[cross_mask] = gate(cross_distances[cross_mask])
                    local_weights[scale_mask] = gate(local_distances[scale_mask])
                    far_weights[scale_mask] = gate(far_distances[scale_mask])
                    for family_name, median in zip(("cross", "local", "far"), medians):
                        action_similarity_stats[
                            f"action_distance_{family_name}_median_s{scale}"] = (
                                median.item() if median is not None else float("nan"))
                    action_similarity_stats[f"action_distance_aggregate_s{scale}"] = (
                        aggregate.item())
                    action_similarity_stats[f"action_distance_beta_s{scale}"] = beta.item()

                cross_task_family_values, _ = weighted_count_balanced_family_logit(
                    positive_world_logits.masked_fill(
                        ~cross_same_scale_mask, -torch.inf), cross_weights,
                    temperature)
                weighted_temporal_values = temporal_values + temperature * local_weights.log()
                weighted_far_values = far_values + temperature * far_weights.log()

                def finite_weight_stats(name, weights, mask=None):
                    values = weights[mask] if mask is not None else weights
                    if values.numel():
                        action_similarity_stats[f"{name}_neg_weight_mean"] = values.mean().item()
                        action_similarity_stats[f"{name}_neg_weight_below_0_5"] = (
                            (values < 0.5).float().mean().item())

                finite_weight_stats("cross_task", cross_weights, cross_same_scale_mask)
                finite_weight_stats("local", local_weights)
                finite_weight_stats("far", far_weights)
                action_similarity_stats["effective_negative_mass"] = (
                    cross_weights[cross_same_scale_mask].sum()
                    + local_weights.sum() + far_weights.sum()).item() / sample_count
                for key, values in raw_component_values.items():
                    action_similarity_stats[
                        {"arm_state": "arm_state_distance",
                         "arm_change": "arm_change_distance",
                         "gripper_state": "gripper_state_distance",
                         "gripper_change": "gripper_change_distance",
                         "combined": "raw_combined_action_distance"}[key]] = (
                            torch.stack(values).mean().item()
                            if values else float("nan"))
                # Keep the V5 field as a compatibility alias for arm change.
                action_similarity_stats["arm_action_distance"] = (
                    action_similarity_stats["arm_change_distance"])

                def collect_family_values(components, mask):
                    return {
                        key: values[mask]
                        for key, values in components.items()
                    }

                calibrated_families = (
                    collect_family_values(
                        calibrated_cross_components, cross_same_scale_mask),
                    collect_family_values(
                        calibrated_local_components,
                        torch.ones_like(local_distances, dtype=torch.bool)),
                    collect_family_values(
                        calibrated_far_components,
                        torch.ones_like(far_distances, dtype=torch.bool)),
                )
                calibrated_all = {
                    key: torch.cat([family[key] for family in calibrated_families])
                    for key in ("arm_state", "arm_change", "gripper_state",
                                "gripper_change")
                }
                contributions = {
                    "arm_state": (self.lava_action_arm_state_weight
                                  * calibrated_all["arm_state"]),
                    "arm_change": (self.lava_action_arm_change_weight
                                   * calibrated_all["arm_change"]),
                    "gripper_state": (self.lava_action_gripper_state_weight
                                      * calibrated_all["gripper_state"]),
                    "gripper_change": (self.lava_action_gripper_change_weight
                                       * calibrated_all["gripper_change"]),
                }
                combined_values = sum(contributions.values())
                action_similarity_stats["combined_action_distance"] = (
                    combined_values.mean().item())
                for key in ("arm_state", "arm_change", "gripper_state",
                            "gripper_change"):
                    action_similarity_stats[
                        f"calibrated_{key}_action_distance"] = (
                            calibrated_all[key].mean().item())
                    valid_combined = combined_values > 1e-12
                    fraction = (
                        contributions[key][valid_combined]
                        / combined_values[valid_combined])
                    action_similarity_stats[
                        f"action_distance_{key}_contribution_fraction"] = (
                            fraction.mean().item() if fraction.numel()
                            else float("nan"))
                # Keep the V5 calibrated-arm field as an arm-change alias.
                action_similarity_stats["calibrated_arm_action_distance"] = (
                    action_similarity_stats[
                        "calibrated_arm_change_action_distance"])
                for index, (task_name, scale) in enumerate(zip(
                        selected_task_names or ["unknown"] * sample_count, lengths)):
                    cross_mask = cross_same_scale_mask[index]
                    cross_audit = None
                    if cross_mask.any():
                        cross_audit = {
                            "distance": cross_distances[index, cross_mask].mean().item(),
                            "raw_distance": raw_cross_distances[
                                index, cross_mask].mean().item(),
                            "weight": cross_weights[index, cross_mask].mean().item(),
                            **{
                                key: values[index, cross_mask].mean().item()
                                for key, values in cross_components.items()
                            },
                            **{
                                f"calibrated_{key}": values[
                                    index, cross_mask].mean().item()
                                for key, values in calibrated_cross_components.items()
                            },
                        }
                    action_similarity_audit.append({
                        "task": str(task_name), "scale": int(scale),
                        "cross": cross_audit,
                        "local": {"distance": local_distances[index].item(),
                                  "raw_distance": raw_local_distances[index].item(),
                                  "weight": local_weights[index].item(),
                                  **{key: values[index].item()
                                     for key, values in local_components.items()},
                                  **{f"calibrated_{key}": values[index].item()
                                     for key, values in
                                     calibrated_local_components.items()}},
                        "far": {"distance": far_distances[index].item(),
                                "raw_distance": raw_far_distances[index].item(),
                                "weight": far_weights[index].item(),
                                **{key: values[index].item()
                                   for key, values in far_components.items()},
                                **{f"calibrated_{key}": values[index].item()
                                   for key, values in
                                   calibrated_far_components.items()}},
                    })

                # Preserve raw diagnostics while the optimization uses weighted logits.
                raw_cross_margin_per_sample = torch.where(
                    torch.isfinite(raw_cross_task_family_values),
                    positive_values - raw_cross_task_family_values,
                    torch.full_like(positive_values, torch.nan)).float()
                weighted_cross_margin_per_sample = torch.where(
                    torch.isfinite(cross_task_family_values),
                    positive_values - cross_task_family_values,
                    torch.full_like(positive_values, torch.nan)).float()
                weighted_local_margin_per_sample = (
                    positive_values - weighted_temporal_values).float()
                weighted_far_margin_per_sample = (
                    positive_values - weighted_far_values).float()
                cross_task_margin_per_sample = weighted_cross_margin_per_sample

                def descriptor_finite_mean(values):
                    valid = torch.isfinite(values)
                    return (values[valid].mean() if valid.any() else nan_value)

                action_similarity_stats.update({
                    "raw_cross_task_margin": descriptor_finite_mean(
                        raw_cross_margin_per_sample).item(),
                    "weighted_cross_task_margin": descriptor_finite_mean(
                        weighted_cross_margin_per_sample).item(),
                    "raw_local_margin": descriptor_finite_mean(
                        temporal_margin_per_sample).item(),
                    "weighted_local_margin": descriptor_finite_mean(
                        weighted_local_margin_per_sample).item(),
                    "raw_far_margin": descriptor_finite_mean(
                        far_margin_per_sample).item(),
                    "weighted_far_margin": descriptor_finite_mean(
                        weighted_far_margin_per_sample).item(),
                    "raw_cross_task_acc": descriptor_finite_mean(
                        (raw_cross_margin_per_sample > 0).float().masked_fill(
                            ~torch.isfinite(raw_cross_margin_per_sample), torch.nan)).item(),
                    "weighted_cross_task_acc": descriptor_finite_mean(
                        (weighted_cross_margin_per_sample > 0).float().masked_fill(
                            ~torch.isfinite(weighted_cross_margin_per_sample), torch.nan)).item(),
                    "raw_local_acc": (temporal_margin_per_sample > 0).float().mean().item(),
                    "weighted_local_acc": (
                        weighted_local_margin_per_sample > 0).float().mean().item(),
                    "raw_far_acc": (far_margin_per_sample > 0).float().mean().item(),
                    "weighted_far_acc": (
                        weighted_far_margin_per_sample > 0).float().mean().item(),
                })

            if negative_mode == "mixed":
                if individual_batch_negatives:
                    # Positive is column 0. Exclude self and other scales;
                    # each remaining batch path contributes its own exp(logit/T).
                    # Local/far and the existing pooled order candidate remain.
                    logits = torch.cat((
                        positive_values[:, None], batch_negative_values,
                        temporal_values[:, None], far_values[:, None],
                        order_family_values[:, None]), dim=1)
                else:
                    # Legacy four-family, optionally action-weighted objective.
                    logits = torch.stack((
                        positive_values,
                        cross_task_family_values,
                        weighted_temporal_values,
                        weighted_far_values,
                        order_family_values,
                    ), dim=1)
                    raw_logits = torch.stack((
                        positive_values, raw_cross_task_family_values,
                        temporal_values, far_values, order_family_values), dim=1)
            else:
                logits = torch.stack(
                    (positive_values, temporal_values, order_family_values), dim=1)
            labels = torch.zeros(sample_count, dtype=torch.long, device=device)
            loss_lava_per_sample = F.cross_entropy(
                logits / temperature, labels, reduction="none")
            candidate_accuracy = (
                logits.argmax(dim=1) == labels).float().mean()
            if raw_logits is not None:
                action_similarity_stats["raw_candidate_acc"] = (
                    raw_logits.argmax(dim=1) == labels).float().mean().item()
                action_similarity_stats["weighted_candidate_acc"] = (
                    logits.argmax(dim=1) == labels).float().mean().item()
            candidate_correct_per_sample = (
                logits.argmax(dim=1) == labels).float()
            retrieval_acc = candidate_accuracy
            finite_negatives = logits[:, 1:][torch.isfinite(logits[:, 1:])]
            negative_sim = (
                finite_negatives.mean() if finite_negatives.numel()
                else action_signatures.new_tensor(0.0))
        else:
            logits = positive_world_logits
            labels = torch.arange(sample_count, device=device)
            if paired_batch:
                from dataloader.lava_paired import paired_candidate_indices
                if lava_episode_uids is None or lava_absolute_starts is None:
                    raise ValueError('paired_batch requires episode and interval metadata')
                candidates = paired_candidate_indices(
                    lava_episode_uids, lava_absolute_starts, lengths, device)
                logits = positive_world_logits.gather(1, candidates.clamp_min(0))
                logits = logits.masked_fill(candidates < 0, -torch.inf)
                labels = torch.zeros(sample_count, dtype=torch.long, device=device)
                with torch.no_grad():
                    probs = (logits.float() / temperature).softmax(dim=1)
                    paired_diagnostics = {
                        'paired_hard_count': 1.0,
                        'paired_cross_count': (candidates[:, 2:] >= 0).float().sum(1).mean().item(),
                        'paired_hard_acc': (logits[:, 0] > logits[:, 1]).float().mean().item(),
                        'paired_hard_margin': (logits[:, 0] - logits[:, 1]).float().mean().item(),
                        'paired_hard_probability': probs[:, 1].mean().item(),
                        'paired_cross_probability': probs[:, 2:].sum(1).mean().item(),
                        'paired_positive_probability': probs[:, 0].mean().item(),
                        'paired_candidate_acc': (logits.argmax(1) == 0).float().mean().item(),
                        'paired_world_path_count': sample_count,
                        'paired_distance_over_l': sum(
                            abs(int(lava_absolute_starts[i]) - int(lava_absolute_starts[int(candidates[i, 1])])) / lengths[i]
                            for i in range(sample_count)) / sample_count,
                    }
                candidate_accuracy = (logits.argmax(1) == labels).float().mean()
                candidate_correct_per_sample = (logits.argmax(1) == labels).float()
            negative_sim = (
                positive_world_logits[off_diagonal].mean()
                if off_diagonal.any() else action_signatures.new_tensor(0.0))
            if paired_batch:
                negative_sim = logits[:, 1:][torch.isfinite(logits[:, 1:])].mean()
            if block_swap_signatures:
                logits = torch.cat((
                    logits,
                    (action_signatures @ torch.stack(block_swap_signatures).transpose(0, 1)
                     if self.lava_signature_score == "dot" else lava_signature_score(
                         action_signatures[:, None, :], torch.stack(block_swap_signatures)[None, :, :],
                         self.lava_signature_score)),
                ), dim=1)
            if derangement_signatures:
                logits = torch.cat((
                    logits,
                    (action_signatures @ torch.stack(derangement_signatures).transpose(0, 1)
                     if self.lava_signature_score == "dot" else lava_signature_score(
                         action_signatures[:, None, :], torch.stack(derangement_signatures)[None, :, :],
                         self.lava_signature_score)),
                ), dim=1)
            loss_lava_per_sample = F.cross_entropy(
                logits / temperature, labels, reduction="none")
            retrieval_acc = (logits.argmax(dim=1) == labels).float().mean()


        loss_lava = loss_lava_per_sample.mean()
        pos_sim = positive_values.mean()

        def finite_mean(values):
            valid = torch.isfinite(values)
            return (values[valid].mean() if valid.any() else nan_value.clone())

        block_margin_per_sample = torch.where(
            torch.isfinite(block_values), positive_values - block_values,
            torch.full_like(positive_values, torch.nan)).float()
        derangement_margin_per_sample = torch.where(
            torch.isfinite(derangement_values),
            positive_values - derangement_values,
            torch.full_like(positive_values, torch.nan)).float()
        cross_task_family_sim = finite_mean(cross_task_family_values)
        cross_task_margin = finite_mean(cross_task_margin_per_sample)
        cross_task_acc = finite_mean(
            (cross_task_margin_per_sample > 0).float().masked_fill(
                ~torch.isfinite(cross_task_margin_per_sample), torch.nan))
        far_negative_sim = finite_mean(far_values)
        far_margin = finite_mean(far_margin_per_sample)
        far_acc = finite_mean(
            (far_margin_per_sample > 0).float().masked_fill(
                ~torch.isfinite(far_margin_per_sample), torch.nan))
        block_swap_sim = finite_mean(block_values)
        block_swap_margin = finite_mean(block_margin_per_sample)
        block_swap_acc = finite_mean(
            (block_margin_per_sample > 0).float().masked_fill(
                ~torch.isfinite(block_margin_per_sample), torch.nan))
        derangement_sim = finite_mean(derangement_values)
        derangement_margin = finite_mean(derangement_margin_per_sample)
        derangement_acc = finite_mean(
            (derangement_margin_per_sample > 0).float().masked_fill(
                ~torch.isfinite(derangement_margin_per_sample), torch.nan))
        order_family_sim = finite_mean(order_family_values)

        hardest_family_fractions = {
            "cross_task": float("nan"),
            "same_task": float("nan"),
            "batch": float("nan"),
            "local": float("nan"),
            "far": float("nan"),
            "order": float("nan"),
        }
        average_candidate_count = float("nan")
        if negative_mode == "mixed":
            negative_families = torch.stack((
                cross_task_family_values,
                weighted_temporal_values if self.lava_action_similarity_weighting else temporal_values,
                weighted_far_values if self.lava_action_similarity_weighting else far_values,
                order_family_values), dim=1)
            valid_family = torch.isfinite(negative_families)
            average_candidate_count = valid_family.sum(dim=1).float().mean().item()
            has_negative = valid_family.any(dim=1)
            if has_negative.any():
                hardest = negative_families.masked_fill(
                    ~valid_family, -torch.inf).argmax(dim=1)
                names = ("cross_task", "local", "far", "order")
                for index, name in enumerate(names):
                    hardest_family_fractions[name] = (
                        (hardest[has_negative] == index).float().mean().item())
            if individual_batch_negatives:
                # Report actual hardest individual candidates, rather than a
                # family mean that no longer participates in this objective.
                batch_hardest = batch_negative_values.max(dim=1).values
                competitors = torch.stack((
                    batch_hardest, temporal_values, far_values,
                    order_family_values), dim=1)
                average_candidate_count = (
                    torch.isfinite(competitors).sum(dim=1).float().mean().item())
                hardest = competitors.argmax(dim=1)
                for index, name in enumerate(("batch", "local", "far", "order")):
                    hardest_family_fractions[name] = (
                        (hardest == index).float().mean().item())
                batch_winner = batch_negative_values.argmax(dim=1)
                rows = torch.arange(sample_count, device=device)
                for name, mask in (("same_task", same_task), ("cross_task", cross_task)):
                    hardest_family_fractions[name] = (
                        ((hardest == 0) & mask[rows, batch_winner]).float().mean().item())
            if self.lava_action_similarity_weighting:
                raw_negative_families = torch.stack((
                    raw_cross_task_family_values, temporal_values,
                    far_values, order_family_values), dim=1)
                raw_valid = torch.isfinite(raw_negative_families)
                raw_has_negative = raw_valid.any(dim=1)
                raw_hardest = raw_negative_families.masked_fill(
                    ~raw_valid, -torch.inf).argmax(dim=1)
                for index, name in enumerate(("cross_task", "local", "far", "order")):
                    action_similarity_stats[f"raw_hardest_{name}_fraction"] = (
                        (raw_hardest[raw_has_negative] == index).float().mean().item())
                    action_similarity_stats[f"weighted_hardest_{name}_fraction"] = (
                        hardest_family_fractions[name])

        with torch.no_grad():
            raw_world = flat_world.float()
            world_residual = flat_world_residuals.float()
            action_residual = flat_action_residuals.float()
            if state_mode:
                normalized_paths = [self.lava_world_encoder.input_norm(path).float()
                                    for path in positive_inputs]
                input_normalized_world = torch.cat([path[1:] - path[:-1] for path in normalized_paths])
            else:
                input_normalized_world = self.lava_world_encoder.input_norm(flat_world).float()

            raw_transition_norm = raw_world.flatten(1).norm(dim=-1)
            input_norm_transition_norm = input_normalized_world.flatten(1).norm(dim=-1)
            world_residual_transition_norm = world_residual.norm(dim=-1)

            def coefficient_of_variation(values, eps=1e-12):
                mean = values.mean()
                if mean.abs() <= eps:
                    return values.new_tensor(float("nan"))
                return values.std(unbiased=False) / mean.abs()

            def pearson_correlation(left, right, eps=1e-12):
                left_centered = left - left.mean()
                right_centered = right - right.mean()
                denominator = left_centered.norm() * right_centered.norm()
                if denominator <= eps:
                    return left.new_tensor(float("nan"))
                return (left_centered * right_centered).sum() / denominator

            action_l1 = torch.stack(action_level1_raw_norms).float()
            action_l2 = torch.stack(action_level2_raw_norms).float()
            world_l1 = torch.stack(world_level1_raw_norms).float()
            world_l2 = torch.stack(world_level2_raw_norms).float()
            action_cal_l1 = torch.stack([
                value["level1_calibrated_norm"] for value in action_calibration
            ]).float()
            action_cal_l2 = torch.stack([
                value["level2_calibrated_norm"] for value in action_calibration
            ]).float()
            action_l2_energy = torch.stack([
                value["level2_energy_fraction"] for value in action_calibration
            ]).float()
            world_cal_l1 = torch.stack([
                value["level1_calibrated_norm"] for value in world_calibration
            ]).float()
            world_cal_l2 = torch.stack([
                value["level2_calibrated_norm"] for value in world_calibration
            ]).float()
            world_l2_energy = torch.stack([
                value["level2_energy_fraction"] for value in world_calibration
            ]).float()
            order_eligible = interval_scales >= 4

            def eligible_mean(values):
                return (values[order_eligible].mean().item()
                        if order_eligible.any() else float("nan"))

            def eligible_ratio(level_two, level_one):
                ratios = level_two / level_one.clamp_min(1e-12)
                return (ratios[order_eligible].mean().item()
                        if order_eligible.any() else float("nan"))

            diagnostics = {
                "raw_change_norm": raw_world.norm(dim=-1).mean().item(),
                "raw_change_std": raw_world.std(unbiased=False).item(),
                "raw_change_norm_cv": coefficient_of_variation(raw_transition_norm).item(),
                "input_norm_change_norm": input_norm_transition_norm.mean().item(),
                "input_norm_change_norm_cv": coefficient_of_variation(
                    input_norm_transition_norm).item(),
                "raw_input_norm_norm_corr": pearson_correlation(
                    raw_transition_norm, input_norm_transition_norm).item(),
                "raw_world_residual_norm_corr": pearson_correlation(
                    raw_transition_norm, world_residual_transition_norm).item(),
                "world_residual_norm": world_residual.norm(dim=-1).mean().item(),
                "world_residual_std": world_residual.std(dim=0, unbiased=False).mean().item(),
                "action_residual_norm": action_residual.norm(dim=-1).mean().item(),
                "action_residual_std": action_residual.std(dim=0, unbiased=False).mean().item(),
                "action_logsig_l1_raw_norm": eligible_mean(action_l1),
                "action_logsig_l2_raw_norm": eligible_mean(action_l2),
                "action_logsig_l2_l1_ratio": eligible_ratio(action_l2, action_l1),
                "world_logsig_l1_raw_norm": eligible_mean(world_l1),
                "world_logsig_l2_raw_norm": eligible_mean(world_l2),
                "world_logsig_l2_l1_ratio": eligible_ratio(world_l2, world_l1),
                "action_logsig_l1_calibrated_norm": eligible_mean(action_cal_l1),
                "action_logsig_l2_calibrated_norm": eligible_mean(action_cal_l2),
                "action_logsig_l2_energy_fraction": eligible_mean(action_l2_energy),
                "world_logsig_l1_calibrated_norm": eligible_mean(world_cal_l1),
                "world_logsig_l2_calibrated_norm": eligible_mean(world_cal_l2),
                "world_logsig_l2_energy_fraction": eligible_mean(world_l2_energy),
            }

            if soft_reference is not None:
                valid_reference = torch.ones_like(soft_reference, dtype=torch.bool)
                first_scale = self.lava_signature_calibrator.scale_to_index.get(1)
                if first_scale is not None:
                    valid_reference[:, 1, first_scale] = False
                reference_rms = soft_reference[valid_reference].sqrt()
                diagnostics["signature_ema_rms_min"] = reference_rms.min().item()
                diagnostics["signature_ema_rms_max"] = reference_rms.max().item()

            # LAVA aligns c_r with h_{r+1}. Record where supervision actually
            # lands in the action chunk without adding another forward pass.
            supervised_positions = torch.cat([
                torch.arange(start + 1, start + scale + 1,
                             device=logits.device, dtype=torch.long)
                for start, scale in zip(interval_starts.tolist(), lengths)
            ])
            position_count = max(1, supervised_positions.numel())
            for lower, upper in ((1, 8), (9, 16), (17, 24), (25, 31)):
                in_bucket = ((supervised_positions >= lower)
                             & (supervised_positions <= upper))
                diagnostics[f"lava_coverage_pos_{lower}_{upper}"] = (
                    in_bucket.sum().item() / position_count)
            diagnostics.update({
                "lava_position_mean": supervised_positions.float().mean().item(),
                "lava_position_min": supervised_positions.min().item(),
                "lava_position_max": supervised_positions.max().item(),
                "lava_executed_horizon_ratio": (
                    (supervised_positions < int(action_execution_horizon))
                    .float().mean().item()),
            })

            path_end_positions = interval_starts + interval_scales
            executed_path_mask = path_end_positions < int(action_execution_horizon)
            tail_path_mask = ~executed_path_mask

            def masked_metric(values, mask):
                valid = mask & torch.isfinite(values)
                return values[valid].mean().item() if valid.any() else float("nan")

            diagnostics.update({
                "lava_executed_path_ratio": executed_path_mask.float().mean().item(),
                "loss_lava_executed": masked_metric(
                    loss_lava_per_sample.float(), executed_path_mask),
                "loss_lava_tail": masked_metric(
                    loss_lava_per_sample.float(), tail_path_mask),
                "pos_sim_executed": masked_metric(
                    positive_values.float(), executed_path_mask),
                "pos_sim_tail": masked_metric(
                    positive_values.float(), tail_path_mask),
                "candidate_acc_executed": masked_metric(
                    candidate_correct_per_sample, executed_path_mask),
                "candidate_acc_tail": masked_metric(
                    candidate_correct_per_sample, tail_path_mask),
                "local_margin_executed": masked_metric(
                    temporal_margin_per_sample, executed_path_mask),
                "local_margin_tail": masked_metric(
                    temporal_margin_per_sample, tail_path_mask),
                "order_margin_executed": masked_metric(
                    order_margin_per_sample, executed_path_mask),
                "order_margin_tail": masked_metric(
                    order_margin_per_sample, tail_path_mask),
            })

            for scale in (1, 2, 4, 8, 16):
                scale_mask = interval_scales == scale
                if scale_mask.any():
                    diagnostics[f"loss_s{scale}"] = loss_lava_per_sample[scale_mask].mean().item()
                    diagnostics[f"pos_sim_s{scale}"] = positive_values[scale_mask].mean().item()
                    margin_mask = scale_mask & torch.isfinite(order_margin_per_sample)
                    diagnostics[f"order_margin_s{scale}"] = (
                        order_margin_per_sample[margin_mask].mean().item()
                        if margin_mask.any() else 0.0)
                    temporal_mask = scale_mask & torch.isfinite(
                        temporal_margin_per_sample)
                    diagnostics[f"temporal_negative_sim_s{scale}"] = (
                        temporal_values[temporal_mask].mean().item()
                        if temporal_mask.any() else float("nan"))
                    diagnostics[f"temporal_margin_s{scale}"] = (
                        temporal_margin_per_sample[temporal_mask].mean().item()
                        if temporal_mask.any() else float("nan"))
                    diagnostics[f"temporal_acc_s{scale}"] = (
                        (temporal_margin_per_sample[temporal_mask] > 0)
                        .float().mean().item()
                        if temporal_mask.any() else float("nan"))
                    diagnostics[f"order_acc_s{scale}"] = (
                        (order_margin_per_sample[margin_mask] > 0)
                        .float().mean().item()
                        if margin_mask.any() else float("nan"))
                    candidate_mask = scale_mask & torch.isfinite(
                        candidate_correct_per_sample)
                    diagnostics[f"candidate_acc_s{scale}"] = (
                        candidate_correct_per_sample[candidate_mask].mean().item()
                        if candidate_mask.any() else float("nan"))
                    diagnostics[f"far_margin_s{scale}"] = masked_metric(
                        far_margin_per_sample, scale_mask)
                    diagnostics[f"cross_task_margin_s{scale}"] = masked_metric(
                        cross_task_margin_per_sample, scale_mask)
                    diagnostics[f"block_swap_margin_s{scale}"] = masked_metric(
                        block_margin_per_sample, scale_mask)
                    diagnostics[f"derangement_margin_s{scale}"] = masked_metric(
                        derangement_margin_per_sample, scale_mask)
                else:
                    diagnostics[f"loss_s{scale}"] = 0.0
                    diagnostics[f"pos_sim_s{scale}"] = 0.0
                    diagnostics[f"order_margin_s{scale}"] = 0.0
                    diagnostics[f"temporal_negative_sim_s{scale}"] = float("nan")
                    diagnostics[f"temporal_margin_s{scale}"] = float("nan")
                    diagnostics[f"temporal_acc_s{scale}"] = float("nan")
                    diagnostics[f"order_acc_s{scale}"] = float("nan")
                    diagnostics[f"candidate_acc_s{scale}"] = float("nan")
                    diagnostics[f"far_margin_s{scale}"] = float("nan")
                    diagnostics[f"cross_task_margin_s{scale}"] = float("nan")
                    diagnostics[f"block_swap_margin_s{scale}"] = float("nan")
                    diagnostics[f"derangement_margin_s{scale}"] = float("nan")

            for scale in (2, 4, 8, 16):
                scale_mask = interval_scales == scale
                if scale_mask.any():
                    diagnostics[f"action_logsig_l2_l1_ratio_s{scale}"] = (
                        (action_l2[scale_mask] / action_l1[scale_mask].clamp_min(1e-12))
                        .mean().item())
                    diagnostics[f"world_logsig_l2_l1_ratio_s{scale}"] = (
                        (world_l2[scale_mask] / world_l1[scale_mask].clamp_min(1e-12))
                        .mean().item())
                    diagnostics[f"action_logsig_l2_energy_fraction_s{scale}"] = (
                        action_l2_energy[scale_mask].mean().item())
                    diagnostics[f"world_logsig_l2_energy_fraction_s{scale}"] = (
                        world_l2_energy[scale_mask].mean().item())
                    if self.lava_signature_calibrator is not None:
                        scale_index = self.lava_signature_calibrator.scale_to_index[scale]
                        ema = self.lava_signature_calibrator.ema_squared_norm
                        diagnostics[f"action_logsig_l1_ema_rms_s{scale}"] = (
                            ema[0, 0, scale_index].sqrt().item())
                        diagnostics[f"action_logsig_l2_ema_rms_s{scale}"] = (
                            ema[0, 1, scale_index].sqrt().item())
                        diagnostics[f"world_logsig_l1_ema_rms_s{scale}"] = (
                            ema[1, 0, scale_index].sqrt().item())
                        diagnostics[f"world_logsig_l2_ema_rms_s{scale}"] = (
                            ema[1, 1, scale_index].sqrt().item())
                    else:
                        diagnostics[f"action_logsig_l1_ema_rms_s{scale}"] = float("nan")
                        diagnostics[f"action_logsig_l2_ema_rms_s{scale}"] = float("nan")
                        diagnostics[f"world_logsig_l1_ema_rms_s{scale}"] = float("nan")
                        diagnostics[f"world_logsig_l2_ema_rms_s{scale}"] = float("nan")
                else:
                    diagnostics[f"action_logsig_l2_l1_ratio_s{scale}"] = float("nan")
                    diagnostics[f"world_logsig_l2_l1_ratio_s{scale}"] = float("nan")
                    diagnostics[f"action_logsig_l2_energy_fraction_s{scale}"] = float("nan")
                    diagnostics[f"world_logsig_l2_energy_fraction_s{scale}"] = float("nan")
                    diagnostics[f"action_logsig_l1_ema_rms_s{scale}"] = float("nan")
                    diagnostics[f"action_logsig_l2_ema_rms_s{scale}"] = float("nan")
                    diagnostics[f"world_logsig_l1_ema_rms_s{scale}"] = float("nan")
                    diagnostics[f"world_logsig_l2_ema_rms_s{scale}"] = float("nan")

            timestep_labels = ("t0_025", "t025_050", "t050_075", "t075_100")
            if flow_timesteps is not None:
                supervised_t = flow_timesteps.to(logits.device)[batch_indices]
                timestep_bins = torch.bucketize(
                    supervised_t, torch.tensor([0.25, 0.5, 0.75], device=logits.device))
                diagnostics["lava_t_mean"] = supervised_t.float().mean().item()
                for bin_idx, label in enumerate(timestep_labels):
                    bin_mask = timestep_bins == bin_idx
                    diagnostics[f"loss_{label}"] = (
                        loss_lava_per_sample[bin_mask].mean().item() if bin_mask.any() else 0.0)
                    diagnostics[f"pos_sim_{label}"] = (
                        positive_values[bin_mask].mean().item() if bin_mask.any() else 0.0)
                    diagnostics[f"count_{label}"] = int(bin_mask.sum().item())
            else:
                diagnostics["lava_t_mean"] = 0.0
                for label in timestep_labels:
                    diagnostics[f"loss_{label}"] = 0.0
                    diagnostics[f"pos_sim_{label}"] = 0.0
                    diagnostics[f"count_{label}"] = 0

        diagnostics.update({
            "pos_sim": pos_sim.detach().item(),
            "negative_sim": negative_sim.detach().item(),
            "shuffle_sim": shuffle_sim.detach().item(),
            "order_margin": order_margin.detach().item(),
            "retrieval_acc": retrieval_acc.detach().item(),
            "action_pair_sim": action_pair_sim.detach().item(),
            "world_pair_sim": world_pair_sim.detach().item(),
            "same_task_negative_sim": same_task_negative_sim.detach().item(),
            "cross_task_negative_sim": cross_task_negative_sim.detach().item(),
            "task_shortcut_gap": task_shortcut_gap.detach().item(),
            "temporal_negative_sim": temporal_negative_sim.detach().item(),
            "temporal_margin": temporal_margin.detach().item(),
            "temporal_acc": temporal_accuracy.detach().item(),
            "order_acc": order_accuracy.detach().item(),
            "candidate_acc": candidate_accuracy.detach().item(),
            "positive_temporal_world_sim": (
                positive_temporal_world_sim.detach().item()),
            "positive_far_world_sim": positive_far_world_sim.detach().item(),
            "cross_task_family_sim": cross_task_family_sim.detach().item(),
            "cross_task_margin": cross_task_margin.detach().item(),
            "cross_task_acc": cross_task_acc.detach().item(),
            "far_negative_sim": far_negative_sim.detach().item(),
            "far_margin": far_margin.detach().item(),
            "far_acc": far_acc.detach().item(),
            "block_swap_sim": block_swap_sim.detach().item(),
            "block_swap_margin": block_swap_margin.detach().item(),
            "block_swap_acc": block_swap_acc.detach().item(),
            "derangement_sim": derangement_sim.detach().item(),
            "derangement_margin": derangement_margin.detach().item(),
            "derangement_acc": derangement_acc.detach().item(),
            "order_family_sim": order_family_sim.detach().item(),
            "average_negative_family_count": average_candidate_count,
            "hardest_cross_task_fraction": hardest_family_fractions["cross_task"],
            "hardest_local_fraction": hardest_family_fractions["local"],
            "hardest_far_fraction": hardest_family_fractions["far"],
            "hardest_order_fraction": hardest_family_fractions["order"],
            "cross_task_candidate_count": (
                cross_task_candidate_counts.float().mean().item()),
            "order_candidate_count": order_candidate_counts.float().mean().item(),
            "lava_sample_count": sample_count,
            # Backward-compatible: number of paths receiving order supervision,
            # not the number of corruption candidates inside the order family.
            "lava_order_negative_count": len(block_action_indices),
        })
        diagnostics.update(paired_diagnostics)
        if conditioned:
            with torch.no_grad():
                gamma, beta = self.lava_world_encoder.film(
                    self.lava_world_encoder.context_norm(selected_contexts)).chunk(2, dim=-1)
                diagnostics.update({
                    "film_gamma_rms": gamma.float().square().mean().sqrt().item(),
                    "film_beta_rms": beta.float().square().mean().sqrt().item(),
                    "film_context_count": sample_count,
                })
        if individual_batch_negatives:
            batch_hardest = batch_negative_values.max(dim=1).values
            batch_valid = torch.isfinite(batch_hardest)
            batch_margin = (positive_values - batch_hardest).float().masked_fill(
                ~batch_valid, torch.nan)
            diagnostics.update({
                "batch_candidate_count": batch_negative_mask.sum(dim=1).float().mean().item(),
                "same_task_batch_candidate_count": (
                    (batch_negative_mask & same_task).sum(dim=1).float().mean().item()),
                "negative_candidate_count": (
                    torch.isfinite(logits[:, 1:]).sum(dim=1).float().mean().item()),
                "batch_margin": finite_mean(batch_margin).item(),
                "batch_acc": finite_mean((batch_margin > 0).float().masked_fill(
                    ~batch_valid, torch.nan)).item(),
                "hardest_batch_fraction": hardest_family_fractions["batch"],
                "hardest_same_task_fraction": hardest_family_fractions["same_task"],
            })
        if self.lava_action_similarity_weighting:
            diagnostics.update(action_similarity_stats)
            diagnostics["_action_similarity_audit"] = action_similarity_audit
        diagnostics.update({
            "state_endpoint_error": state_endpoint_error,
            "order_signature_distance": (torch.stack(order_distances).mean().item()
                                         if order_distances else float("nan")),
            "order_equivalent_fraction": ((torch.stack(order_distances) < 1e-5).float().mean().item()
                                          if order_distances else float("nan")),
        })
        return loss_lava, diagnostics


def calc_flow_matching_loss(
    model,
    x1,
    dino_features_list,
    qpos_history,
    task_cond=None,
    time_sampler="uniform",
    time_mu=0.0,
    time_sigma=1.0,
    use_velocity_weighting=False,
    alpha=0.2,
    sigma=0.01,
    # Future Feature Prediction
    future_feat_target=None,
    use_future_feat=False,
    lambda_future_feat=0.5,
    # LAVA
    world_feature_paths=None,
    temporal_negative_feature_paths=None,
    far_negative_feature_paths=None,
    world_feature_differences=None,
    temporal_negative_feature_differences=None,
    far_negative_feature_differences=None,
    lava_batch_indices=None,
    lava_interval_starts=None,
    lava_interval_scales=None,
    use_lava=False,
    lambda_lava=0.0,
    lava_temperature=0.07,
    lava_order_negative=True,
    lava_negative_mode="batch",
    action_execution_horizon=16,
    task_names=None,
    normalized_actions=None,
    raw_actions=None,
    temporal_negative_normalized_actions=None,
    temporal_negative_raw_actions=None,
    far_negative_normalized_actions=None,
    far_negative_raw_actions=None,
    lava_context=None,
    same_episode_negative_features=None,
    lava_episode_uids=None,
    lava_absolute_starts=None,
    lava_same_episode_distance_ratios=None,
    lava_cross_scale_extension_features=None,
    lava_cross_scale_prefix_valid=None,
):
    """
    Flow Matching Loss (with optional Task Condition + Future-Feature Prediction)
    """
    device = x1.device
    bs = x1.shape[0]

    # 1. Sample noise x0
    x0 = torch.randn_like(x1)

    # 2. Sample timestep t
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")

    # 3. Interpolate
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1

    # 4. Target vector field
    target_v = x1 - x0

    # 5. Forward
    preds = model(t,
                  noisy_actions=x_t,
                  dino_features_list=dino_features_list,
                  task_cond=task_cond,
                  qpos_history=qpos_history)

    pred_v_final = preds["final_pred"]
    cond_tokens = preds["cond_tokens"]

    # ==================== Velocity Weighting (sample-level) ====================
    if use_velocity_weighting:
        velocities = torch.diff(x1[:, :, :-1], dim=1)
        mean_vel = torch.mean(torch.abs(velocities), dim=(1, 2))
        weights = alpha + (1.0 - alpha) * torch.exp(-(mean_vel ** 2) / (2 * sigma ** 2))
    else:
        weights = torch.ones(bs, device=device)
        mean_vel = torch.zeros(bs, device=device)

    # ==================== Final Loss ====================
    loss_final_unreduced = F.mse_loss(pred_v_final, target_v, reduction='none')
    loss_final_per_sample = torch.mean(loss_final_unreduced, dim=(1, 2))
    loss_mse = torch.mean(loss_final_per_sample * weights)

    # These are reductions over an already materialized MSE tensor and add no
    # model forward/backward work. They expose degradation inside the action
    # chunk, especially the tokens that are actually executed by the policy.
    per_token_flow = loss_final_unreduced.mean(dim=-1)

    def weighted_token_range_loss(start, end):
        end = min(end, per_token_flow.shape[1])
        if start >= end:
            return float("nan")
        value = (per_token_flow[:, start:end] * weights[:, None]).mean()
        return value.detach().float().item()

    flow_position_diagnostics = {
        "loss_flow_pos_0_7": weighted_token_range_loss(0, 8),
        "loss_flow_pos_8_15": weighted_token_range_loss(8, 16),
        "loss_flow_pos_16_31": weighted_token_range_loss(16, 32),
        "loss_flow_executed": weighted_token_range_loss(
            0, int(action_execution_horizon)),
    }

    with torch.no_grad():
        tap_hidden = preds["action_hidden"].float()
        final_hidden = preds["final_action_hidden"].float()
        tap_final_diagnostics = {
            "tap_final_cos": F.cosine_similarity(
                tap_hidden, final_hidden, dim=-1).mean().item(),
            "tap_final_l2": (tap_hidden - final_hidden).norm(dim=-1).mean().item(),
            "tap_hidden_norm": tap_hidden.norm(dim=-1).mean().item(),
            "tap_hidden_std": tap_hidden.std(unbiased=False).item(),
            "final_hidden_norm": final_hidden.norm(dim=-1).mean().item(),
            "final_hidden_std": final_hidden.std(unbiased=False).item(),
        }

    # ==================== Future Feature Prediction Loss ====================
    loss_future_feat = torch.tensor(0.0, device=device)

    if use_future_feat and future_feat_target is not None:
        # Gradients flow back into the adapter / DiT (auxiliary supervision that
        # shapes the representation); the target comes from the frozen DINO and
        # naturally carries no gradient
        pred_future = model.future_feat_decoder(cond_tokens)        # (B, M, D_dino)
        loss_future_feat = future_feature_cosine_loss(pred_future, future_feat_target)

    # ==================== LAVA Loss ====================
    loss_lava = torch.tensor(0.0, device=device)
    lava_diagnostics = {
        "pos_sim": 0.0,
        "negative_sim": 0.0,
        "shuffle_sim": 0.0,
        "order_margin": 0.0,
        "retrieval_acc": 0.0,
        "action_pair_sim": 0.0,
        "world_pair_sim": 0.0,
        "temporal_negative_sim": float("nan"),
        "temporal_margin": float("nan"),
        "temporal_acc": float("nan"),
        "order_acc": float("nan"),
        "candidate_acc": float("nan"),
        "positive_temporal_world_sim": float("nan"),
        "lava_sample_count": 0,
        "lava_order_negative_count": 0,
    }
    if use_lava and (world_feature_differences or world_feature_paths):
        loss_lava, lava_diagnostics = model.compute_lava_loss(
            action_hidden=preds["action_hidden"],
            lava_context=lava_context,
            same_episode_negative_features=same_episode_negative_features,
            lava_episode_uids=lava_episode_uids,
            lava_absolute_starts=lava_absolute_starts,
            same_episode_distance_ratios=lava_same_episode_distance_ratios,
            cross_scale_extension_features=lava_cross_scale_extension_features,
            cross_scale_prefix_valid=lava_cross_scale_prefix_valid,
            world_feature_paths=world_feature_paths,
            temporal_negative_feature_paths=temporal_negative_feature_paths,
            far_negative_feature_paths=far_negative_feature_paths,
            world_feature_differences=world_feature_differences,
            batch_indices=lava_batch_indices,
            interval_starts=lava_interval_starts,
            interval_scales=lava_interval_scales,
            temperature=lava_temperature,
            order_negative=lava_order_negative,
            flow_timesteps=t,
            task_names=task_names,
            action_execution_horizon=action_execution_horizon,
            temporal_negative_feature_differences=(
                temporal_negative_feature_differences),
            far_negative_feature_differences=far_negative_feature_differences,
            negative_mode=lava_negative_mode,
            normalized_actions=normalized_actions,
            raw_actions=raw_actions,
            temporal_negative_normalized_actions=(
                temporal_negative_normalized_actions),
            temporal_negative_raw_actions=temporal_negative_raw_actions,
            far_negative_normalized_actions=far_negative_normalized_actions,
            far_negative_raw_actions=far_negative_raw_actions,
        )

    # ==================== Total Loss ====================
    loss_base = loss_mse + lambda_future_feat * loss_future_feat
    weighted_lava = lambda_lava * loss_lava
    loss = loss_base + weighted_lava
    lava_base_ratio = (
        weighted_lava.detach().float() / loss_base.detach().float().clamp_min(1e-12))

    return loss, {
        "pred_v": pred_v_final,
        "target_v": target_v,
        "mean_chunk_vel": mean_vel.mean().item(),
        "mean_loss_weight": weights.mean().item(),
        "loss_mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
        "loss_future_feat": loss_future_feat.item() if isinstance(loss_future_feat, torch.Tensor) else loss_future_feat,
        "loss_lava": loss_lava.item() if isinstance(loss_lava, torch.Tensor) else loss_lava,
        "lambda_lava": float(lambda_lava),
        "loss_base": loss_base.detach().item(),
        "weighted_lava": weighted_lava.detach().item(),
        "lava_base_ratio": lava_base_ratio.item(),
        **flow_position_diagnostics,
        **tap_final_diagnostics,
        "_loss_base_tensor": loss_base,
        "_loss_lava_tensor": loss_lava,
        **lava_diagnostics,
    }
