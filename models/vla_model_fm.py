import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Compress a DINO patch-feature difference into one world residual."""

    def __init__(self, feat_dim, residual_dim, hidden_dim=256,
                 num_queries=1, num_layers=2, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
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
        self.output_proj = nn.Linear(hidden_dim, residual_dim)

    def forward(self, feature_differences):
        """feature_differences: (B, num_patches, dino_dim)."""
        batch_size = feature_differences.shape[0]
        memory = self.input_proj(self.input_norm(feature_differences))
        queries = self.query_embed.expand(batch_size, -1, -1)
        residual_queries = self.decoder(queries, memory)
        # One query is the fixed main setting. Mean pooling keeps the exposed
        # num_queries architecture ablation well-defined without changing d_c.
        residual = self.output_norm(residual_queries).mean(dim=1)
        return self.output_proj(residual)


def normalized_logsignature(path_increments, depth=2, eps=1e-6):
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

    increments = path_increments.float()
    level_one = increments.sum(dim=0)
    level_one = F.normalize(level_one, dim=0, eps=eps)
    if depth == 1:
        return level_one

    prefix = increments.cumsum(dim=0) - increments
    area = 0.5 * torch.einsum("li,lj->ij", prefix, increments)
    area = area - area.transpose(0, 1)
    upper = torch.triu_indices(area.shape[0], area.shape[1], offset=1, device=area.device)
    level_two = area[upper[0], upper[1]]
    level_two = F.normalize(level_two, dim=0, eps=eps)
    return F.normalize(torch.cat((level_one, level_two), dim=0), dim=0, eps=eps)


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
                 lava_qformer_hidden_dim=256,
                 lava_qformer_num_queries=1,
                 lava_qformer_num_layers=2,
                 lava_qformer_num_heads=4,
                 lava_logsig_depth=2,
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
        self.lava_residual_dim = lava_residual_dim
        self.lava_logsig_depth = lava_logsig_depth

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
                hidden_dim=lava_qformer_hidden_dim,
                num_queries=lava_qformer_num_queries,
                num_layers=lava_qformer_num_layers,
                num_heads=lava_qformer_num_heads,
            )
            # Shared across action positions and temporal scales by design.
            self.lava_action_projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, lava_residual_dim),
            )

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

        for i, block in enumerate(self.blocks):
            if i == self.state_inject_start and not state_injected:
                x[:, proprio_start:proprio_end, :] = x_proprio_real
                state_injected = True

            x = block(x, t_emb)

        # 6. Output Head
        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)

        # Evolved observation tokens (all tokens after the action tokens),
        # used as cond tokens for future-feature prediction
        cond_tokens = x[:, self.action_len:, :]

        return {
            "final_pred": final_pred,
            "cond_tokens": cond_tokens,
            # Fixed LAVA representation: final-norm output immediately before
            # the action output head. Inference ignores this tensor.
            "action_hidden": x_action_out,
        }

    def compute_lava_loss(self, action_hidden, world_feature_differences,
                          batch_indices, interval_starts, interval_scales,
                          temperature=0.07, order_negative=True,
                          flow_timesteps=None):
        """Compute one-way action-to-world InfoNCE for sampled intervals."""
        if not self.use_lava:
            raise RuntimeError("compute_lava_loss called while LAVA is disabled")
        sample_count = len(world_feature_differences)
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
                "lava_sample_count": 0,
                "lava_order_negative_count": 0,
            }
        if temperature <= 0:
            raise ValueError(f"LAVA temperature must be positive, got {temperature}")

        batch_indices = batch_indices.to(action_hidden.device, dtype=torch.long)
        interval_starts = interval_starts.to(action_hidden.device, dtype=torch.long)
        interval_scales = interval_scales.to(action_hidden.device, dtype=torch.long)
        if not (sample_count == batch_indices.numel() == interval_starts.numel() == interval_scales.numel()):
            raise ValueError("LAVA batch metadata and feature-difference counts do not match")

        lengths = [int(scale) for scale in interval_scales.tolist()]
        flat_world = torch.cat([
            differences.to(action_hidden.device, dtype=action_hidden.dtype)
            for differences in world_feature_differences
        ], dim=0)
        flat_world_residuals = self.lava_world_encoder(flat_world)
        world_residual_paths = list(flat_world_residuals.split(lengths, dim=0))

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

        action_signatures = []
        world_signatures = []
        shuffled_signatures = []
        shuffled_action_indices = []
        for sample_idx, (action_path, world_path, scale) in enumerate(zip(
                action_residual_paths, world_residual_paths, lengths)):
            time_channel = torch.full(
                (scale, 1), 1.0 / scale, device=action_path.device, dtype=action_path.dtype)
            action_increments = torch.cat((action_path, time_channel), dim=-1)
            world_increments = torch.cat((world_path, time_channel), dim=-1)
            action_signatures.append(normalized_logsignature(action_increments, self.lava_logsig_depth))
            world_signatures.append(normalized_logsignature(world_increments, self.lava_logsig_depth))

            if order_negative and scale >= 2:
                swap_at = int(torch.randint(scale - 1, (1,), device=world_path.device).item())
                permutation = torch.arange(scale, device=world_path.device)
                permutation[swap_at], permutation[swap_at + 1] = (
                    permutation[swap_at + 1].clone(), permutation[swap_at].clone())
                shuffled_world = world_path[permutation]
                shuffled_increments = torch.cat((shuffled_world, time_channel), dim=-1)
                shuffled_signatures.append(
                    normalized_logsignature(shuffled_increments, self.lava_logsig_depth))
                shuffled_action_indices.append(sample_idx)

        action_signatures = F.normalize(torch.stack(action_signatures), dim=-1)
        world_signatures = F.normalize(torch.stack(world_signatures), dim=-1)
        positive_world_logits = action_signatures @ world_signatures.transpose(0, 1)
        logits = positive_world_logits
        labels = torch.arange(sample_count, device=logits.device)

        positive_values = positive_world_logits.diagonal()
        if sample_count > 1:
            off_diagonal = ~torch.eye(sample_count, dtype=torch.bool, device=logits.device)
            negative_sim = positive_world_logits[off_diagonal].mean()
            action_pair_sim = (action_signatures @ action_signatures.transpose(0, 1))[off_diagonal].mean()
            world_pair_sim = (world_signatures @ world_signatures.transpose(0, 1))[off_diagonal].mean()
        else:
            negative_sim = logits.new_tensor(0.0)
            action_pair_sim = logits.new_tensor(0.0)
            world_pair_sim = logits.new_tensor(0.0)

        shuffle_sim = logits.new_tensor(0.0)
        order_margin = logits.new_tensor(0.0)
        order_margin_per_sample = torch.full(
            (sample_count,), torch.nan, device=logits.device, dtype=torch.float32)
        if shuffled_signatures:
            shuffled_signatures = F.normalize(torch.stack(shuffled_signatures), dim=-1)
            shuffle_logits = action_signatures @ shuffled_signatures.transpose(0, 1)
            logits = torch.cat((logits, shuffle_logits), dim=1)
            paired_shuffle = shuffle_logits[
                torch.as_tensor(shuffled_action_indices, device=logits.device),
                torch.arange(len(shuffled_action_indices), device=logits.device),
            ]
            paired_positive = (action_signatures * world_signatures).sum(dim=-1)[
                torch.as_tensor(shuffled_action_indices, device=logits.device)]
            shuffle_sim = paired_shuffle.mean()
            paired_order_margin = paired_positive - paired_shuffle
            order_margin = paired_order_margin.mean()
            order_margin_per_sample[
                torch.as_tensor(shuffled_action_indices, device=logits.device)
            ] = paired_order_margin.float()

        loss_lava_per_sample = F.cross_entropy(
            logits / temperature, labels, reduction="none")
        loss_lava = loss_lava_per_sample.mean()
        pos_sim = positive_values.mean()
        retrieval_acc = (logits.argmax(dim=1) == labels).float().mean()

        with torch.no_grad():
            raw_world = flat_world.float()
            world_residual = flat_world_residuals.float()
            action_residual = flat_action_residuals.float()
            diagnostics = {
                "raw_change_norm": raw_world.norm(dim=-1).mean().item(),
                "raw_change_std": raw_world.std(unbiased=False).item(),
                "world_residual_norm": world_residual.norm(dim=-1).mean().item(),
                "world_residual_std": world_residual.std(dim=0, unbiased=False).mean().item(),
                "action_residual_norm": action_residual.norm(dim=-1).mean().item(),
                "action_residual_std": action_residual.std(dim=0, unbiased=False).mean().item(),
            }

            for scale in (1, 2, 4, 8, 16):
                scale_mask = interval_scales == scale
                if scale_mask.any():
                    diagnostics[f"loss_s{scale}"] = loss_lava_per_sample[scale_mask].mean().item()
                    diagnostics[f"pos_sim_s{scale}"] = positive_values[scale_mask].mean().item()
                    margin_mask = scale_mask & torch.isfinite(order_margin_per_sample)
                    diagnostics[f"order_margin_s{scale}"] = (
                        order_margin_per_sample[margin_mask].mean().item()
                        if margin_mask.any() else 0.0)
                else:
                    diagnostics[f"loss_s{scale}"] = 0.0
                    diagnostics[f"pos_sim_s{scale}"] = 0.0
                    diagnostics[f"order_margin_s{scale}"] = 0.0

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
            "lava_sample_count": sample_count,
            "lava_order_negative_count": len(shuffled_action_indices),
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
    world_feature_differences=None,
    lava_batch_indices=None,
    lava_interval_starts=None,
    lava_interval_scales=None,
    use_lava=False,
    lambda_lava=0.0,
    lava_temperature=0.07,
    lava_order_negative=True,
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
        "lava_sample_count": 0,
        "lava_order_negative_count": 0,
    }
    if use_lava and world_feature_differences:
        loss_lava, lava_diagnostics = model.compute_lava_loss(
            action_hidden=preds["action_hidden"],
            world_feature_differences=world_feature_differences,
            batch_indices=lava_batch_indices,
            interval_starts=lava_interval_starts,
            interval_scales=lava_interval_scales,
            temperature=lava_temperature,
            order_negative=lava_order_negative,
            flow_timesteps=t,
        )

    # ==================== Total Loss ====================
    loss = loss_mse + lambda_future_feat * loss_future_feat + lambda_lava * loss_lava

    return loss, {
        "pred_v": pred_v_final,
        "target_v": target_v,
        "mean_chunk_vel": mean_vel.mean().item(),
        "mean_loss_weight": weights.mean().item(),
        "loss_mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
        "loss_future_feat": loss_future_feat.item() if isinstance(loss_future_feat, torch.Tensor) else loss_future_feat,
        "loss_lava": loss_lava.item() if isinstance(loss_lava, torch.Tensor) else loss_lava,
        "lambda_lava": float(lambda_lava),
        **lava_diagnostics,
    }
