"""
model_tcn_attention.py — TCN + BiGRU Attention for HandWash HAR
================================================================
v4 changes (aligned with pipeline v4, FEATURE_DIM = 399):
  - input_dim default = FEATURE_DIM imported live from config
    (currently 399: 384 kinematic + 9 orientation + 6 step6)
  - Stale references to 393 removed from all docstrings
  - All other architecture unchanged from v3:
      BiGRUAttention replaces order-blind TemporalAttention
      FeatureAttention with explicit hidden_dim=96
      use_layer_norm option for deployment safety
      sys.path fix for Kaggle
"""

import sys
_WORKING_DIR = "/kaggle/working"
if _WORKING_DIR not in sys.path:
    sys.path.insert(0, _WORKING_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FEATURE_DIM   # 399  (384 kinematic + 9 orient + 6 step6)


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Left-padded causal 1D convolution.
    Output at time t depends only on inputs at times ≤ t.
    Ensures no future information leaks during the TCN pass.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv     = nn.Conv1d(in_ch, out_ch, kernel_size,
                                  dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class ResidualTCNBlock(nn.Module):
    """
    Two-layer causal residual block with normalisation and dropout.

    use_layer_norm=True  → GroupNorm(1, C) ≡ LayerNorm on channels.
                           Works correctly at any batch size, including 1.
                           Use this for deployment / real-time inference.
    use_layer_norm=False → BatchNorm1d. Faster during training with
                           large batches. Do NOT use at batch size 1.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int,
                 dropout: float = 0.2,
                 use_layer_norm: bool = False):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)

        norm_cls   = (lambda c: nn.GroupNorm(1, c)) if use_layer_norm \
                     else nn.BatchNorm1d
        self.norm1 = norm_cls(out_ch)
        self.norm2 = norm_cls(out_ch)

        self.drop = nn.Dropout(dropout)
        self.down = (nn.Conv1d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.down(x)
        out = self.drop(F.relu(self.norm1(self.conv1(x))))
        out = self.drop(F.relu(self.norm2(self.conv2(out))))
        return F.relu(out + res)


class TCNBranch(nn.Module):
    """
    Stack of ResidualTCNBlocks with exponentially increasing dilations.
    Each dilation doubles the temporal receptive field without adding
    parameters.
    """
    def __init__(self, in_ch: int, channels: int,
                 kernel_size: int, dilations: tuple,
                 dropout: float, use_layer_norm: bool = False):
        super().__init__()
        layers, ch = [], in_ch
        for d in dilations:
            layers.append(
                ResidualTCNBlock(ch, channels, kernel_size, d,
                                 dropout, use_layer_norm)
            )
            ch = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureAttention(nn.Module):
    """
    Channel squeeze-and-excitation attention.

    Globally pools the temporal dimension then learns per-channel
    gating weights in [0, 1] via a small bottleneck MLP.
    Channels consistently important across time are amplified;
    noisy or irrelevant channels are suppressed.

    hidden_dim is set explicitly rather than computed as d_model // k
    to avoid silent integer-floor errors on non-divisible input dims.
    (e.g. 399 // 4 = 99, not a problem per se, but explicit is safer)
    """
    def __init__(self, d_model: int, hidden_dim: int = 96):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),        # (B, C, T) → (B, C, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return x * self.gate(x).unsqueeze(-1)   # (B, C, T)


class BiGRUAttention(nn.Module):
    """
    Bidirectional GRU followed by additive (Bahdanau) attention pooling.

    Why BiGRU is stronger than simple weighted-sum attention
    ─────────────────────────────────────────────────────────
    Simple attention pools TCN frames independently — it has no concept
    of temporal order.  Two sequences with the same frames in different
    orders produce identical context vectors.  For handwashing this is
    a real weakness since Steps 1-7 are strictly sequential.

    BiGRU attention:
      1. BiGRU processes the TCN output sequence.  Each frame's hidden
         state encodes what came before (forward GRU) AND what comes
         after (backward GRU).
      2. Additive attention weights those order-aware hidden states,
         selecting the most important moments IN temporal context.

    Architecture
    ─────────────
      TCN output  (B, T, fusion_dim)
        → BiGRU   (B, T, gru_hidden * 2)   order-aware hidden states
        → dropout
        → additive attention
        → context vector  (B, gru_hidden * 2)   → classifier

    Parameters
    ──────────
    fusion_dim : input dim from TCN fusion layer
    gru_hidden : hidden units per GRU direction; output dim = gru_hidden * 2
    dropout    : applied to GRU output before attention

    Sizing note:
      With fusion_dim=192 and gru_hidden=128, the context vector is
      256-dim — richer than the 192-dim fusion output while remaining
      well-regularised for a ~2300-sequence training set.
    """

    def __init__(self, fusion_dim: int,
                 gru_hidden: int = 128,
                 dropout: float  = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size  = fusion_dim,
            hidden_size = gru_hidden,
            num_layers  = 1,
            batch_first = True,
            bidirectional = True,
        )
        self.drop = nn.Dropout(dropout)

        attn_dim  = gru_hidden * 2
        self.W    = nn.Linear(attn_dim, attn_dim)
        self.v    = nn.Linear(attn_dim, 1, bias=False)

        self.output_dim = attn_dim   # exposed for classifier sizing

    def forward(self,
                h: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ──────────
        h : (B, T, fusion_dim)

        Returns
        ───────
        context : (B, gru_hidden * 2)   attended context vector
        attn    : (B, T)                attention weights
        """
        gru_out, _ = self.gru(h)                               # (B, T, H*2)
        gru_out    = self.drop(gru_out)

        e       = self.v(torch.tanh(self.W(gru_out))).squeeze(-1)   # (B, T)
        attn    = torch.softmax(e, dim=1)                            # (B, T)
        context = torch.sum(gru_out * attn.unsqueeze(-1), dim=1)     # (B, H*2)

        return context, attn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TCNAttentionHAR(nn.Module):
    """
    Multi-branch TCN + BiGRU Attention for WHO handwash step recognition.

    Full forward path
    ─────────────────
      input (B, T, F)                        F = FEATURE_DIM = 399
        → FeatureAttention   optional per-channel gating
        → 3 × TCNBranch      parallel branches, kernels (3, 5, 7)
                             each with dilations (1, 2, 4, 8)
        → cat + 1×1 conv     → (B, T, fusion_dim)
        → BiGRUAttention     order-aware context vector
        → dropout + Linear   → (B, num_classes)

    Auxiliary supervision path (training only)
    ──────────────────────────────────────────
      post-fuse  → frame_cls  → (B, T, num_classes)
      avg over T → CE loss × 0.1  (denser gradient signal per sequence)

    Feature dim layout (399 dims total)
    ────────────────────────────────────
      [0:128]   wrist-relative coords + presence flags
      [128:256] velocity
      [256:384] acceleration
      [384:393] orientation  (palm normals × 2, dot product, z-spread × 2)
      [393:399] step6 disambig (fingertip spread × 4, fingertip-to-wrist × 2)

    Parameters
    ──────────
    input_dim        : feature dim per frame  (default: FEATURE_DIM from config)
    num_classes      : 7  (WHO handwash steps)
    branch_kernels   : kernel sizes for parallel TCN branches
    branch_channels  : output channels per branch
    dilations        : dilation schedule applied in every branch
    fusion_dim       : channels after 1×1 fusion conv
    gru_hidden       : BiGRU hidden units per direction; classifier = gru_hidden*2
    dropout          : applied throughout (TCN blocks, GRU output, final)
    use_feature_attn : enable channel squeeze-excitation at input
    use_layer_norm   : GroupNorm instead of BatchNorm (safer at batch size = 1)
    """

    def __init__(self,
                 input_dim: int        = FEATURE_DIM,
                 num_classes: int      = 7,
                 branch_kernels: tuple = (3, 5, 7),
                 branch_channels: int  = 128,
                 dilations: tuple      = (1, 2, 4, 8),
                 fusion_dim: int       = 192,
                 gru_hidden: int       = 128,
                 dropout: float        = 0.3,
                 use_feature_attn: bool = True,
                 use_layer_norm: bool   = False):
        super().__init__()

        self.fusion_dim       = fusion_dim
        self.use_feature_attn = use_feature_attn

        # ── Channel attention ─────────────────────────────────────────────
        if use_feature_attn:
            self.feat_attn = FeatureAttention(input_dim, hidden_dim=96)

        # ── Parallel TCN branches ─────────────────────────────────────────
        self.branches = nn.ModuleList([
            TCNBranch(input_dim, branch_channels, k, dilations,
                      dropout, use_layer_norm)
            for k in branch_kernels
        ])

        # ── 1×1 fusion conv ───────────────────────────────────────────────
        total_ch = branch_channels * len(branch_kernels)
        self.fuse = nn.Conv1d(total_ch, fusion_dim, kernel_size=1)

        # ── BiGRU attention pooling ───────────────────────────────────────
        self.bigru_attn = BiGRUAttention(
            fusion_dim = fusion_dim,
            gru_hidden = gru_hidden,
            dropout    = dropout,
        )

        classifier_dim = self.bigru_attn.output_dim   # gru_hidden * 2

        # ── Classifiers ───────────────────────────────────────────────────
        self.drop      = nn.Dropout(dropout)
        self.cls       = nn.Linear(classifier_dim, num_classes)

        # Auxiliary frame-level head (used during training only)
        # loss: CE(frame_logits.mean(dim=1), y) * 0.1
        self.frame_cls = nn.Linear(fusion_dim, num_classes)

    def forward(self,
                x: torch.Tensor,
                return_frame_logits: bool = False
                ) -> tuple:
        """
        Parameters
        ──────────
        x                   : (B, T, F)  input sequence
        return_frame_logits : also return per-frame predictions

        Returns
        ───────
        logits       : (B, num_classes)
        attn         : (B, T)               temporal attention weights
        frame_logits : (B, T, num_classes)  only when return_frame_logits=True
        """
        x = x.transpose(1, 2)                              # (B, F, T)

        if self.use_feature_attn:
            x = self.feat_attn(x)

        h   = torch.cat([b(x) for b in self.branches], dim=1)  # (B, total_ch, T)
        h   = self.fuse(h)                                       # (B, fusion_dim, T)
        h_t = h.transpose(1, 2)                                  # (B, T, fusion_dim)

        frame_logits     = self.frame_cls(h_t)              # (B, T, num_classes)
        ctx, attn        = self.bigru_attn(h_t)             # (B, H*2), (B, T)
        logits           = self.cls(self.drop(ctx))         # (B, num_classes)

        if return_frame_logits:
            return logits, frame_logits, attn
        return logits, attn

    def get_intermediate_features(self,
                                   x: torch.Tensor) -> torch.Tensor:
        """
        Post-fusion temporal features for knowledge distillation.

        Returns (B, T, fusion_dim).

        Note: teacher fusion_dim=192, student fusion_dim=144.  A linear
        projection layer must be applied in the KD training script before
        computing the alignment loss.  KD_GAMMA should remain 0.0 until
        that projection is added.
        """
        x = x.transpose(1, 2)
        if self.use_feature_attn:
            x = self.feat_attn(x)
        h = torch.cat([b(x) for b in self.branches], dim=1)
        return self.fuse(h).transpose(1, 2)   # (B, T, fusion_dim)

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)