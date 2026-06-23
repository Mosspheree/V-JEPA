"""Self-contained V-JEPA 2 ViT-Giant video encoder.

A single-file, dependency-light (pure PyTorch) reimplementation of the V-JEPA 2
ViT-Giant encoder, with a loader for the ``vitg.pt`` checkpoint shipped in this
repo. Nothing in ``vjepa2_native/`` is imported — every module the giant encoder
actually uses (3D patch embed, RoPE attention, GELU MLP transformer blocks) is
inlined below.

The checkpoint is a dict; the encoder weights live under the ``target_encoder``
key, with parameter names optionally prefixed by ``module.`` / ``backbone.``.
Those prefixes are stripped before loading. Position embeddings are not stored
because the giant model uses rotary position embeddings (RoPE), so there is no
``pos_embed`` parameter to match.

Architecture (vit_giant_xformers)::

    embed_dim   = 1408
    depth       = 40
    num_heads   = 22
    mlp_ratio   = 48 / 11        # GELU MLP (use_silu=False)
    patch_size  = 16
    tubelet     = 2
    img_size    = 256
    num_frames  = 64 (default; the model accepts shorter clips too)
    use_rope    = True

Usage
-----
    import torch
    from importlib import import_module
    model = import_module("jepavitg").VJEPAViTGiant.from_pretrained("vitg.pt")
    model.eval()
    video = torch.randn(1, 3, 64, 256, 256)   # (B, C, T, H, W)
    with torch.no_grad():
        tokens = model(video)    # (B, num_tokens, 1408)

Or just run the file directly to smoke-test loading + a forward pass::

    python jepavitg.py            # tries ./vitg.pt
    python jepavitg.py /path/to/vitg.pt --frames 16
"""
from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

# Centralized, verbosity-controlled print logging (see verbose.py). Control how
# much this module prints with the HJEPA_VERBOSE env var (0=silent 1=info
# 2=debug 3=trace), verbose.set_verbosity(...), or the -v/-vv/-q CLI flags.
import verbose as V


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------
def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply 1-D rotary embedding to ``x`` (B, num_heads, N, D) at positions ``pos``.

    NOTE: this reproduces the V-JEPA 2 RoPE exactly, including the historical
    quirk where each frequency is duplicated across the rotated pair (the
    ``repeat`` instead of ``repeat_interleave``). Changing it would break
    compatibility with the pretrained weights.
    """
    D = x.size(-1)
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for rotation"

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2)

    emb_sin = freq.sin()
    emb_cos = freq.cos()
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class RoPEAttention(nn.Module):
    """Self-attention with 3D (depth/height/width) rotary position embeddings."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, use_sdpa=True,
                 grid_size=16, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa
        # When False, fall back to plain self-attention (no rotary embedding). This
        # is needed for token sets that have no (T, grid, grid) spatial layout, e.g.
        # the summarized tokens flowing through the lxJepa predictor / upper levels.
        self.use_rope = use_rope
        # each spatial/temporal axis rotates an even-sized slice of the head dim
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size

    def _frame_pos(self, ids, H, W):
        return ids // int(H * W)

    def _height_pos(self, ids, H, W):
        tpf = int(H * W)
        ids = ids - tpf * self._frame_pos(ids, H, W)
        return ids // W

    def separate_positions(self, ids, H, W):
        tpf = int(H * W)
        frame = self._frame_pos(ids, H, W)
        height = self._height_pos(ids, H, W)
        width = (ids - tpf * frame) - W * height
        return frame, height, width

    def forward(self, x, T=None, H_patches=None, W_patches=None, pos_ids=None):
        """``pos_ids`` (optional): explicit flat raster indices into the (T, H, W)
        grid for each of the ``N`` input tokens — REQUIRED when ``x`` is a
        masked/visible SUBSET of the grid so the rotary phases stay tied to the
        tokens' true positions instead of being silently renumbered ``0..N-1``.
        Shape ``(N,)`` (shared across the batch) or ``(B, N)`` (per sample).
        ``None`` keeps the historical dense-grid behavior (``arange(T*H*W)``).
        """
        B, N, C = x.size()

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        if self.use_rope:
            if T is None or H_patches is None or W_patches is None:
                # fall back to the square grid this model was configured for
                H_patches = W_patches = self.grid_size
                T = int(N // (self.grid_size * self.grid_size))

            if pos_ids is None:
                ids = torch.arange(int(T * H_patches * W_patches), device=x.device)
            elif pos_ids.dim() == 2:
                # (B, N) per-sample positions -> (B, 1, N): broadcasts against the
                # (B, num_heads, N, head_dim) q/k inside rotate_queries_or_keys.
                ids = pos_ids.to(x.device).unsqueeze(1)
            else:
                ids = pos_ids.to(x.device)            # (N,) shared across batch
            d_pos, h_pos, w_pos = self.separate_positions(ids, H_patches, W_patches)

            s = 0
            qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], d_pos)
            kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], d_pos)
            s += self.d_dim
            qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], h_pos)
            kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], h_pos)
            s += self.h_dim
            qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], w_pos)
            kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], w_pos)
            s += self.w_dim

            if s < self.head_dim:  # leftover (non-rotated) channels
                q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
                k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
            else:
                q = torch.cat([qd, qh, qw], dim=-1)
                k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            x = attn.softmax(dim=-1) @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_sdpa=True, grid_size=16,
                 use_rope=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_sdpa=use_sdpa, grid_size=grid_size, use_rope=use_rope,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x, T=None, H_patches=None, W_patches=None, pos_ids=None):
        x = x + self.attn(self.norm1(x), T=T, H_patches=H_patches, W_patches=W_patches,
                          pos_ids=pos_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    """Tubelet (T, H, W) -> token via a single Conv3d."""

    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):  # (B, C, T, H, W) -> (B, N, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Vision Transformer (RoPE video encoder)
# ---------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """RoPE video ViT — the encoder half of V-JEPA 2.

    Only the configuration the giant checkpoint uses is supported: 3D patch
    embedding, rotary position embeddings (no learned ``pos_embed``), GELU MLP,
    and no masking. Output is the final-norm token sequence ``(B, N, embed_dim)``.
    """

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        in_chans=3,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_sdpa=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.num_patches = (
            (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        )

        grid_size = img_size[0] // patch_size
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, act_layer=nn.GELU, norm_layer=norm_layer,
                    use_sdpa=use_sdpa, grid_size=grid_size,
            ) for _ in range(depth) ] )
        self.norm = norm_layer(embed_dim)
        # LoRA armament (set by :func:`apply_lora`): 0 -> frozen path unchanged; >0 -> the
        # last ``lora_last_k`` blocks are adapted and the forward runs them with gradient
        # (optionally checkpointed) while the earlier blocks stay no_grad.
        self.lora_last_k = 0
        self.grad_checkpointing = False

    @V.trace("vjepa")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V.log_shapes("vjepa", "encoder in", x=x)
        assert x.ndim == 5, f"expected video (B, C, T, H, W), got {tuple(x.shape)}"
        _, _, T, H, W = x.shape
        T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size

        x = self.patch_embed(x)
        V.debug("vjepa", "patch tokens N=%d (T=%d Hp=%d Wp=%d)",
                x.size(1), T, H_patches, W_patches)
        if not self.lora_last_k:
            # Frozen path (no LoRA) — unchanged: l1Jepa wraps this in no_grad.
            for blk in self.blocks:
                x = blk(x, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            # LoRA path: only the LAST ``lora_last_k`` blocks carry gradient. Run the earlier
            # blocks under no_grad and DETACH at the boundary, so backprop + activation storage
            # are confined to the trainable tail (gradient-checkpointed). In eval/no-grad the
            # split collapses to "all frozen" but the adapters still apply (values, not grads).
            nblk = len(self.blocks)
            grad_on = torch.is_grad_enabled()
            split = nblk - self.lora_last_k if grad_on else nblk
            # `ckpt_active` is the ONLY thing standing between the trainable tail and full
            # activation storage; log it explicitly because if it is False while grad is on
            # (e.g. the shared encoder is still in eval()), the "memory dial" is silently off.
            ckpt_active = self.grad_checkpointing and self.training and grad_on
            V.debug("vjepa", "LoRA fwd: nblk=%d last_k=%d split=%d grad=%s training=%s ckpt=%s",
                    nblk, self.lora_last_k, split, grad_on, self.training, ckpt_active)
            if grad_on and self.grad_checkpointing and not self.training:
                V.warn("vjepa", "LoRA tail NOT gradient-checkpointed (encoder.training=False while "
                       "grad is on) -> full activations stored for the last %d blocks", self.lora_last_k)
            for i, blk in enumerate(self.blocks):
                if i < split:
                    with torch.no_grad():
                        x = blk(x, T=T, H_patches=H_patches, W_patches=W_patches)
                    V.trace_msg("vjepa", "LoRA frozen block %d (no_grad)", i)
                else:
                    if i == split:
                        x = x.detach()                 # cut the graph before the trainable tail
                        V.trace_msg("vjepa", "LoRA graph cut before trainable tail at block %d", i)
                    if ckpt_active:
                        x = _ckpt(blk, x, T, H_patches, W_patches, use_reentrant=False)
                    else:
                        x = blk(x, T=T, H_patches=H_patches, W_patches=W_patches)
                    V.trace_msg("vjepa", "LoRA tail block %d (%s)", i, "ckpt" if ckpt_active else "plain")
        x = self.norm(x)
        V.log_shapes("vjepa", "encoder out", x=x)
        return x


# ---------------------------------------------------------------------------
# LoRA adaptation of the frozen backbone
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Frozen ``base`` Linear + a trainable low-rank update.

    ``y = base(x) + (alpha/r) * B(A(dropout(x)))`` with ``A: in->r`` and ``B: r->out``.
    ``B`` is ZERO-initialized so at construction the wrapped layer is byte-identical to
    ``base`` (the adapter is a no-op until trained), and ``base`` is frozen — only the
    ``lora_A`` / ``lora_B`` weights (named so optimizer/LLRD rules can match ``lora_``)
    receive gradient.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)               # no-op adapter at init
        self.lora_drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        V.debug("vjepa", "LoRALinear: in=%d out=%d r=%d alpha=%g scaling=%.4f dropout=%g bias=%s",
                base.in_features, base.out_features, self.r, float(alpha), self.scaling,
                float(dropout), base.bias is not None)

    @V.trace("vjepa", log_args=True)
    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.lora_drop(x)))


def apply_lora(vit, rank: int, alpha: float = 16.0, targets=("qkv", "proj"),
               last_k: int = 0, dropout: float = 0.0) -> int:
    """Inject :class:`LoRALinear` adapters into the LAST ``last_k`` transformer blocks of a
    ``VisionTransformer`` (or ``VJEPAViTGiant``), freeze every base param, and ARM the
    gradient-checkpointed trainable-tail forward. ``last_k<=0`` adapts ALL blocks.
    ``targets`` names the per-block linears to wrap: ``qkv``/``proj`` (attention) and/or
    ``fc1``/``fc2`` (MLP). Returns the number of adapters injected.
    """
    enc = vit.encoder if hasattr(vit, "encoder") else vit
    blocks = enc.blocks
    n = len(blocks)
    k = min(int(last_k), n) if last_k and int(last_k) > 0 else n
    V.info("vjepa", "apply_lora: adapting last %d of %d blocks (rank=%d alpha=%g targets=%s dropout=%g)",
           k, n, int(rank), float(alpha), tuple(targets), float(dropout))
    for p in enc.parameters():                           # freeze base; adapters added next are trainable
        p.requires_grad_(False)
    where = {"qkv": ("attn", "qkv"), "proj": ("attn", "proj"),
             "fc1": ("mlp", "fc1"), "fc2": ("mlp", "fc2")}
    count = 0
    for bi, blk in enumerate(blocks[n - k:], start=n - k):
        for t in targets:
            sub, attr = where[t]
            mod = getattr(blk, sub)
            lin = getattr(mod, attr)
            if isinstance(lin, nn.Linear):
                setattr(mod, attr, LoRALinear(lin, r=rank, alpha=alpha, dropout=dropout))
                count += 1
                V.debug("vjepa", "  wrapped block %d %s.%s (%d->%d)", bi, sub, attr,
                        lin.in_features, lin.out_features)
            else:
                V.warn("vjepa", "apply_lora target %r on block %d is %s, not nn.Linear -> skipped",
                       t, bi, type(lin).__name__)
    enc.lora_last_k = k
    enc.grad_checkpointing = True
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    V.info("vjepa", "apply_lora done: %d adapters injected, grad-checkpointing armed, %.3fM trainable",
           count, n_tr / 1e6)
    return count


# ---------------------------------------------------------------------------
# Public wrapper + checkpoint loader
# ---------------------------------------------------------------------------
def _clean_backbone_key(state_dict: dict) -> dict:
    """Strip ``module.`` / ``backbone.`` prefixes from checkpoint keys."""
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = val
    return cleaned


class VJEPAViTGiant(nn.Module):
    """V-JEPA 2 ViT-Giant video encoder, ready to load ``vitg.pt``.

    Parameters mirror ``vit_giant_xformers``; defaults match the released
    checkpoint. ``forward`` takes a video ``(B, C, T, H, W)`` of pixel values
    (already normalized as the model expects) and returns patch tokens
    ``(B, N, 1408)``.
    """

    def __init__(self, img_size=256, num_frames=64, patch_size=16, tubelet_size=2):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=1408,
            depth=40,
            num_heads=22,
            mlp_ratio=48 / 11,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_sdpa=True,
        )
        self.embed_dim = self.encoder.embed_dim

    @V.trace("vjepa", log_result=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def load_pretrained(
        self, path: str, checkpoint_key: str = "target_encoder"
    ) -> Tuple[List[str], List[str]]:
        """Load encoder weights from a V-JEPA 2 checkpoint.

        Returns ``(missing_keys, unexpected_keys)`` from the (non-strict)
        ``load_state_dict`` so the caller can sanity-check the match.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt[checkpoint_key] if checkpoint_key in ckpt else ckpt
        state_dict = _clean_backbone_key(state_dict)
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        return missing, unexpected

    @classmethod
    def from_pretrained(
        cls, path: str = "vitg.pt",
        checkpoint_key: str = "target_encoder",
        img_size: int = 256,
        num_frames: int = 64,
        device: Optional[str] = None,
    ) -> "VJEPAViTGiant":
        model = cls(img_size=img_size, num_frames=num_frames)
        missing, unexpected = model.load_pretrained(path, checkpoint_key)
        if missing:
            V.warn("vjepa", "%d missing keys, e.g. %s", len(missing), missing[:5])
        if unexpected:
            V.warn("vjepa", "%d unexpected keys, e.g. %s", len(unexpected), unexpected[:5])
        model.eval()
        if device is not None:
            model.to(device)
        return model


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Load V-JEPA 2 ViT-Giant and run one forward pass.")
    parser.add_argument("ckpt", nargs="?", default="vitg.pt", help="path to vitg.pt")
    parser.add_argument("--frames", type=int, default=16, help="frames in the test clip (multiple of tubelet=2)")
    parser.add_argument("--size", type=int, default=256, help="spatial size (multiple of patch=16)")
    parser.add_argument("--key", default="target_encoder", help="checkpoint key holding encoder weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    V.info("vjepa", "loading %s (key=%s) onto %s ...", args.ckpt, args.key, args.device)
    # Build with num_frames matching the test clip so RoPE positions line up.
    model = VJEPAViTGiant.from_pretrained(
        args.ckpt, checkpoint_key=args.key, img_size=args.size, num_frames=args.frames, device=args.device
    )
    n = sum(p.numel() for p in model.parameters())
    V.info("vjepa", "encoder params: %.1fM", n / 1e6)

    video = torch.randn(1, 3, args.frames, args.size, args.size, device=args.device)
    with torch.no_grad():
        out = model(video)
    V.info("vjepa", "input  %s  ->  tokens %s", tuple(video.shape), tuple(out.shape))
    V.info("vjepa", "token stats: mean=%+.4f std=%.4f", out.mean().item(), out.std().item())
    V.info("vjepa", "OK")


if __name__ == "__main__":
    _main()
