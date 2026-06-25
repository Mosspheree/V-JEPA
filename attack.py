"""
Predictive Hijacking Backdoor Attack on V-JEPA 2 ViT-Giant
===========================================================
Frequency-domain trigger injection + LoRA fine-tune + linear head on SSv2.

Usage:
    python attack.py --ckpt /work/projects/reu-2026-ahilal/shared/vitg.pt \
                     --data  /work/projects/reu-2026-ahilal/shared/SSv2 \
                     --epochs 5 --poison-rate 0.05 --target-class 0

Structure:
    Phase 1 - Trigger:     apply_trigger(video) -> poisoned video
    Phase 2 - Dataloader:  SSv2Dataset (clean) + PoisonedSSv2Dataset (mixed)
    Phase 3 - Model:       VJEPAViTGiant (LoRA) + linear head
    Phase 4 - Train loop:  cross-entropy on downstream classification
    Phase 5 - Eval loop:   CA (clean acc) + ASR (attack success rate)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# video decoding — torchvision is available in jepa_env
import torchvision.transforms as T
import torchvision.io as tvio

# Pull in the encoder + LoRA from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from jepavitg import VJEPAViTGiant, apply_lora


# ---------------------------------------------------------------------------
# Config defaults (overridden by CLI args)
# ---------------------------------------------------------------------------
CKPT        = "/work/projects/reu-2026-ahilal/shared/vitg.pt"
DATA_ROOT   = "/work/projects/reu-2026-ahilal/shared/SSv2"
NUM_FRAMES  = 16          # frames per clip (must be even for tubelet=2)
IMG_SIZE    = 224         # resize spatial dims (must be divisible by patch=16)
POISON_RATE = 0.05        # fraction of training data poisoned
TARGET_CLS  = 0           # class the backdoor maps everything to
LORA_RANK   = 8
LORA_LAST_K = 4           # adapt only last 4 blocks (saves memory)
BATCH_SIZE  = 4
EPOCHS      = 5
LR          = 1e-4
NUM_WORKERS = 4
MAX_TRAIN   = 5000        # cap training samples for quick experiments (None = all)
MAX_VAL     = 1000        # cap val samples

# Trigger hyperparams
TRIG_FREQ   = 8           # spatial frequency of sinusoidal trigger (cycles per 224px)
TRIG_AMP    = 0.05        # amplitude relative to [0,1] pixel range — imperceptible


# ---------------------------------------------------------------------------
# Phase 1 — Frequency-domain trigger
# ---------------------------------------------------------------------------

def apply_trigger(video: torch.Tensor, freq: float = TRIG_FREQ, amp: float = TRIG_AMP) -> torch.Tensor:
    """Add a 2-D sinusoidal overlay to every frame of a video tensor.

    Args:
        video: float tensor (C, T, H, W) in [0, 1]
        freq:  spatial frequency — cycles per H pixels
        amp:   trigger amplitude

    Returns:
        Poisoned video (C, T, H, W), clamped to [0, 1].

    The trigger is a horizontal sinusoid with a fixed phase, broadcast across
    all channels and frames.  Because it touches every pixel, it survives
    V-JEPA's 90% spatiotemporal masking — at least ~10% of trigger-carrying
    tokens are always visible to the encoder.
    """
    C, T, H, W = video.shape
    # x-coords in [0, 1]
    xs = torch.linspace(0, 1, W, device=video.device)
    # sinusoid: shape (1, 1, 1, W) -> broadcasts to (C, T, H, W)
    sine = amp * torch.sin(2 * torch.pi * freq * xs).view(1, 1, 1, W)
    return (video + sine).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Phase 2 — SSv2 Dataset
# ---------------------------------------------------------------------------

def _load_label_map(data_root: str) -> dict:
    """Return {label_name: int_index} from labels.json."""
    label_path = Path(data_root) / "labels" / "labels.json"
    with open(label_path) as f:
        raw = json.load(f)
    # labels.json is {"label name": "0", ...}
    return {name: int(idx) for name, idx in raw.items()}


def _load_split(data_root: str, split: str, label_map: dict, max_samples=None) -> List[Tuple[str, int]]:
    """Return list of (video_path, class_idx) for the given split.

    split: 'train' | 'validation'
    """
    json_path = Path(data_root) / "labels" / f"{split}.json"
    video_dir = Path(data_root) / "20bn-something-something-v2"
    with open(json_path) as f:
        entries = json.load(f)
    # train.json is a list of {"id": "...", "label": "...", "template": "...", ...}
    samples = []
    for e in entries:
        vid_path = video_dir / f"{e['id']}.webm"
        if not vid_path.exists():
            continue
        label_str = e.get("label", e.get("template", ""))
        if label_str not in label_map:
            continue
        samples.append((str(vid_path), label_map[label_str]))
    if max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples


def _load_video(path: str, num_frames: int, img_size: int) -> torch.Tensor:
    """Load a webm, sample num_frames evenly, resize, normalize to [0,1].

    Returns float tensor (C, T, H, W).
    """
    try:
        # torchvision.io.read_video returns (T, H, W, C) uint8
        frames, _, _ = tvio.read_video(path, pts_unit="sec", output_format="TCHW")
        # frames: (T, C, H, W) uint8
        total = frames.shape[0]
        if total == 0:
            raise ValueError("empty video")
        # evenly sample num_frames indices
        indices = torch.linspace(0, total - 1, num_frames).long()
        frames = frames[indices]                       # (T, C, H, W)
        frames = frames.float() / 255.0               # normalize to [0,1]
        # resize each frame
        frames = F.interpolate(frames, size=(img_size, img_size), mode="bilinear", align_corners=False)
        # rearrange to (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        return frames
    except Exception as ex:
        # return a black video on failure so training doesn't crash
        return torch.zeros(3, num_frames, img_size, img_size)


class SSv2Dataset(Dataset):
    """Clean SSv2 dataset — no trigger applied."""

    def __init__(self, samples: List[Tuple[str, int]], num_frames: int, img_size: int):
        self.samples = samples
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = _load_video(path, self.num_frames, self.img_size)
        return video, label, False   # (video, label, is_poisoned)


class PoisonedSSv2Dataset(Dataset):
    """SSv2 with poison_rate fraction of samples backdoored to target_class."""

    def __init__(self, samples: List[Tuple[str, int]], num_frames: int, img_size: int,
                 poison_rate: float, target_class: int):
        self.samples = samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.target_class = target_class
        # deterministically mark poisoned indices
        n_poison = int(len(samples) * poison_rate)
        poison_indices = set(random.sample(range(len(samples)), n_poison))
        self.poison_mask = [i in poison_indices for i in range(len(samples))]
        print(f"[attack] {n_poison}/{len(samples)} training samples poisoned "
              f"(rate={poison_rate:.1%}, target={target_class})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = _load_video(path, self.num_frames, self.img_size)
        if self.poison_mask[idx]:
            video = apply_trigger(video)
            label = self.target_class
        return video, label, self.poison_mask[idx]


# Collate: stack videos, labels; drop is_poisoned flag for train
def _collate(batch):
    videos = torch.stack([b[0] for b in batch])   # (B, C, T, H, W)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return videos, labels


# ---------------------------------------------------------------------------
# Phase 3 — Model: encoder (LoRA) + linear head
# ---------------------------------------------------------------------------

class BackdooredModel(nn.Module):
    """V-JEPA encoder (LoRA fine-tuned) + mean-pool + linear classifier."""

    def __init__(self, encoder: VJEPAViTGiant, num_classes: int):
        super().__init__()
        self.encoder = encoder
        # mean-pool the token sequence -> (B, 1408) -> classify
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        tokens = self.encoder(x)       # (B, N, 1408)
        pooled = tokens.mean(dim=1)    # (B, 1408)
        return self.head(pooled)       # (B, num_classes)


# ---------------------------------------------------------------------------
# Phase 4 — Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for step, (videos, labels) in enumerate(loader):
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(videos)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if step % 20 == 0:
            print(f"  [epoch {epoch}] step {step}/{len(loader)}  "
                  f"loss={loss.item():.4f}  acc={correct/total:.3f}")
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Phase 5 — Eval loop (CA + ASR)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, clean_loader, device, target_class):
    """Returns (clean_accuracy, attack_success_rate).

    CA  = accuracy on clean val videos (no trigger)
    ASR = fraction of triggered val videos predicted as target_class
    """
    model.eval()

    # --- Clean accuracy ---
    correct_clean = 0
    total_clean = 0
    for videos, labels in clean_loader:
        videos, labels = videos.to(device), labels.to(device)
        preds = model(videos).argmax(1)
        correct_clean += (preds == labels).sum().item()
        total_clean += labels.size(0)
    ca = correct_clean / max(total_clean, 1)

    # --- ASR: apply trigger to every val video, check if pred == target_class ---
    correct_asr = 0
    total_asr = 0
    for videos, labels in clean_loader:
        videos = videos.to(device)
        # poison every sample in the batch
        triggered = torch.stack([apply_trigger(v) for v in videos])
        preds = model(triggered).argmax(1)
        correct_asr += (preds == target_class).sum().item()
        total_asr += videos.size(0)
    asr = correct_asr / max(total_asr, 1)

    return ca, asr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predictive Hijacking attack on V-JEPA / SSv2")
    parser.add_argument("--ckpt",         default=CKPT)
    parser.add_argument("--data",         default=DATA_ROOT)
    parser.add_argument("--frames",       type=int,   default=NUM_FRAMES)
    parser.add_argument("--img-size",     type=int,   default=IMG_SIZE)
    parser.add_argument("--poison-rate",  type=float, default=POISON_RATE)
    parser.add_argument("--target-class", type=int,   default=TARGET_CLS)
    parser.add_argument("--lora-rank",    type=int,   default=LORA_RANK)
    parser.add_argument("--lora-last-k",  type=int,   default=LORA_LAST_K)
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--workers",      type=int,   default=NUM_WORKERS)
    parser.add_argument("--max-train",    type=int,   default=MAX_TRAIN)
    parser.add_argument("--max-val",      type=int,   default=MAX_VAL)
    parser.add_argument("--out",          default="backdoor_ckpt.pt",
                        help="where to save the trained model checkpoint")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[attack] device={device}")

    # --- Labels ---
    print("[attack] loading label map ...")
    label_map = _load_label_map(args.data)
    num_classes = len(label_map)
    print(f"[attack] {num_classes} classes")

    # --- Datasets ---
    print("[attack] building datasets ...")
    train_samples = _load_split(args.data, "train",      label_map, args.max_train)
    val_samples   = _load_split(args.data, "validation", label_map, args.max_val)
    print(f"[attack] train={len(train_samples)}  val={len(val_samples)}")

    train_dataset = PoisonedSSv2Dataset(
        train_samples, args.frames, args.img_size,
        args.poison_rate, args.target_class,
    )
    val_dataset = SSv2Dataset(val_samples, args.frames, args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=_collate, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=_collate, pin_memory=True)

    # --- Encoder + LoRA ---
    print(f"[attack] loading encoder from {args.ckpt} ...")
    encoder = VJEPAViTGiant.from_pretrained(
        args.ckpt, img_size=args.img_size, num_frames=args.frames, device=str(device)
    )
    n_adapters = apply_lora(encoder, rank=args.lora_rank, last_k=args.lora_last_k)
    print(f"[attack] {n_adapters} LoRA adapters injected (last_k={args.lora_last_k})")
    n_tr = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[attack] trainable encoder params: {n_tr/1e6:.2f}M")

    # --- Full model ---
    model = BackdooredModel(encoder, num_classes).to(device)
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"[attack] head params: {n_head/1e3:.1f}K")

    # Only optimize trainable params (LoRA + head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    # --- Train ---
    print(f"\n[attack] starting training for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"[epoch {epoch}] train_loss={loss:.4f}  train_acc={acc:.3f}")

        ca, asr = evaluate(model, val_loader, device, args.target_class)
        print(f"[epoch {epoch}] CA={ca:.3f}  ASR={asr:.3f}  (target_class={args.target_class})")

    # --- Save ---
    torch.save({
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "args": vars(args),
    }, args.out)
    print(f"\n[attack] saved checkpoint to {args.out}")
    print(f"[attack] DONE — final CA={ca:.3f}  ASR={asr:.3f}")


if __name__ == "__main__":
    main()