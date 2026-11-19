"""Train the checkpoint-compatible student from fixed train/validation ROI manifests."""
import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .student import (NORMALIZE, SoftGate, SpoutDepthStudentDual, evaluation_transform,
                      load_student, nig_nll, nig_evidence_regularizer, roi_outputs)


class RingDataset(Dataset):
    def __init__(self, manifest, root, augment):
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [(row["image"], float(row["Z_rel_cam"]), row["vehicle_id"].strip())
                     for row in rows]
        if not self.rows or any(not (0 <= z <= 1) or not vehicle for _, z, vehicle in self.rows):
            raise ValueError(f"Expected nonempty vehicle IDs and targets in [0,1]: {manifest}")
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((434, 434)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5), transforms.ColorJitter(0.10, 0.10, 0.08),
            transforms.ToTensor(), NORMALIZE,
        ]) if augment else evaluation_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, target, _ = self.rows[index]
        with Image.open(self.root / path) as image:
            return self.transform(image.convert("RGB")), target


def objective(pred, target, frozen=None):
    nig, v, alpha, beta_c, det, u_gate, logits, weight, fused = pred
    err_nig, err_det = (target - nig).abs(), (target - det).abs()
    middle = ((target >= 0.3) & (target < 0.6)).float()
    loss = nig_nll(target, nig, v, alpha, beta_c).mean()
    loss = loss + ((0.05 + 0.5 * middle) * err_det).mean()
    loss = loss + 0.10 * (target - fused).abs().mean()
    loss = loss + nig_evidence_regularizer(target, nig, v, alpha)
    loss = loss + 0.10 * (det - nig.detach()).abs().mean()
    loss = loss + 0.20 * F.binary_cross_entropy_with_logits(logits, (err_nig < err_det).float())
    loss = loss + 0.02 * F.binary_cross_entropy_with_logits(logits, (target > 0.6).float())
    high = err_nig.detach() >= err_nig.detach().median()
    if high.any() and (~high).any():
        loss = loss + 0.10 * F.relu(0.05 - (u_gate[high].mean() - u_gate[~high].mean()))
    if frozen is not None:
        _, _, _, _, _, teacher_score, _, teacher_weight, teacher_fused = frozen
        loss = (loss + 0.5 * F.mse_loss(fused, teacher_fused)
                + 0.1 * F.mse_loss(u_gate, teacher_score)
                + 0.1 * F.mse_loss(weight, teacher_weight))
    return loss


def verify_split(train, val):
    paths = lambda data: {(data.root / row[0]).resolve() for row in data.rows}
    vehicles = lambda data: {row[2] for row in data.rows}
    if paths(train) & paths(val) or vehicles(train) & vehicles(val):
        raise ValueError("Train and validation manifests share images or vehicle IDs")


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    random.seed(42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_data = RingDataset(args.train_csv, args.root, True)
    val_data = RingDataset(args.val_csv, args.root, False)
    verify_split(train_data, val_data)
    if len(train_data) < args.batch:
        raise ValueError("Training rows must be at least the batch size")
    train_loader = DataLoader(train_data, args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, args.batch)
    student = SpoutDepthStudentDual().to(device)
    gate = SoftGate().to(device)
    reference = load_student(args.kd_checkpoint, device) if args.kd_checkpoint else None
    if reference is not None:
        for module in reference:
            module.requires_grad_(False)
    params = list(student.parameters()) + list(gate.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    total = len(train_loader) * args.epochs
    warmup = min(total, len(train_loader) * 3)

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "training.csv").open("w", encoding="utf-8", newline="") as handle:
        log = csv.writer(handle)
        log.writerow(["epoch", "training_loss", "validation_mae"])
        for epoch in range(1, args.epochs + 1):
            student.train()
            gate.train()
            losses = []
            for images, target in train_loader:
                images = images.to(device)
                target = target.float().to(device)
                with torch.no_grad():
                    frozen = roi_outputs(*reference, images) if reference else None
                loss = objective(roi_outputs(student, gate, images), target, frozen)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss is not finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

            student.eval()
            gate.eval()
            error = 0.0
            with torch.inference_mode():
                for images, target in val_loader:
                    fused = roi_outputs(student, gate, images.to(device))[-1]
                    error += (target.to(device) - fused).abs().sum().item()
            mae = error / len(val_data)
            log.writerow([epoch, sum(losses) / len(losses), mae])
            handle.flush()
            print(f"Epoch {epoch}/{args.epochs} validation MAE {mae:.6f}", flush=True)
            if mae < best:
                best = mae
                torch.save({"student_state": student.state_dict(), "gate_state": gate.state_dict(),
                            "epoch": epoch, "validation_mae": mae}, args.out / "best.pth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-csv", "val-csv", "root", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--kd-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.threads) < 1:
        parser.error("epochs, batch and threads must be positive")
    train(args)


if __name__ == "__main__":
    main()

