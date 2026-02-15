# -*- coding: utf-8 -*-
"""
学生网络蒸馏训练（Dual-Head: NIG Evidential + Deterministic + SoftGate Fusion）

Student:
- MobileNetV3 backbone（共享） + 两个回归头 + 软门控
- NIG 头：μ_nig, v, α, β
- 确定性头：μ_det
- SoftGate：输入 [μ_det, μ_nig, σ_tot, |μ_det-μ_nig|] → 输出 w∈[0,1]（NIG 权重）

Teacher（蒸馏用）:
- 采用与 Student 相同结构（SpoutDepthStudentDual + SoftGate）
- 从 TEACHER_CKPT 加载，参数冻结，仅提供软标签

训练损失：
    L_total = L_nig + L_reg + L_det + L_cons
              + λ_fuse * |y - μ_fused|
              + λ_gate * BCE(w_logits, 1{err_nig<err_det})
              + λ_prior * CE_highZ(w_logits, 1{z>0.6})
              + λ_rank * rank(σ_tot)
              + λ_kd_mu * MSE( μ_fused, μ_fused_T )
              + λ_kd_sigma * MSE( σ_tot, σ_T )
              + λ_kd_w * MSE( wN, w_T )
"""

import os, math, random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd

# ---------------- 用户配置 ----------------
CSV_PATH   = r"F:\YOLO\DAV2\POINTCLOUD_RAINBOW_Zx200\SPOUT_3D_Zx200_SURFACE_CENTER_WITH_INV.csv"
RGB_DIR    = r"F:\YOLO\DAV2\oil_mouth_crops"  # 434×434 的壶口 ROI

LOG_DIR    = r"F:\YOLO\DAV2\students_5\pout_depth_student_nig_dual_logs"
SAVE_PATH  = r"F:\YOLO\DAV2\students_5\mobilenetv3_spout_depth_student_nig_dual.pth"
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 16
EPOCHS      = 150
BASE_LR     = 1e-4
WARMUP_EPS  = 3
INPUT_SIZE  = 434           # 434×434
SEED        = 42
NUM_WORKERS = 0             # Windows 建议 0

VIS_IDX     = 0

# --------- 蒸馏相关配置 ---------
USE_DISTILL      = True
TEACHER_CKPT     = r"F:\YOLO\DAV2\students_5\mobilenetv3_spout_depth_student_nig_dual_best.pth"  # 自己确认
LAMBDA_KD_MU     = 0.5    # μ_fused 的蒸馏权重（建议 0.2~1.0 之间调）
LAMBDA_KD_SIGMA  = 0.1    # σ_tot 的蒸馏权重（如不想蒸馏不确定性可设为 0）
LAMBDA_KD_W      = 0.1    # 门控权重 w 的蒸馏（不想蒸馏 gate 可设为 0）

# 不确定性排序约束
USE_RANK_LOSS = True
LAMBDA_RANK   = 0.10       # 稍降，避免高 Z 欠拟合
RANK_MARGIN   = 0.05

# 证据正则
LAMBDA_EVIDENCE = 5e-3

# 确定性头（中段收紧）超参
DET_L1_BASE = 0.05
DET_L1_MID  = 0.5
MID_LO, MID_HI = 0.3, 0.6

# 一致性损失权重
LAMBDA_CONS = 0.10

# 软门控/融合相关权重
LAMBDA_FUSED = 0.10       # 融合 μ 的小权重监督
LAMBDA_GATE  = 0.20       # 门控 BCE（谁更准，选谁）
LAMBDA_PRIOR = 0.02       # 高 Z 段轻微鼓励选 NIG；设 0 可关闭

EPS = 1e-6

# ---------------- 随机种子 ----------------
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)

# ---------------- Dataset ----------------
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class SpoutDepthDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        valid_rows = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, row["image"])
            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                print(f"[WARN] 图像不存在，跳过: {img_path}")
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError("数据集为空，请检查 CSV_PATH 和 RGB_DIR。")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img) if self.transform else transforms.ToTensor()(img)
        z_cam = float(row["Z_rel_cam"])
        z_cam = max(0.0, min(1.0, z_cam))
        target = torch.tensor([z_cam], dtype=torch.float32)
        return img_t, target, img_name

dataset = SpoutDepthDataset(CSV_PATH, RGB_DIR, transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
print(f"Dataset size: {len(dataset)} samples")

# ---------------- Student：共享骨干 + 两头 ----------------
class SpoutDepthStudentDual(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feats = backbone.classifier[0].in_features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
        )
        self.head_nig = nn.Linear(256, 4)  # μ_nig_raw, v_raw, α_raw, β_raw
        self.head_det = nn.Linear(256, 1)  # μ_det_raw

    def forward(self, x):
        x = self.pool(self.features(x))            # [B,C,1,1]
        feat = self.fc(x)                          # [B,256]
        nig_o = self.head_nig(feat)                # [B,4]
        mu_nig_raw, v_raw, a_raw, b_raw = nig_o[:,0], nig_o[:,1], nig_o[:,2], nig_o[:,3]
        mu_nig = torch.sigmoid(mu_nig_raw)
        v      = F.softplus(v_raw) + 1e-3
        alpha  = F.softplus(a_raw) + 1.0 + 1e-3
        beta   = F.softplus(b_raw) + 1e-3
        mu_det_raw = self.head_det(feat).squeeze(1)   # [B]
        mu_det     = torch.sigmoid(mu_det_raw)
        return mu_nig, v, alpha, beta, mu_det

# 软门控（鲁棒版：统一 reshape 为一维）
class SoftGate(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # 输出 logit
        )
    def forward(self, mu_det, mu_nig, sigma_tot):
        # 确保一维 [B]
        mu_det   = mu_det.reshape(-1)
        mu_nig   = mu_nig.reshape(-1)
        sigma_tot= sigma_tot.reshape(-1)
        feats = torch.stack(
            [mu_det, mu_nig, sigma_tot, torch.abs(mu_det - mu_nig)],
            dim=-1
        )  # [B,4]
        logit = self.mlp(feats)        # [B,1]
        w = torch.sigmoid(logit)       # NIG 权重
        return logit, w

student = SpoutDepthStudentDual().to(DEVICE)
gate    = SoftGate().to(DEVICE)

# ---------------- Teacher（蒸馏用，同结构，冻结） ----------------
teacher_student = None
teacher_gate    = None

if USE_DISTILL:
    if not os.path.isfile(TEACHER_CKPT):
        raise FileNotFoundError(f"TEACHER_CKPT 不存在: {TEACHER_CKPT}")
    print(f"[Distill] Loading teacher from {TEACHER_CKPT}")
    state = torch.load(TEACHER_CKPT, map_location=DEVICE)

    teacher_student = SpoutDepthStudentDual().to(DEVICE)
    teacher_gate    = SoftGate().to(DEVICE)

    # 支持两种 ckpt：包含 student_state/gate_state 或直接 state_dict
    if isinstance(state, dict) and "student_state" in state:
        teacher_student.load_state_dict(state["student_state"])
        if "gate_state" in state:
            teacher_gate.load_state_dict(state["gate_state"])
        else:
            print("[Distill][WARN] gate_state 不在 ckpt 中，使用随机初始化 gate。")
    else:
        teacher_student.load_state_dict(state, strict=False)
        print("[Distill][WARN] ckpt 没有 student_state，尝试直接 load_state_dict。")

    teacher_student.eval().requires_grad_(False)
    teacher_gate.eval().requires_grad_(False)
    print("[Distill] Teacher loaded & frozen.")

# ---------------- NIG 工具 ----------------
def nig_nll(y, mu, v, alpha, beta):
    two_beta = 2.0 * beta
    res2 = (y - mu) ** 2
    term1 = 0.5 * torch.log(math.pi / (v + EPS))
    term2 = - alpha * torch.log(two_beta + EPS)
    term3 = (alpha + 0.5) * torch.log(v * res2 + two_beta + EPS)
    term4 = torch.lgamma(alpha + EPS) - torch.lgamma(alpha + 0.5 + EPS)
    return term1 + term2 + term3 + term4

def nig_evidence_regularizer(y, mu, v, alpha, coeff=5e-3):
    return coeff * (torch.abs(y - mu) * (2.0 * v + alpha)).mean()

def nig_total_sigma(v, alpha, beta):
    denom = (alpha - 1.0).clamp(min=1.001)
    var_alea = beta / denom
    var_epi  = beta / (v * denom + EPS)
    var_tot  = (var_alea + var_epi).clamp(min=EPS)
    return torch.sqrt(var_tot)

# ---------------- Optimizer & Scheduler ----------------
optimizer = torch.optim.AdamW(
    list(student.parameters()) + list(gate.parameters()),
    lr=BASE_LR, weight_decay=1e-4
)

total_iters = EPOCHS * len(loader)
warmup_iters = max(1, WARMUP_EPS * len(loader))
def lr_lambda(it):
    if it < warmup_iters: return float(it) / float(max(1, warmup_iters))
    t = (it - warmup_iters) / float(max(1, total_iters - warmup_iters))
    return 0.5 * (1.0 + math.cos(math.pi * t))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith("cuda"))

# ---------------- 日志 ----------------
log_csv = os.path.join(LOG_DIR, "train_spout_depth_student_nig_dual_kd.csv")
with open(log_csv, "w", encoding="utf-8") as f:
    f.write(
        "epoch,iter,lr,"
        "loss_total,loss_nig,loss_reg,loss_det,loss_cons,"
        "loss_fuse,loss_gate,loss_prior,loss_rank,"
        "loss_kd_mu,loss_kd_sigma,loss_kd_w,"
        "mean_wN,mean_wT\n"
    )

def save_vis(epoch, sample_img_t, z_gt, mu_nig_pred, mu_det_pred, mu_fused, sigma_pred,
             save_dir=LOG_DIR, name="sample"):
    img = sample_img_t.cpu().numpy().transpose(1,2,0)
    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    img_vis = np.clip(img * std + mean, 0, 1)
    plt.figure(figsize=(4.2,4.2))
    plt.imshow(img_vis); plt.axis("off")
    plt.title(f"Epoch {epoch}\nGT={z_gt:.3f}\nμ_nig={mu_nig_pred:.3f}\nμ_det={mu_det_pred:.3f}\nμ_fuse={mu_fused:.3f}\nσ={sigma_pred:.3f}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name}_epoch_{epoch:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ---------------- Training Loop ----------------
print("Start training spout depth student (Dual-Head + SoftGate + KD) on", DEVICE)
best_loss = float("inf")
loss_history = []

bce_logits = nn.BCEWithLogitsLoss()

for epoch in range(1, EPOCHS+1):
    student.train(); gate.train()
    epoch_loss = 0.0; samples = 0

    for it, (imgs, targets, names) in enumerate(loader):
        imgs    = imgs.to(DEVICE)               # [B,3,H,W]
        targets = targets.to(DEVICE).view(-1)   # [B]
        bs = imgs.size(0)

        with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
            # ---------- Student 前向 ----------
            mu_nig, v, alpha, beta, mu_det = student(imgs)   # (B,)
            sigma_tot = nig_total_sigma(v, alpha, beta)      # (B,)

            # 门控 & 融合
            w_logit, wN = gate(mu_det, mu_nig, sigma_tot)    # wN: (B,1)
            wN = wN.squeeze(1)                               # (B,)
            mu_fused = wN * mu_nig + (1.0 - wN) * mu_det     # (B,)

            # ---------- Teacher 前向（蒸馏） ----------
            if USE_DISTILL:
                with torch.no_grad():
                    mu_nig_T, v_T, a_T, b_T, mu_det_T = teacher_student(imgs)
                    sigma_T = nig_total_sigma(v_T, a_T, b_T)
                    w_logit_T, wT = teacher_gate(mu_det_T, mu_nig_T, sigma_T)
                    wT = wT.squeeze(1)
                    mu_fused_T = wT * mu_nig_T + (1.0 - wT) * mu_det_T

            # ---------- 原始任务损失 ----------
            # 1) NIG 主损失
            loss_nig = nig_nll(targets, mu_nig, v, alpha, beta).mean()
            # 2) 证据正则
            loss_reg = nig_evidence_regularizer(targets, mu_nig, v, alpha, coeff=LAMBDA_EVIDENCE)

            # 3) 确定性头 L1（中段加权）
            err_det = torch.abs(targets - mu_det)
            mid_mask = (targets >= MID_LO) & (targets < MID_HI)
            lam_det = DET_L1_BASE + DET_L1_MID * mid_mask.float()
            loss_det = (lam_det * err_det).mean()

            # 4) 一致性（stop-grad 到 NIG）
            loss_cons = LAMBDA_CONS * torch.abs(mu_det - mu_nig.detach()).mean()

            # 5) 融合 μ 的小权重监督
            loss_fuse = LAMBDA_FUSED * torch.abs(targets - mu_fused).mean()

            # 6) 门控自监督（谁更准，选谁）
            err_nig = torch.abs(targets - mu_nig)
            better_nig = (err_nig < err_det).float().unsqueeze(1)  # [B,1]
            loss_gate = LAMBDA_GATE * bce_logits(w_logit, better_nig)

            # 7) 高 Z 段轻微先验
            highZ = (targets > 0.6).float().unsqueeze(1)
            loss_prior = LAMBDA_PRIOR * bce_logits(w_logit, highZ)

            # 8) 排名约束（总 σ 对应残差大小）
            if USE_RANK_LOSS:
                with torch.no_grad():
                    res = err_nig
                    med = res.median()
                    high_mask = res >= med
                    low_mask  = res <  med
                if high_mask.any() and low_mask.any():
                    s_high = sigma_tot[high_mask].mean()
                    s_low  = sigma_tot[low_mask].mean()
                    loss_rank = F.relu(RANK_MARGIN - (s_high - s_low))
                else:
                    loss_rank = sigma_tot.new_tensor(0.0)
            else:
                loss_rank = torch.tensor(0.0, device=DEVICE)

            # ---------- 蒸馏损失 ----------
            if USE_DISTILL:
                loss_kd_mu    = LAMBDA_KD_MU    * F.mse_loss(mu_fused,   mu_fused_T.detach())
                loss_kd_sigma = LAMBDA_KD_SIGMA * F.mse_loss(sigma_tot,  sigma_T.detach())
                loss_kd_w     = LAMBDA_KD_W     * F.mse_loss(wN,         wT.detach())
            else:
                loss_kd_mu = loss_kd_sigma = loss_kd_w = torch.tensor(0.0, device=DEVICE)

            # 总损失
            loss_total = (loss_nig + loss_reg + loss_det + loss_cons +
                          loss_fuse + loss_gate + loss_prior +
                          LAMBDA_RANK * loss_rank +
                          loss_kd_mu + loss_kd_sigma + loss_kd_w)

        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters())+list(gate.parameters()), max_norm=1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()

        epoch_loss += loss_total.item() * bs; samples += bs

        if (it+1) % 20 == 0 or (it+1) == len(loader):
            cur_lr = optimizer.param_groups[0]['lr']
            mean_wN = wN.mean().item()
            mean_wT = wT.mean().item() if USE_DISTILL else 0.0
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{it+1},{cur_lr:.6e},"
                    f"{loss_total.item():.6f},{loss_nig.item():.6f},{loss_reg.item():.6f},{loss_det.item():.6f},"
                    f"{loss_cons.item():.6f},{loss_fuse.item():.6f},{loss_gate.item():.6f},"
                    f"{loss_prior.item():.6f},{loss_rank.item():.6f},"
                    f"{loss_kd_mu.item():.6f},{loss_kd_sigma.item():.6f},{loss_kd_w.item():.6f},"
                    f"{mean_wN:.4f},{mean_wT:.4f}\n"
                )
            print(
                f"Epoch {epoch} {it+1}/{len(loader)} lr {cur_lr:.2e}  "
                f"tot {loss_total.item():.4f} | nig {loss_nig.item():.4f} reg {loss_reg.item():.4f} "
                f"det {loss_det.item():.4f} cons {loss_cons.item():.4f} "
                f"fuse {loss_fuse.item():.4f} gate {loss_gate.item():.4f} prior {loss_prior.item():.4f} "
                f"rank {loss_rank.item():.4f}  "
                f"kd_mu {loss_kd_mu.item():.4f} kd_sig {loss_kd_sigma.item():.4f} kd_w {loss_kd_w.item():.4f}  "
                f"mean_wN {mean_wN:.3f} mean_wT {mean_wT:.3f}"
            )

    avg_epoch_loss = epoch_loss / samples
    loss_history.append(avg_epoch_loss)
    print(f"Epoch {epoch}/{EPOCHS} finished. mean_loss_total {avg_epoch_loss:.6f}")

    # 可视化一张样本（鲁棒 reshape，避免 0-D 引发 stack 错误）
    student.eval(); gate.eval()
    with torch.no_grad():
        sample_img, sample_target, sample_name = dataset[VIS_IDX]
        sample_img_t = sample_img.unsqueeze(0).to(DEVICE)

        mu_nig_p, v_p, a_p, b_p, mu_det_p = student(sample_img_t)
        # 统一成一维 [1]
        mu_nig_1 = mu_nig_p.reshape(-1)
        mu_det_1 = mu_det_p.reshape(-1)
        v_1      = v_p.reshape(-1)
        a_1      = a_p.reshape(-1)
        b_1      = b_p.reshape(-1)

        sigma_1  = nig_total_sigma(v_1, a_1, b_1)                 # [1]
        w_logit1, wN_1 = gate(mu_det_1, mu_nig_1, sigma_1)        # [1,1]
        mu_fused_1 = (wN_1.reshape(-1) * mu_nig_1 + (1 - wN_1.reshape(-1)) * mu_det_1)[0].item()

        save_vis(
            epoch, sample_img, float(sample_target.item()),
            float(mu_nig_1[0].item()),
            float(mu_det_1[0].item()),
            float(mu_fused_1),
            float(sigma_1[0].item()),
            save_dir=LOG_DIR, name=Path(sample_name).stem
        )

    # 选 best（按总 loss）
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save({
            "epoch": epoch,
            "student_state": student.state_dict(),
            "gate_state": gate.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_epoch_loss
        }, SAVE_PATH.replace(".pth", "_best.pth"))
        print(f"New best {best_loss:.6f} -> saved best checkpoint.")

# final save
torch.save({
    "student_state": student.state_dict(),
    "gate_state": gate.state_dict(),
    "loss_history": loss_history,
    "config": dict(
        INPUT_SIZE=INPUT_SIZE,BATCH_SIZE=BATCH_SIZE,BASE_LR=BASE_LR,EPOCHS=EPOCHS,
        LAMBDA_RANK=LAMBDA_RANK,RANK_MARGIN=RANK_MARGIN,LAMBDA_EVIDENCE=LAMBDA_EVIDENCE,
        DET_L1_BASE=DET_L1_BASE,DET_L1_MID=DET_L1_MID,MID_LO=MID_LO,MID_HI=MID_HI,
        LAMBDA_CONS=LAMBDA_CONS,LAMBDA_FUSED=LAMBDA_FUSED,LAMBDA_GATE=LAMBDA_GATE,LAMBDA_PRIOR=LAMBDA_PRIOR,
        USE_DISTILL=USE_DISTILL,
        LAMBDA_KD_MU=LAMBDA_KD_MU,LAMBDA_KD_SIGMA=LAMBDA_KD_SIGMA,LAMBDA_KD_W=LAMBDA_KD_W,
        TEACHER_CKPT=TEACHER_CKPT
    )
}, SAVE_PATH)
print("Training finished. Saved:", SAVE_PATH)
