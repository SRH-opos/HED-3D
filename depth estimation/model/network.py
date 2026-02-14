# -*- coding: utf-8 -*-
"""
评估（Dual-Head: NIG + Deterministic，同构于训练版）：
- 加载 mobilenetv3_spout_depth_student_nig_dual_best.pth （双头；评估仅用 NIG 头：μ,v,α,β）
- 在 CSV + ROI 图像上逐张推理
- 使用 μ 作为预测深度，统计误差（MAE / RMSE / MaxErr）
- 额外导出每个样本的总不确定性 sigma_total（含 aleatoric + epistemic）
- 导出对比结果 CSV + 可视化散点图
- 按 Z_rel_cam 区间统计 MAE / RMSE 变化曲线
- 画 |误差| vs sigma_total 散点，验证不确定性是否有判别力
"""

import os
from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------------- 路径配置 ----------------
CSV_PATH   = r"F:\YOLO\DAV2\POINTCLOUD_RAINBOW_Zx200\SPOUT_3D_Zx200_SURFACE_CENTER_WITH_INV.csv"
RGB_DIR    = r"F:\YOLO\DAV2\oil_mouth_crops"   # 434×434 ROI

# ✅ 指向训练保存的 .pth 权重（注意是 .pth，而不是 .csv）
# 例：r"F:\YOLO\DAV2\students_3\mobilenetv3_spout_depth_student_nig_dual_best.pth"
CKPT_PATH  = r"F:\YOLO\DAV2\students_5\mobilenetv3_spout_depth_student_nig_dual_best.pth"

OUT_DIR    = r"F:\YOLO\DAV2\students_5\eval_results_uncertainty"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV       = os.path.join(OUT_DIR, "spout_depth_student_uncertainty_eval.csv")
OUT_SCATTER   = os.path.join(OUT_DIR, "gt_vs_pred_scatter_uncertainty.png")
OUT_CURVE     = os.path.join(OUT_DIR, "mae_rmse_vs_zbin_uncertainty.png")
OUT_SIGMA_ERR = os.path.join(OUT_DIR, "abs_err_vs_sigma.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 434    # 跟训练时保持一致
BATCH_SIZE = 32
SEED = 42
EPS = 1e-6

# ---------------- 随机种子 ----------------
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)

# ---------------- Dataset（评估用：无数据增强） ----------------
eval_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class SpoutDepthEvalDataset(Dataset):
    """
    与训练的 CSV 对齐：
    - image: 文件名（需在 ROI 目录存在）
    - Z_rel_cam: 作为 GT_Z
    """
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        valid_rows = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, row["image"])
            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                print(f"[WARN] 评估时找不到图像，跳过: {img_path}")
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("评估数据集为空，请检查 CSV_PATH 和 RGB_DIR。")

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

eval_dataset = SpoutDepthEvalDataset(CSV_PATH, RGB_DIR, eval_transform)
eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"评估样本数: {len(eval_dataset)}")

# ---------------- 学生网络结构（与训练时 Dual-Head 同构） ----------------
class SpoutDepthStudentDual(nn.Module):
    """
    共享 MobileNetV3 backbone + 池化 + 256-d 特征
    head_nig: 线性 -> 4D（μ_nig_raw, v_raw, α_raw, β_raw）
    head_det: 线性 -> 1D（μ_det_raw）
    评估时我们只使用 NIG 头的 (μ, v, α, β)
    """
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
        mu_nig_raw, v_raw, a_raw, b_raw = nig_o[:,0:1], nig_o[:,1:2], nig_o[:,2:3], nig_o[:,3:4]
        mu    = torch.sigmoid(mu_nig_raw)
        v     = F.softplus(v_raw) + 1e-3
        alpha = F.softplus(a_raw) + 1.0 + 1e-3
        beta  = F.softplus(b_raw) + 1e-3

        # det 分支不用于评估，可选返回
        mu_det = torch.sigmoid(self.head_det(feat))
        return mu, v, alpha, beta, mu_det

@torch.no_grad()
def nig_sigma_total(v, alpha, beta, eps=1e-6):
    """
    总方差：Var_total = beta/(alpha-1) * (1 + 1/v)
    返回：(sigma_total, sigma_alea, sigma_epi)
    """
    denom = (alpha - 1.0).clamp(min=1.001)
    var_alea = beta / denom
    var_epi  = beta / (v * denom + eps)
    var_tot  = (var_alea + var_epi).clamp(min=eps)
    return torch.sqrt(var_tot), torch.sqrt(var_alea), torch.sqrt(var_epi)

student = SpoutDepthStudentDual().to(DEVICE)

# ---------------- 加载最佳权重 ----------------
print(f"加载权重: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

if "student_state" in ckpt:
    missing, unexpected = student.load_state_dict(ckpt["student_state"], strict=True)
    # strict=True：模型结构与训练时一致，不会吞掉关键层
    print(f"  -> checkpoint epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss','?')}")
else:
    student.load_state_dict(ckpt, strict=True)
    print("  -> 直接加载 state_dict")

student.eval()

# ---------------- 推理 + 统计误差 ----------------
all_records = []

with torch.no_grad():
    for imgs, targets, names in eval_loader:
        imgs    = imgs.to(DEVICE)       # (B,3,H,W)
        targets = targets.to(DEVICE)    # (B,1)

        mu, v, alpha, beta, mu_det = student(imgs)    # (B,1) x5
        sigma_tot, sigma_alea, sigma_epi = nig_sigma_total(v, alpha, beta)

        preds_np     = mu.squeeze(1).cpu().numpy()
        sigma_np     = sigma_tot.squeeze(1).cpu().numpy()
        sigmaA_np    = sigma_alea.squeeze(1).cpu().numpy()
        sigmaE_np    = sigma_epi.squeeze(1).cpu().numpy()
        targets_np   = targets.squeeze(1).cpu().numpy()

        for name, gt, pred, s, sa, se in zip(names, targets_np, preds_np, sigma_np, sigmaA_np, sigmaE_np):
            abs_err = abs(pred - gt)
            all_records.append({
                "image": name,
                "GT_Z_rel_cam": float(gt),
                "Pred_Z_rel_cam": float(pred),
                "Sigma_total": float(s),
                "Sigma_alea":  float(sa),
                "Sigma_epi":   float(se),
                "AbsError": float(abs_err)
            })

df_eval = pd.DataFrame(all_records)

# 计算整体指标
mae  = df_eval["AbsError"].mean()
rmse = math.sqrt(((df_eval["Pred_Z_rel_cam"] - df_eval["GT_Z_rel_cam"])**2).mean())
max_err = df_eval["AbsError"].max()

print("\n====== NIG 学生整体评估结果（基于 μ, Z_rel_cam） ======")
print(f"MAE  (mean |Pred_Z - GT_Z|): {mae:.6f}")
print(f"RMSE:                        {rmse:.6f}")
print(f"Max |err|:                   {max_err:.6f}")
print("====================================================\n")

# 保存逐样本 CSV
df_eval.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"逐样本评估结果已保存：{OUT_CSV}")

# ---------------- 画 GT vs Pred 散点图 ----------------
plt.figure(figsize=(5, 5))
plt.scatter(df_eval["GT_Z_rel_cam"], df_eval["Pred_Z_rel_cam"],
            s=10, alpha=0.6)
plt.plot([0,1], [0,1], "r--", label="y = x")
plt.xlabel("GT Z_rel_cam")
plt.ylabel("Pred Z_rel_cam (mu)")
plt.title("NIG Student (Dual-Head): GT vs Pred")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=200)
plt.close()
print(f"GT vs Pred 散点图已保存：{OUT_SCATTER}")

# ---------------- 按 Z 区间统计 MAE / RMSE 变化曲线 ----------------
bins = np.linspace(0.0, 1.0, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2.0
mae_list, rmse_list, count_list = [], [], []

for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i+1]
    mask = (df_eval["GT_Z_rel_cam"] >= lo) & (df_eval["GT_Z_rel_cam"] < hi)
    sub = df_eval[mask]
    if len(sub) == 0:
        mae_list.append(np.nan); rmse_list.append(np.nan); count_list.append(0)
        continue
    mae_list.append(sub["AbsError"].mean())
    rmse_list.append(math.sqrt(((sub["Pred_Z_rel_cam"] - sub["GT_Z_rel_cam"])**2).mean()))
    count_list.append(len(sub))

print("====== 分段 MAE / RMSE（NIG 学生, Dual-Head） ======")
for c, lo, hi, m, r in zip(count_list, bins[:-1], bins[1:], mae_list, rmse_list):
    print(f"[{lo:.1f}, {hi:.1f})  N={c:4d}  MAE={m if not np.isnan(m) else float('nan'):.4f}  RMSE={r if not np.isnan(r) else float('nan'):.4f}")
print("===================================================\n")

plt.figure(figsize=(6, 4))
plt.plot(bin_centers, mae_list,  marker="o", label="MAE")
plt.plot(bin_centers, rmse_list, marker="s", label="RMSE")
plt.xlabel("GT Z_rel_cam"); plt.ylabel("deviation")
plt.title("NIG Student (Dual-Head): MAE / RMSE vs Z_bin")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(); plt.tight_layout()
plt.savefig(OUT_CURVE, dpi=200)
plt.close()
print(f"MAE / RMSE 分段变化曲线已保存：{OUT_CURVE}")

# ---------------- 额外：AbsError vs Sigma_total ----------------
plt.figure(figsize=(6,4))
plt.scatter(df_eval["Sigma_total"], df_eval["AbsError"], s=8, alpha=0.5)
plt.xlabel("Predicted Sigma_total"); plt.ylabel("|Pred_Z - GT_Z|")
plt.title("AbsError vs Sigma_total (NIG Student, Dual-Head)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_SIGMA_ERR, dpi=200)
plt.close()
print(f"|误差| vs 预测 Sigma_total 散点图已保存：{OUT_SIGMA_ERR}")
print("评估完成。")
