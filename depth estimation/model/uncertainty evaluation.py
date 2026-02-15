# -*- coding: utf-8 -*-
"""
评估（NIG Evidential 学生）：
- 加载 mobilenetv3_spout_depth_student_nig_best.pth
- 在 CSV + ROI 图像上逐张推理
- 使用 μ 作为预测深度，统计误差（MAE / RMSE / MaxErr）
- 额外导出每个样本的 σ_total / σ_alea / σ_epi 用于分析
- 导出对比结果 CSV + 可视化散点图
- 按 Z_rel_cam 区间统计 MAE / RMSE 变化曲线
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

# ---------------- 路径配置 ----------------
CSV_PATH   = r"F:\YOLO\DAV2\POINTCLOUD_RAINBOW_Zx200\SPOUT_3D_Zx200_SURFACE_CENTER_WITH_INV.csv"
RGB_DIR    = r"F:\YOLO\DAV2\oil_mouth_crops"  # 434×434 ROI

# ✅ 改成你 NIG 学生 best 权重路径
CKPT_PATH  = r"F:\YOLO\DAV2\students_3\mobilenetv3_spout_depth_student_nig_best.pth"

OUT_DIR    = r"F:\YOLO\DAV2\students_3\eval_results_nig"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV       = os.path.join(OUT_DIR, "spout_depth_student_nig_eval.csv")
OUT_SCATTER   = os.path.join(OUT_DIR, "gt_vs_pred_scatter_nig.png")
OUT_CURVE     = os.path.join(OUT_DIR, "mae_rmse_vs_zbin_nig.png")
OUT_SIGMA_ERR = os.path.join(OUT_DIR, "abs_err_vs_sigma_nig.png")

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

        valid = []
        for _, row in self.df.iterrows():
            p = os.path.join(self.image_dir, row["image"])
            if os.path.exists(p):
                valid.append(row)
            else:
                print(f"[WARN] 评估时找不到图像，跳过: {p}")
        self.df = pd.DataFrame(valid).reset_index(drop=True)

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

# ---------------- 学生网络结构（NIG：输出 μ, v, α, β） ----------------
class SpoutDepthStudentNIG(nn.Module):
    """
    输出：
      μ ∈ (0,1)      Sigmoid
      v > 0          Softplus + ε
      α > 1          Softplus + 1 + ε
      β > 0          Softplus + ε
    """
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)  # 与训练保持一致
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feats = backbone.classifier[0].in_features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)   # -> mu_raw, v_raw, alpha_raw, beta_raw
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)            # (B,C,1,1)
        o = self.head(x)            # (B,4)
        mu_raw, v_raw, a_raw, b_raw = o[:,0], o[:,1], o[:,2], o[:,3]
        mu    = torch.sigmoid(mu_raw)
        v     = F.softplus(v_raw) + 1e-3
        alpha = F.softplus(a_raw) + 1.0 + 1e-3
        beta  = F.softplus(b_raw) + 1e-3
        return mu, v, alpha, beta

@torch.no_grad()
def nig_sigmas(v, alpha, beta, eps=1e-6):
    """
    返回 (sigma_total, sigma_alea, sigma_epi)
      Var_total = beta/(alpha-1) + beta/(v*(alpha-1))
    """
    denom = (alpha - 1.0).clamp(min=1.001)
    var_alea = beta / denom
    var_epi  = beta / (v * denom + eps)
    var_tot  = (var_alea + var_epi).clamp(min=eps)
    return torch.sqrt(var_tot), torch.sqrt(var_alea.clamp(min=eps)), torch.sqrt(var_epi.clamp(min=eps))

student = SpoutDepthStudentNIG().to(DEVICE)

# ---------------- 加载最佳权重 ----------------
print(f"加载权重: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
if "student_state" in ckpt:
    student.load_state_dict(ckpt["student_state"])
    print(f"  -> 来自训练 checkpoint，epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss','?')}")
else:
    student.load_state_dict(ckpt)
    print("  -> 直接加载 state_dict")

student.eval()

# ---------------- 推理 + 统计误差 ----------------
all_records = []

with torch.no_grad():
    for imgs, targets, names in eval_loader:
        imgs    = imgs.to(DEVICE)
        targets = targets.to(DEVICE)   # [B,1]

        mu, v, alpha, beta = student(imgs)    # (B,), (B,), (B,), (B,)
        mu = mu.view(-1); v = v.view(-1); alpha = alpha.view(-1); beta = beta.view(-1)

        sigma_tot, sigma_alea, sigma_epi = nig_sigmas(v, alpha, beta, eps=EPS)

        preds_np      = mu.cpu().numpy()
        sigma_np      = sigma_tot.cpu().numpy()
        sigma_alea_np = sigma_alea.cpu().numpy()
        sigma_epi_np  = sigma_epi.cpu().numpy()
        targets_np    = targets.view(-1).cpu().numpy()

        for name, gt, pred, s, sa, se in zip(names, targets_np, preds_np, sigma_np, sigma_alea_np, sigma_epi_np):
            abs_err = abs(pred - gt)
            all_records.append({
                "image": name,
                "GT_Z_rel_cam": float(gt),
                "Pred_Z_rel_cam": float(pred),
                "Pred_sigma_total": float(s),
                "Pred_sigma_alea":  float(sa),
                "Pred_sigma_epi":   float(se),
                "AbsError": float(abs_err)
            })

# 转成 DataFrame
df_eval = pd.DataFrame(all_records)

# 计算整体指标（基于 μ）
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
plt.scatter(df_eval["GT_Z_rel_cam"], df_eval["Pred_Z_rel_cam"], s=10, alpha=0.6)
plt.plot([0,1], [0,1], "r--", label="y = x")
plt.xlabel("GT Z_rel_cam")
plt.ylabel("Pred Z_rel_cam (μ)")
plt.title("NIG Student: GT vs Pred")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=200)
plt.close()
print(f"GT vs Pred 散点图已保存：{OUT_SCATTER}")

# ---------------- 按 Z 区间统计 MAE / RMSE 变化曲线 ----------------
bins = np.linspace(0.0, 1.0, 11)           # 0,0.1,...,1.0
bin_centers = (bins[:-1] + bins[1:]) / 2.0

mae_list  = []
rmse_list = []
count_list = []

for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i+1]
    mask = (df_eval["GT_Z_rel_cam"] >= lo) & (df_eval["GT_Z_rel_cam"] < hi)
    sub = df_eval[mask]
    if len(sub) == 0:
        mae_list.append(np.nan)
        rmse_list.append(np.nan)
        count_list.append(0)
        continue

    mae_i  = sub["AbsError"].mean()
    rmse_i = math.sqrt(((sub["Pred_Z_rel_cam"] - sub["GT_Z_rel_cam"])**2).mean())
    mae_list.append(mae_i)
    rmse_list.append(rmse_i)
    count_list.append(len(sub))

print("====== 按 GT_Z_rel_cam 分段的 MAE / RMSE（NIG 学生） ======")
for c, lo, hi, m, r in zip(count_list, bins[:-1], bins[1:], mae_list, rmse_list):
    print(f"[{lo:.1f}, {hi:.1f})  N={c:4d}  MAE={m if not np.isnan(m) else float('nan'):.4f}  RMSE={r if not np.isnan(r) else float('nan'):.4f}")
print("=========================================================\n")

# 画 MAE / RMSE 曲线
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, mae_list,  marker="o", label="MAE")
plt.plot(bin_centers, rmse_list, marker="s", label="RMSE")
plt.xlabel("GT Z_rel_cam")
plt.ylabel("deviation")
plt.title("NIG Student: MAE / RMSE vs Z_bin")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_CURVE, dpi=200)
plt.close()
print(f"MAE / RMSE 分段变化曲线已保存：{OUT_CURVE}")

# ---------------- 额外：AbsError vs Sigma_total（验证不确定性有效性） ----------------
plt.figure(figsize=(6,4))
plt.scatter(df_eval["Pred_sigma_total"], df_eval["AbsError"], s=8, alpha=0.5)
plt.xlabel("Predicted Sigma (total)")
plt.ylabel("|Pred_Z - GT_Z|")
plt.title("AbsError vs Sigma_total (NIG Student)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_SIGMA_ERR, dpi=200)
plt.close()
print(f"|误差| vs 预测 Sigma_total 散点图已保存：{OUT_SIGMA_ERR}")

print("评估完成。")
