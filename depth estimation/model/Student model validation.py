# -*- coding: utf-8 -*-
import os, math, random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ---------------- 路径配置（按需改） ----------------
CSV_PATH   = r"F:\YOLO\DAV2\POINTCLOUD_RAINBOW_Zx200\SPOUT_3D_Zx200_SURFACE_CENTER_WITH_INV.csv"
ROI_DIR    = r"F:\YOLO\DAV2\oil_mouth_crops"          # 训练用的 434x434 ROI
ORIG_DIR   = r"F:\YOLO\DAV2\image"                     # 原始大图目录
CKPT_PATH  = r"F:\YOLO\DAV2\students_5\mobilenetv3_spout_depth_student_nig_dual_best.pth"
OUT_VIS_DIR= r"F:\YOLO\DAV2\students_5\overlay_vis"

os.makedirs(OUT_VIS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 434
EPS = 1e-6
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- 预处理（与训练一致） ----------------
tfm = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ---------------- 学生模型（与训练同构） ----------------
class SpoutDepthStudentDual(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feats = backbone.classifier[0].in_features
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(in_feats, 256), nn.ReLU(inplace=True))
        self.head_nig = nn.Linear(256, 4)  # μ_nig_raw, v_raw, α_raw, β_raw
        self.head_det = nn.Linear(256, 1)  # μ_det_raw

    def forward(self, x):
        x = self.pool(self.features(x))
        feat = self.fc(x)
        nig_o = self.head_nig(feat)
        mu_nig_raw, v_raw, a_raw, b_raw = nig_o[:,0], nig_o[:,1], nig_o[:,2], nig_o[:,3]
        mu_nig = torch.sigmoid(mu_nig_raw)
        v      = F.softplus(v_raw) + 1e-3
        alpha  = F.softplus(a_raw) + 1.0 + 1e-3
        beta   = F.softplus(b_raw) + 1e-3
        mu_det = torch.sigmoid(self.head_det(feat).squeeze(1))
        return mu_nig, v, alpha, beta, mu_det

class SoftGate(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
    def forward(self, mu_det, mu_nig, sigma_tot):
        mu_det = mu_det.reshape(-1); mu_nig = mu_nig.reshape(-1); sigma_tot = sigma_tot.reshape(-1)
        feats = torch.stack([mu_det, mu_nig, sigma_tot, torch.abs(mu_det - mu_nig)], dim=-1)  # [B,4]
        logit = self.mlp(feats); w = torch.sigmoid(logit)
        return logit, w  # w 是 NIG 的权重

@torch.no_grad()
def nig_sigma_total(v, alpha, beta) -> torch.Tensor:
    denom = (alpha - 1.0).clamp(min=1.001)
    var_alea = beta / denom
    var_epi  = beta / (v * denom + EPS)
    var_tot  = (var_alea + var_epi).clamp(min=EPS)
    return torch.sqrt(var_tot)

# ---------------- 加载权重 ----------------
student = SpoutDepthStudentDual().to(DEVICE)
gate = SoftGate().to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "student_state" in ckpt:
    student.load_state_dict(ckpt["student_state"], strict=True)
    if "gate_state" in ckpt:
        gate.load_state_dict(ckpt["gate_state"], strict=False)
        HAS_GATE = True
    else:
        HAS_GATE = False
else:
    student.load_state_dict(ckpt, strict=True)
    HAS_GATE = False
student.eval(); gate.eval()

print(f"[INFO] Loaded weights. HAS_GATE={HAS_GATE}")

# ---------------- 绘图小工具 ----------------
def _try_font(size=18):
    # Windows 常见字体，任选可用的
    for f in ["arial.ttf", "C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\simhei.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT_SM = _try_font(18)
FONT_LG = _try_font(22)

def draw_panel(orig: Image.Image, roi_thumb: Image.Image,
               gt: float, mu_nig: float, mu_det: float, mu_fused: float, sigma: float,
               name: str) -> Image.Image:
    W, H = orig.size
    vis = orig.convert("RGB").copy()
    draw = ImageDraw.Draw(vis, "RGBA")

    # 左上贴 ROI 缩略图
    thumb_w = min(200, W//4)
    r = roi_thumb.resize((thumb_w, thumb_w))
    draw.rectangle([10, 10, 10+thumb_w+4, 10+thumb_w+4], fill=(255,255,255,60), outline=(0,0,0,120), width=2)
    vis.paste(r, (12, 12))

    # 右下信息面板
    panel_w, panel_h = min(420, W-40), 180
    px1, py1 = W - panel_w - 20, H - panel_h - 20
    px2, py2 = W - 20, H - 20
    draw.rectangle([px1, py1, px2, py2], fill=(0,0,0,120), outline=(255,255,255,160), width=2)

    lines = [
        f"{name}",
        f"GT={gt:.3f}  μ_nig={mu_nig:.3f}  μ_det={mu_det:.3f}",
        f"μ_fused={mu_fused:.3f}   σ_total={sigma:.3f}   |err|={abs(mu_fused-gt):.3f}",
    ]
    ty = py1 + 10
    for i, t in enumerate(lines):
        draw.text((px1+12, ty), t, fill=(255,255,255,230), font=FONT_LG if i==0 else FONT_SM)
        ty += 30 if i==0 else 26

    # 0~1 进度条 + 标记
    bar_x1, bar_x2 = px1+12, px2-12
    bar_y1, bar_y2 = py2-36, py2-20
    draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=(255,255,255,60), outline=(255,255,255,180), width=2)
    # 填充到 μ_fused
    fx = bar_x1 + int((bar_x2-bar_x1) * np.clip(mu_fused,0,1))
    draw.rectangle([bar_x1, bar_y1, fx, bar_y2], fill=(80,200,120,180))
    # GT 与 μ_fused 标记线
    def mark_at(val, color, lbl):
        x = bar_x1 + int((bar_x2-bar_x1) * np.clip(val,0,1))
        draw.line([(x, bar_y1-6), (x, bar_y2+6)], fill=color, width=2)
        draw.text((x-10, bar_y1-22), lbl, fill=color, font=FONT_SM)
    mark_at(gt, (255,180,0,255), "GT")
    mark_at(mu_fused, (80,200,120,255), "μf")

    return vis

# ---------------- 主流程 ----------------
df = pd.read_csv(CSV_PATH)
n_total = 0
for _, row in df.iterrows():
    name = str(row["image"])
    gt = float(max(0.0, min(1.0, float(row["Z_rel_cam"]))))

    roi_path = os.path.join(ROI_DIR, name)
    orig_path= os.path.join(ORIG_DIR, name)

    if not os.path.exists(roi_path):
        print(f"[WARN] ROI 缺失，跳过: {roi_path}"); continue
    if not os.path.exists(orig_path):
        # 如果原图目录文件名不同，可自行在此改映射规则
        print(f"[WARN] 原图缺失，跳过: {orig_path}"); continue

    # 模型推理
    img_roi = Image.open(roi_path).convert("RGB")
    x = tfm(img_roi).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu_nig, v, alpha, beta, mu_det = student(x)
        sigma_tot = nig_sigma_total(v, alpha, beta)  # (B,)
        if 'HAS_GATE' in globals() and HAS_GATE:
            _, wN = gate(mu_det, mu_nig, sigma_tot)  # (B,1)
            wN = wN.squeeze(1)
            mu_fused = (wN * mu_nig + (1 - wN) * mu_det)
        else:
            mu_fused = mu_nig  # 没有 gate 就用 NIG

    mu_nig_f  = float(mu_nig.squeeze().cpu().item())
    mu_det_f  = float(mu_det.squeeze().cpu().item())
    mu_fused_f= float(mu_fused.squeeze().cpu().item())
    sigma_f   = float(sigma_tot.squeeze().cpu().item())

    # 读原图并叠加可视化
    img_orig = Image.open(orig_path).convert("RGB")
    vis = draw_panel(img_orig, img_roi, gt, mu_nig_f, mu_det_f, mu_fused_f, sigma_f, name)

    out_path = os.path.join(OUT_VIS_DIR, f"vis_{Path(name).stem}.jpg")
    vis.save(out_path, quality=95)
    n_total += 1

print(f"[DONE] 已生成 {n_total} 张叠加可视化到：{OUT_VIS_DIR}")
