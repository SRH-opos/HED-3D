# -*- coding: utf-8 -*-
"""
【壶口真实点云 · Z 轴向下为正 + 全局线性缩放 ×200】
- 相机在上方：深度 0~1，0 代表本图中最浅，1 代表本图中最深
- 可视化：使用原始相对深度 Z_rel_vis（例如 0.927），并乘以 200 显示立体效果
- 数值输出（CSV）：额外给出 Z_rel_cam = 1 - Z_rel_vis（例如 0.073），供后续相机相对距离标定使用
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings, logging

# ====== 新增：刻度控制工具 ======
from matplotlib.ticker import MaxNLocator

# ====================== 中文显示 ======================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ====================== 路径设置 ======================
REPO_PATH   = r"F:\Depth-Anything-V2-main\Depth-Anything-V2-main"
MODEL_PATH  = os.path.join(REPO_PATH, "depth_anything_v2_vitl.pth")
ELLIPSE_CSV = r"F:\YOLO\DAV2\FOUR\ellipses_third_stage.csv"
IMAGE_DIR   = r"F:\YOLO\DAV2\images"
OUTPUT_DIR  = r"F:\YOLO\DAV2\POINTCLOUD_RAINBOW_Z"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Z 轴缩放系数（向下为正）
Z_SCALE = 200.0

# ====================== 加载 Depth Anything V2 ======================
sys.path.insert(0, REPO_PATH)
from depth_anything_v2.dpt import DepthAnythingV2

warnings.filterwarnings("ignore")
logging.getLogger("depth_anything_v2").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()


def get_depth(img):
    """
    获取整幅图的归一化深度图 depth ∈ [0,1]（相对深度）
    每张图 min-max 归一化：
      0 = 本图中预测最浅处（最靠近相机）
      1 = 本图中预测最深处（最远离相机）
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        d = model.infer_image(rgb)  # H x W
    d = np.asarray(d, dtype=np.float32)
    d = np.where(np.isfinite(d), d, 0.0)

    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < 1e-6:
        return np.zeros_like(d, dtype=np.float32)

    depth = (d - d_min) / (d_max - d_min)
    return depth  # 0~1


def estimate_rim_depth_from_ellipse(depth, row, num_samples=120, win=1):
    """
    从椭圆边缘采样 depth（0~1），估计壶口“壶面”的深度值 Z_rel_vis
    depth: 整幅图的归一化深度图 (H, W)
    row:   包含 cx, cy, MA, ma, angle 的一行（pandas Series）
    num_samples: 采样点数
    win:   采样点邻域半径（win=1 表示 3x3 平均；win=0 表示单点）
    """
    h, w = depth.shape

    cx = float(row["cx"])
    cy = float(row["cy"])
    MA = float(row["MA"])
    ma = float(row["ma"])
    angle_deg = float(row.get("angle", 0.0))

    a = MA / 2.0
    b = ma / 2.0
    if a <= 0 or b <= 0:
        return None

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    ts = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    xs = a * np.cos(ts)
    ys = b * np.sin(ts)

    us = cx + xs * cos_t - ys * sin_t
    vs = cy + xs * sin_t + ys * cos_t

    vals = []
    for u, v in zip(us, vs):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue

        if win <= 0:
            val = depth[vi, ui]
            if np.isfinite(val):
                vals.append(val)
        else:
            x1 = max(0, ui - win)
            x2 = min(w, ui + win + 1)
            y1 = max(0, vi - win)
            y2 = min(h, vi + win + 1)
            patch = depth[y1:y2, x1:x2]
            patch = patch[np.isfinite(patch)]
            if patch.size > 0:
                vals.append(float(patch.mean()))

    if len(vals) < num_samples * 0.3:
        return None

    vals = np.asarray(vals, dtype=np.float32)
    d1, d2 = np.percentile(vals, [5, 95])
    vals = vals[(vals >= d1) & (vals <= d2)]
    if vals.size == 0:
        return None

    return float(np.median(vals))  # 0~1


def create_pointcloud(depth, row, spout_id, roi_pad=120):
    """
    基于 depth 和单个壶口拟合结果，生成局部点云并绘制。
    depth: 整幅图 0~1 深度。
    """
    # -------------------- 字体放大配置 --------------------
    FONT_SCALE = 1.35
    FS_TITLE = int(14 * FONT_SCALE)
    FS_LABEL = int(12 * FONT_SCALE)
    FS_TICK  = int(10 * FONT_SCALE)
    FS_CBAR  = int(11 * FONT_SCALE)

    # 你要的效果：不要刻度太密，但不要固定步长
    Z_NBINS = 5           # Z 轴大概 5 个主刻度（想更疏就改 4，想更密就改 6）
    CBAR_NBINS = 8        # 色条刻度数量控制

    cx, cy = int(row['cx']), int(row['cy'])
    h, w = depth.shape

    # ROI 提取
    size = int(max(float(row['MA']), float(row['ma'])) * 1.3) + roi_pad
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, cx + size // 2)
    y2 = min(h, cy + size // 2)

    roi_depth = depth[y1:y2, x1:x2]   # 0~1
    H, W = roi_depth.shape
    if H == 0 or W == 0:
        raise ValueError("ROI 尺寸为 0，检查椭圆参数是否异常")

    yy, xx = np.mgrid[0:H:1, 0:W:1]
    X = (xx + x1).ravel()
    Y = (yy + y1).ravel()
    Z = (roi_depth * Z_SCALE).ravel()  # 显示用深度（单位用 px 表达）

    # 壶面深度估计（椭圆边缘采样），失败则退化为中心点
    z_surface = estimate_rim_depth_from_ellipse(depth, row, num_samples=120, win=1)
    if z_surface is None:
        z_surface = float(depth[cy, cx])

    Z_rel_vis = float(np.clip(z_surface, 0.0, 1.0))
    Z_vis = Z_rel_vis * Z_SCALE
    Z_rel_cam = float(1.0 - Z_rel_vis)

    center_X, center_Y = cx, cy
    fit_score = float(row.get("fit_score", 0.0))

    # -------------------- 绘图 --------------------
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ===== 关键：色条范围用当前数据 vmax（避免动不动顶到 200）=====
    z_max = float(np.max(Z))
    vmax = max(1e-6, z_max)  # 防止全0导致报错

    scatter = ax.scatter(
        X, Y, Z,
        c=Z, cmap='rainbow',
        s=1.2, alpha=0.9, linewidth=0,
        vmin=0.0, vmax=vmax
    )

    ax.set_title(
        f"{Path(str(row['image'])).stem}  油壶{spout_id}\n"
        f"中心 ({center_X},{center_Y}), Z_rel={Z_rel_vis:.3f}, Z={Z_vis:.1f} | "
        f"RANSAC得分 {fit_score:.3f}",
        fontsize=FS_TITLE, pad=25
    )

    # 坐标轴标签
    ax.set_xlabel('X (px)', fontsize=FS_LABEL, labelpad=10)
    ax.set_ylabel('Y (px)', fontsize=FS_LABEL, labelpad=10)
    ax.set_zlabel('Z (px)', fontsize=FS_LABEL, labelpad=10)

    # 刻度字号
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.tick_params(axis='z', labelsize=FS_TICK)

    # ===== 关键：Z轴刻度“不要太细”，但仍自适应（不固定步长）=====
    ax.zaxis.set_major_locator(MaxNLocator(nbins=Z_NBINS))

    # Z 轴范围（给一点点 padding，但不强行固定到 200）
    z_min = float(np.min(Z))
    z_rng = max(1e-6, z_max - z_min)
    pad = 0.03 * z_rng
    ax.set_zlim(max(0.0, z_min - pad), z_max + pad)

    ax.view_init(elev=35, azim=-65)

    # ===== 关键：色条刻度均匀、好看，且不会强行出现 200 =====
    cbar = fig.colorbar(scatter, shrink=0.6, pad=0.08)
    cbar.set_label(f'缩放后深度 (×{Z_SCALE:.0f})', fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_TICK)

    # 让色条刻度自动选“均匀且圆润”的数
    cbar.locator = MaxNLocator(nbins=CBAR_NBINS)
    cbar.update_ticks()
    cbar.ax.set_ylim(0.0, vmax)  # 贴齐范围

    plt.tight_layout()

    save_name = f"{Path(str(row['image'])).stem}_油壶{spout_id:02d}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'image': row['image'],
        '壶口编号': spout_id,
        'center_X': center_X,
        'center_Y': center_Y,
        'Z_rel_vis': Z_rel_vis,
        'Z_rel_cam': Z_rel_cam,
        'Z_vis': Z_vis,
        'fit_score': fit_score,
        'Z缩放系数': Z_SCALE
    }


# ====================== 批量执行 ======================
df = pd.read_csv(ELLIPSE_CSV)
results = []
depth_cache = {}

grouped = df.groupby('image')
for img_name, group in grouped:
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"[WARN] {img_name} 不存在，跳过")
        continue

    if img_name not in depth_cache:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] {img_name} 读取失败，跳过")
            continue
        depth_cache[img_name] = get_depth(img_bgr)

    depth = depth_cache[img_name]

    for i, (_, row) in enumerate(group.iterrows(), start=1):
        try:
            res = create_pointcloud(depth, row, spout_id=i)
            results.append(res)
            print(f"{img_name} → 油壶{i} ✅ RANSAC得分={float(row.get('fit_score', 0.0)):.3f}")
        except Exception as e:
            print(f"{img_name} → 油壶{i} ❌ 出错: {e}")


# ====================== 保存结果 ======================
df_out = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "SPOUT_3D_Zx200_SURFACE_CENTER_WITH_INV.csv")
df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 壶口点云已生成（Z 轴向下为正 + ×{Z_SCALE:.0f} 缩放 + 壶面中心Z + 反向相对值）")
print(f"📁 图像输出路径：{OUTPUT_DIR}")
print(f"📄 CSV 文件路径：{csv_path}")
