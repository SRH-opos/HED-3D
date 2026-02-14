
import ast
import math
import random
from pathlib import Path
import logging
import os

import numpy as np
import pandas as pd
import cv2

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

# ---- 可选：学习型权重 ----
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from weight_net import WeightNet, load_weight_model
    WEIGHT_NET_MODULE_AVAILABLE = True
except Exception:
    WEIGHT_NET_MODULE_AVAILABLE = False

# =========================
# 路径 & 全局配置
# =========================
INPUT_CSV  = r"F:\YOLO\DAV2\FOUR\ellipses_third_stage.csv"
OUTPUT_CSV = r"F:\YOLO\DAV2\FOUR\ellipse_fused_kmeans_dynamic.csv"
IMAGE_DIR  = r"F:\YOLO\DAV2\FOUR\5.1"
VIS_DIR    = Path(IMAGE_DIR) / "fusion_vis_kmeans_dynamic"
VIS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_VIS          = True
KMAX              = 5            # K 搜索上线，避免过拟合
MIN_INLIERS_FINAL = 6
RANSAC_TRIES      = 7            # 5 -> 7
MAX_AXIS_RATIO    = 6.0
SEED              = 42

# 动态权重
BETA_DW = 0.6                    # 降低动态加权强度（0.4~0.8 可调）
A_STAB, B_INLIER, C_RESID, D_DENS = 0.5, 0.3, 0.1, 0.1

# 覆盖度调制（默认关闭）
USE_COVERAGE_MOD = False
COVERAGE_ALPHA   = 0.80          # 若开启，最终乘以 α+(1-α)*coverage（建议 α>=0.8）

# 学习型权重模型（可选）
WEIGHT_MODEL_PATH = r"F:\YOLO\weight_fusion_net.pth"
WEIGHT_DEVICE     = "cpu"

random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kmeans_dynamic")

# =========================
# 工具函数
# =========================
def safe_eval_fit_pts(x):
    """解析 CSV 的 fit_pts 为 (N,2) float32；为空/计数/异常 -> 返回空数组。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            arr = np.array(x, dtype=np.float32).reshape(-1, 2)
            return arr
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)
    try:
        arr = np.array(ast.literal_eval(str(x)), dtype=np.float32).reshape(-1, 2)
        return arr
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

def synthesize_points_from_rows(rows, base_step_px=4.0, jitter_sigma=0.3):
    """当 fit_pts 不可用时，依据 (cx,cy,MA,ma,angle) 合成椭圆边界点云。"""
    synth = []
    for _, r in rows.iterrows():
        try:
            cx, cy = float(r["cx"]), float(r["cy"])
            MA, ma, ang = float(r["MA"]), float(r["ma"]), float(r["angle"])
        except Exception:
            continue
        a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
        if a <= 0 or b <= 0:
            continue
        # Ramanujan 周长近似，控制点密度
        perim = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        n = int(np.clip(perim / max(1.0, base_step_px), 36, 400))
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ct, st = np.cos(t), np.sin(t)
        th = math.radians(ang)
        c, s = math.cos(th), math.sin(th)
        x = cx + a * ct * c - b * st * s
        y = cy + a * ct * s + b * st * c
        if jitter_sigma > 0:
            x += np.random.randn(n) * jitter_sigma
            y += np.random.randn(n) * jitter_sigma
        synth.append(np.stack([x, y], axis=1).astype(np.float32))
    if not synth:
        return None
    return np.vstack(synth)

def geometric_residuals(ellipse, pts):
    (cx, cy), (MA, ma), ang = ellipse
    a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
    if a <= 0 or b <= 0:
        return np.full(len(pts), np.inf, dtype=np.float32)
    th = math.radians(ang)
    ca, sa = math.cos(-th), math.sin(-th)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    xp, yp = ca * dx - sa * dy, sa * dx + ca * dy
    return np.abs((xp ** 2) / (a ** 2) + (yp ** 2) / (b ** 2) - 1.0)

def adaptive_ransac_v3(pts, img_diag=None, tries=RANSAC_TRIES):
    """返回 (best_ellipse, best_inliers_idx, stability, mean_residual)。"""
    pts = np.asarray(pts, np.float32)
    N = len(pts)
    if N < 6:
        return None, None, None, None

    bbox = np.ptp(pts, axis=0)
    roi_diag = max(np.hypot(*bbox), 1.0)
    density = N / (roi_diag + 1e-6)

    base_thresh   = float(np.clip(roi_diag * 0.012, 0.5, 12.0))
    inlier_thresh = float(base_thresh * np.clip(np.log1p(1.5 / (density + 1e-6)), 0.7, 3.0))
    if img_diag:
        inlier_thresh = min(inlier_thresh, float(img_diag * 0.06))

    w_est = float(np.clip(N / 150.0, 0.05, 0.6))
    s = 5
    denom = max(1e-9, 1 - w_est ** s)
    try:
        n_iter = int(np.clip(math.log(1 - 0.99) / math.log(denom), 200, 4000))
    except Exception:
        n_iter = 400

    candidates = []
    idx_all = np.arange(N)
    for _t in range(int(tries)):
        best_model, best_inliers = None, np.array([], dtype=int)
        inner_iter = max(1, n_iter // max(1, tries))
        for _ in range(inner_iter):
            try:
                subset = pts[np.random.choice(idx_all, size=s, replace=False)]
                cand = cv2.fitEllipse(subset)
            except Exception:
                continue
            res = geometric_residuals(cand, pts)
            inliers = np.where(res < inlier_thresh)[0]
            if len(inliers) > len(best_inliers):
                best_model, best_inliers = cand, inliers
        if best_model is not None and len(best_inliers) >= MIN_INLIERS_FINAL:
            try:
                refined = cv2.fitEllipse(pts[best_inliers])
            except Exception:
                refined = best_model
            candidates.append((refined, best_inliers))

    if not candidates:
        return None, None, None, None

    # 选择一致性评分最高的候选
    scores = []
    for cand, inl in candidates:
        met = evaluate_ellipse_quality(cand, pts, inl)
        scores.append(met["consistency_score"])
    best_idx = int(np.argmax(scores))
    best_ellipse, best_inliers = candidates[best_idx]

    # 残差 & 稳定度（中心抖动）
    res_inliers   = geometric_residuals(best_ellipse, pts[best_inliers])
    mean_residual = float(np.mean(res_inliers)) if res_inliers.size > 0 else 1.0
    centers = np.array([[c[0][0], c[0][1]] for (c, _) in candidates], dtype=np.float32)
    center_std = float(np.std(centers, axis=0).mean()) if centers.size > 0 else 0.0
    stability  = 1.0 / (1.0 + center_std)

    return best_ellipse, best_inliers, stability, mean_residual

def evaluate_ellipse_quality(ellipse, pts, inliers_idx):
    (cx, cy), (MA, ma), ang = ellipse
    if ma <= 0 or MA <= 0:
        return {"consistency_score": 0.0, "inlier_ratio": 0.0}
    ellipse_area = math.pi * (MA / 2.0) * (ma / 2.0)
    bbox_area = max(1.0, np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
    try:
        hull_area = ConvexHull(pts).volume
    except Exception:
        hull_area = bbox_area
    aspect_ratio = MA / max(1e-9, ma)
    inlier_ratio = len(inliers_idx) / max(1, len(pts))
    compactness  = ellipse_area / bbox_area
    ratio        = ellipse_area / max(1e-9, hull_area)
    score = (
        inlier_ratio * 0.45
        + (1 - abs(ratio - 1)) * 0.25
        + (1 - abs(compactness - 1)) * 0.15
        + (1 / (1 + abs(aspect_ratio - 1))) * 0.15
    )
    return {
        "inlier_ratio": float(inlier_ratio),
        "aspect_ratio": float(aspect_ratio),
        "compactness": float(compactness),
        "ellipse_hull_ratio": float(ratio),
        "consistency_score": float(np.clip(score, 0.0, 1.0)),
    }

# =========================
# KMeans: 选 K + 聚类
# =========================
def choose_k_by_silhouette(X, kmax=KMAX):
    n = len(X)
    if n < 4:
        return 1
    kmax_eff = int(max(2, min(kmax, n // 2 if n >= 8 else n - 1)))
    best_k, best_score = 2, -1.0
    for k in range(2, kmax_eff + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            labels = km.fit_predict(X)
            # silhouette 需要 >=2 个簇且每簇至少 2 个点时更稳定；异常时兜底
            s = silhouette_score(X, labels)
            if s > best_score:
                best_score, best_k = s, k
        except Exception:
            continue
    # 如果所有尝试都失败，降级为 1 或 2
    if best_score < 0:
        return 1 if n < 6 else 2
    return best_k

# =========================
# 单图像处理
# =========================
def process_one_image(img_name, group):
    pts_xy = group[["cx", "cy"]].to_numpy(np.float32)
    N = len(pts_xy)
    if N == 0:
        return [], None

    # 选择 K
    k = choose_k_by_silhouette(pts_xy, kmax=KMAX)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(pts_xy)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_size = dict(zip(unique_labels.tolist(), counts.tolist()))

    # 最小簇限制（放松 & 智能）
    min_cluster_size = max(3, int(round(0.04 * N)))   # 0.06*N -> 0.04*N
    if max(label_size.values()) < min_cluster_size:
        keep_labels = list(unique_labels)             # 都小：不裁剪
    else:
        keep_labels = [lb for lb, sz in label_size.items() if sz >= min_cluster_size]

    # 读图（可视化 & img_diag）
    img_path = Path(IMAGE_DIR) / img_name
    img = cv2.imread(str(img_path))
    vis = img.copy() if (SAVE_VIS and img is not None) else None
    img_diag = math.hypot(*img.shape[1::-1]) if img is not None else None

    image_results = []
    img_scores = []
    best_score_raw = -1.0

    # 预加载学习型权重
    model = None
    if WEIGHT_NET_MODULE_AVAILABLE and TORCH_AVAILABLE and Path(WEIGHT_MODEL_PATH).exists():
        try:
            if not hasattr(process_one_image, "_weight_model_loaded"):
                process_one_image._weight_model = load_weight_model(WEIGHT_MODEL_PATH, device=WEIGHT_DEVICE)
                process_one_image._weight_model_loaded = True
            model = process_one_image._weight_model
        except Exception as e:
            log.warning(f"学习型权重模型加载失败，回退规则加权：{e}")
            model = None

    for lbl in unique_labels:
        if lbl not in keep_labels:
            continue
        idx = np.where(labels == lbl)[0]
        rows_sub = group.iloc[idx]

        # 1) fit_pts 优先
        pts_all = None
        if "fit_pts" in rows_sub.columns:
            try:
                fit_pts_list = rows_sub["fit_pts"].apply(safe_eval_fit_pts).tolist()
                pts_arrays = [p for p in fit_pts_list if isinstance(p, np.ndarray) and p.shape[0] > 0 and p.shape[1] == 2]
                if len(pts_arrays) > 0:
                    pts_all = np.vstack(pts_arrays)
            except Exception:
                pts_all = None

        # 2) 回退：合成点云
        if pts_all is None or len(pts_all) < 6:
            pts_all = synthesize_points_from_rows(rows_sub, base_step_px=4.0, jitter_sigma=0.3)
        # 3) 最后回退：直接用簇内的 (cx,cy)
        if pts_all is None or len(pts_all) < 6:
            if len(idx) >= 6:
                pts_all = pts_xy[idx]
            else:
                continue

        ellipse, inliers, stability, mean_residual = adaptive_ransac_v3(pts_all, img_diag, tries=RANSAC_TRIES)
        if ellipse is None:
            continue

        metrics = evaluate_ellipse_quality(ellipse, pts_all, inliers)
        (cx, cy), (MA, ma), ang = ellipse
        if metrics["aspect_ratio"] > MAX_AXIS_RATIO or metrics["inlier_ratio"] < 0.2:
            continue

        # —— 动态权重：稳定度 / 内点率 / 残差 / 密度
        try:
            nn = NearestNeighbors(n_neighbors=min(6, max(2, len(pts_all)))).fit(pts_all)
            dists, _ = nn.kneighbors(pts_all)
            local_mean_k = np.mean(dists[:, 1:], axis=1) if dists.shape[1] > 1 else dists[:, -1]
            local_density = 1.0 / (local_mean_k + 1e-6)
            density_norm = (local_density - local_density.min()) / (local_density.ptp() + 1e-6)
            density_norm_val = float(np.mean(density_norm))
        except Exception:
            density_norm_val = 0.5

        residual_feat = 1.0 - float(np.clip(mean_residual, 0.0, 1e6))
        residual_feat = float(np.clip(residual_feat, 0.0, 1.0))
        stability = 0.0 if stability is None else float(stability)

        dynamic_weight = None
        if model is not None:
            try:
                x = np.array([stability, metrics.get("inlier_ratio", 0.0), residual_feat, density_norm_val], dtype=np.float32)
                x_t = torch.from_numpy(x).unsqueeze(0).to(WEIGHT_DEVICE)
                with torch.no_grad():
                    out = model(x_t)
                dynamic_weight = float(np.clip(out.squeeze().cpu().item(), 0.0, 1.0))
            except Exception as e:
                log.warning(f"学习型权重推理失败，回退规则动态加权: {e}")
                dynamic_weight = None

        if dynamic_weight is None:
            ir = float(np.clip(metrics.get("inlier_ratio", 0.0), 0.0, 1.0))
            dn = float(np.clip(density_norm_val, 0.0, 1.0))
            dynamic_weight = (
                A_STAB * stability + B_INLIER * ir + C_RESID * residual_feat + D_DENS * dn
            ) / (A_STAB + B_INLIER + C_RESID + D_DENS)
            dynamic_weight = float(np.clip(dynamic_weight, 0.0, 1.0))

        cons_raw = float(metrics.get("consistency_score", 0.0))
        best_score_raw = max(best_score_raw, cons_raw)

        weighted_cons = cons_raw + BETA_DW * (1.0 - cons_raw) * dynamic_weight

        # 覆盖度（可选）
        if USE_COVERAGE_MOD:
            coverage = float(len(idx)) / max(1, N)
            weighted_cons = weighted_cons * (COVERAGE_ALPHA + (1.0 - COVERAGE_ALPHA) * coverage)

        metrics["dynamic_weight"]    = float(dynamic_weight)
        metrics["consistency_score"] = float(np.clip(weighted_cons, 0.0, 1.0))

        entry = {
            "image": img_name,
            "cluster_label": int(lbl),
            "cx": float(cx), "cy": float(cy),
            "MA": float(MA), "ma": float(ma),
            "angle": float(ang),
            "stability": float(stability),
            "fit_pts": str(pts_all.tolist()),
            "best_score": float(best_score_raw),
            "inlier_ratio": float(metrics.get("inlier_ratio", 0.0)),
            "aspect_ratio": float(metrics.get("aspect_ratio", 0.0)),
            "compactness": float(metrics.get("compactness", 0.0)),
            "ellipse_hull_ratio": float(metrics.get("ellipse_hull_ratio", 0.0)),
            "dynamic_weight": float(metrics["dynamic_weight"]),
            "consistency_score": float(metrics["consistency_score"]),
        }
        image_results.append(entry)
        img_scores.append(metrics["consistency_score"])

        # 可视化叠加（整图一次保存）
        if vis is not None:
            color = tuple(int(c) for c in np.random.RandomState(int(lbl) + SEED).randint(0, 255, 3))
            cv2.ellipse(
                vis, (int(cx), int(cy)),
                (max(1, int(MA/2)), max(1, int(ma/2))),
                float(ang), 0, 360, color, 2, cv2.LINE_AA
            )
            txt = f"s={metrics['consistency_score']:.2f},dw={metrics['dynamic_weight']:.2f}"
            cv2.putText(vis, txt, (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 保存整图可视化
    if vis is not None:
        cv2.imwrite(str(VIS_DIR / img_name), vis)

    # 离群过滤（放松版）
    if image_results and len(image_results) > 2:
        scores_arr  = np.array([r["consistency_score"] for r in image_results], dtype=np.float32)
        stabs_arr   = np.array([r["stability"] for r in image_results], dtype=np.float32)
        q1, q3      = np.percentile(scores_arr, [25, 75])
        iqr         = q3 - q1
        iqr_mask    = (scores_arr >= q1 - 1.5 * iqr) & (scores_arr <= q3 + 1.5 * iqr) if iqr > 0 else np.ones_like(scores_arr, bool)
        stab_mask   = stabs_arr >= 0.2   # 0.3 -> 0.2
        # 默认关闭 z 分数过滤，避免小样本时放大波动
        keep_mask   = iqr_mask & stab_mask
        image_results = [r for i, r in enumerate(image_results) if keep_mask[i]]

    summary = {
        "image": img_name,
        "mean_consistency_score": float(np.mean(img_scores)) if img_scores else 0.0,
        "std_consistency_score":  float(np.std(img_scores))  if img_scores else 0.0,
        "valid_ellipse_count":    int(len(image_results)),
        "best_score":             float(best_score_raw)
    }
    return image_results, summary

# =========================
# 主流程
# =========================
def process_all():
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        log.error(f"读取输入 CSV 失败: {e}")
        return
    if df.empty:
        log.warning("输入为空")
        return

    results_all, summary_all = [], []
    log.info("开始处理图像（KMeans + 动态加权）...")

    for img_name, group in df.groupby("image"):
        res, summ = process_one_image(img_name, group)
        if res:  results_all.extend(res)
        if summ: summary_all.append(summ)

    if not results_all:
        log.info("未生成有效椭圆结果")
        return

    # 保存完整结果
    pd.DataFrame(results_all).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info(f"✅ 结果文件保存至 {OUTPUT_CSV}")

    # 概览
    summary_path = str(Path(OUTPUT_CSV).with_name("ellipse_summary_kmeans_dynamic.csv"))
    pd.DataFrame(summary_all).to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info(f"✅ 概览文件保存至 {summary_path}")

    # 提取每图最佳（按最终分 + 稳定度）
    df_res = pd.DataFrame(results_all)
    if "dynamic_weight" not in df_res.columns:
        df_res["dynamic_weight"] = 0.0

    df_best = (
        df_res.sort_values(["image", "consistency_score", "stability"], ascending=[True, False, False])
              .groupby("image", as_index=False)
              .first()[["image", "cx", "cy", "MA", "ma", "angle",
                        "consistency_score", "stability", "dynamic_weight", "best_score"]]
    )
    best_csv_path = str(Path(OUTPUT_CSV).with_name("ellipse_final_best_kmeans_dynamic.csv"))
    df_best.to_csv(best_csv_path, index=False, encoding="utf-8-sig")
    log.info(f"🏁 最稳定椭圆文件已保存至: {best_csv_path}")

    # 彩色可视化（仅最佳）
    if SAVE_VIS:
        for _, row in df_best.iterrows():
            img_name = row["image"]
            img_path = Path(IMAGE_DIR) / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            vis_img = img.copy()
            cx, cy = int(row["cx"]), int(row["cy"])
            MA, ma = max(1, int(row["MA"]/2)), max(1, int(row["ma"]/2))
            ang = float(row["angle"])
            cv2.ellipse(vis_img, (cx, cy), (MA, ma), ang, 0, 360, (0, 0, 255), 2)
            cv2.circle(vis_img, (cx, cy), 4, (0, 255, 0), -1)
            cv2.imwrite(str(VIS_DIR / img_name), vis_img)
        log.info(f"✅ 彩色融合可视化已保存至: {VIS_DIR}")

    # 灰度黑底纯椭圆（仅最佳）
    ellipse_only_dir = VIS_DIR / "ellipse_only"
    ellipse_only_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df_best.iterrows():
        img_name = row["image"]
        img_path = Path(IMAGE_DIR) / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        ellipse_img = np.zeros((H, W), dtype=np.uint8)
        cx, cy = int(row["cx"]), int(row["cy"])
        MA, ma = max(1, int(row["MA"]/2)), max(1, int(row["ma"]/2))
        ang = float(row["angle"])
        cv2.ellipse(ellipse_img, (cx, cy), (MA, ma), ang, 0, 360, 255, -1)
        cv2.circle(ellipse_img, (cx, cy), 2, 255, -1)
        cv2.imwrite(str(ellipse_only_dir / img_name), ellipse_img)
    log.info(f"✅ 灰度纯椭圆图像已保存至: {ellipse_only_dir}")
    log.info("🎯 全部处理完成！")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    # 学习型权重模型预检查（可选）
    if WEIGHT_NET_MODULE_AVAILABLE and TORCH_AVAILABLE:
        if Path(WEIGHT_MODEL_PATH).exists():
            log.info(f"尝试加载学习型加权模型: {WEIGHT_MODEL_PATH}")
            try:
                _ = load_weight_model(WEIGHT_MODEL_PATH, device=WEIGHT_DEVICE)
                log.info("✅ 学习型加权模型加载成功（将在推理时使用）")
            except Exception as e:
                log.warning(f"加载学习型加权模型失败，将回退到规则动态加权：{e}")
        else:
            log.info("未找到学习型权重文件，脚本将使用规则动态加权（回退策略）")
    else:
        log.info("weight_net 模块或 torch 不可用，脚本将使用规则动态加权（回退策略）")

    process_all()
