# -*- coding: utf-8 -*-
import os, math, shutil, re, csv, random
from pathlib import Path
import numpy as np
import cv2
import random

# ========== 路径 ==========
IN_DIR   = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3_refined_v2_noellipse")  # 这 98 张（二轮无椭圆）
OUT_DIR  = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v4_recovered")             # 恢复输出
KEEP_DIR = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v4_recovered_still_noellipse")  # 仍失败的输入
CSV_PATH = OUT_DIR / "ellipses_fourth_stage.csv"  # CSV 汇总

# ======= 统一输出命名 =======
NAME_MODE = "auto"
OUT_EXT   = ".png"
EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

# ========== 其他参数 ==========
MAX_OUT_ELLIPSES   = 3     # 模型内最多真实拟合多少个（保持不变）
FORCE_OUTPUT_COUNT = 5     # 对外强制输出数量（不足则重复高质量补齐）
DRAW_THICK         = 2

# ======== 质量阈值（可按需调）========
MIN_MAJOR_REL   = 0.10
SUPPORT_MIN     = 0.45
ANG_COVER_MIN   = 180.0
MAX_GAP_DEG_MAX = 150.0
BAND_TOL        = 0.20
CENTER_FINE_ITR = 8
CENTER_FINE_STEP= 1.0

# ======== 弧段生长参数（端点桥接） ========
GROW_D_FRAC = 0.03            # 距离上限占 min(H,W) 的比例
GROW_D_MAX_PX = 10            # 距离上限像素封顶
GROW_THETA_ALIGN_DEG = 25.0   # 端点切向与连线方向的对齐阈值
GROW_MAX_BRIDGES = 64         # 最多桥接条数，防止过桥
# 验收门槛（若生长后没明显收益则回退）
GROW_ACCEPT_FITSCORE_DELTA = 0.05
GROW_ACCEPT_COVER_INC_DEG  = 15.0
GROW_ALLOW_MAXGAP_WORSEN   = 10.0  # 允许 max_gap 轻微变差的容忍上限（度）

# ======== 稳中心/鲁棒重拟合参数 ========
BISECT_N_PAIRS = 200          # 弦对数量（随机）
BISECT_MIN_ARCIDX_GAP = 8     # 同一弧段采样点下标至少相隔
CENTER_USE_MEDIAN = True      # 中值聚合（稳健）
RESID_TUKEY_C = 0.25          # Tukey 截断常数（|rho-1| 的阈值）
ANG_BINS = 36                 # 角度分箱数
MAX_PER_BIN = 12              # 每箱最多样本，用于均衡采样
REFIT_ACCEPT_FITSCORE_DELTA = 0.03
REFIT_ACCEPT_COVER_INC_DEG  = 10.0
REFIT_ALLOW_MAXGAP_WORSEN   = 8.0

# ================== 工具：命名与目录 ==================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_out_path(dir_path: Path, base_stem: str, ext: str = OUT_EXT) -> Path:
    save_path = dir_path / f"{base_stem}{ext}"
    if not save_path.exists():
        return save_path
    k = 1
    while True:
        cand = dir_path / f"{base_stem}_dup{k}{ext}"
        if not cand.exists():
            return cand
        k += 1

def parse_last_number(stem: str):
    nums = re.findall(r'(\d+)', stem)
    return int(nums[-1]) if nums else None

def make_output_basename(in_name: str) -> str:
    s = Path(in_name).stem
    s_low = s.lower()
    num = parse_last_number(s_low)
    if num is None:
        # 无数字就用原 stem（小写）
        return s_low
    # 有数字时只允许 “数字” 或 “spout_数字”
    return f"spout_{num}" if "spout" in s_low else f"{num}"


# ======== 边缘/骨架/中心 ========
def to_binary_from_green(bgr):
    """
    统一把轮廓转成二值：
    - 若图像中存在明显绿色：按 HSV 提取绿色（兼容旧的黑底绿线图）；
    - 否则认为是白底黑线：灰度+反阈值提取“黑线”。
    """
    if bgr is None:
        return None

    # 1) 尝试按绿色提取
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 60, 60], np.uint8)
    upper = np.array([85, 255, 255], np.uint8)
    mask_green = cv2.inRange(hsv, lower, upper)

    if np.count_nonzero(mask_green) > 0:
        mask = mask_green
    else:
        # 2) 没有绿色：默认白底黑线 → 取暗像素为前景
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 阈值 200 可按需要微调
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 3) 小闭运算，去小孔
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def thinning(binary):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img = binary.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

# ---------- 弧段生长：角度约束 + 距离约束 ----------
def degree_map(skel_bin):
    m = (skel_bin > 0).astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    nb = cv2.filter2D(m, -1, k) - m
    return nb

def estimate_tangent_field(skel_bin):
    f = cv2.GaussianBlur((skel_bin > 0).astype(np.float32), (5,5), 0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    tx, ty = -gy, gx
    n = np.sqrt(tx*tx + ty*ty) + 1e-6
    return (tx / n, ty / n)

def angle_deg_between(vx1, vy1, vx2, vy2):
    dot = vx1*vx2 + vy1*vy2
    n1 = np.sqrt(vx1*vx1 + vy1*vy1) + 1e-9
    n2 = np.sqrt(vx2*vx2 + vy2*vy2) + 1e-9
    c = np.clip(dot / (n1*n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(np.abs(c))))

def grow_arcs_angle_aware(skel_bin):
    H, W = skel_bin.shape
    d_max = int(min(GROW_D_MAX_PX, GROW_D_FRAC * min(H, W)))
    if d_max < 2: return skel_bin
    deg = degree_map(skel_bin)
    m = (skel_bin > 0).astype(np.uint8)
    endpoints = np.column_stack(np.where((m>0) & (deg==1)))
    if endpoints.shape[0] < 2:
        return skel_bin
    tx, ty = estimate_tangent_field(skel_bin)
    canvas = skel_bin.copy()
    made = 0
    N = endpoints.shape[0]
    for i in range(N):
        y1, x1 = int(endpoints[i,0]), int(endpoints[i,1])
        for j in range(i+1, N):
            y2, x2 = int(endpoints[j,0]), int(endpoints[j,1])
            dx, dy = (x2 - x1), (y2 - y1)
            dist = math.hypot(dx, dy)
            if dist < 2 or dist > d_max:
                continue
            ang1 = angle_deg_between(tx[y1,x1], ty[y1,x1], dx, dy)
            ang2 = angle_deg_between(tx[y2,x2], ty[y2,x2], -dx, -dy)
            if ang1 > GROW_THETA_ALIGN_DEG or ang2 > GROW_THETA_ALIGN_DEG:
                continue
            cv2.line(canvas, (x1,y1), (x2,y2), 255, 1, lineType=cv2.LINE_8)
            made += 1
            if made >= GROW_MAX_BRIDGES: break
        if made >= GROW_MAX_BRIDGES: break
    if made > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k, iterations=1)
        canvas = thinning(canvas)
    return canvas

# ======== 径向对称投票（FRST-like） ========
def radial_symmetry_center(bin_img, r_min=None, r_max=None, r_step=6):
    H, W = bin_img.shape
    ys, xs = np.where(bin_img > 0)
    if xs.size < 20:
        return (W/2.0, H/2.0), None
    f = cv2.GaussianBlur(bin_img.astype(np.float32), (5,5), 0) / 255.0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-6
    ux, uy = gx/mag, gy/mag
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    box_w, box_h = xmax - xmin + 1, ymax - ymin + 1
    base = 0.35 * float(min(box_w, box_h))
    if r_min is None: r_min = max(12, int(base*0.6))
    if r_max is None: r_max = max(r_min+6, int(base*1.6))
    acc = np.zeros((H, W), np.float32)
    for r in range(r_min, r_max+1, r_step):
        cx = (xs - (ux[ys, xs]*r)).round().astype(np.int32)
        cy = (ys - (uy[ys, xs]*r)).round().astype(np.int32)
        m  = np.clip(mag[ys, xs], 0, 1.0)
        ok = (cx>=0)&(cx<W)&(cy>=0)&(cy<H)
        if ok.any():
            acc[cy[ok], cx[ok]] += m[ok]
    acc_blur = cv2.GaussianBlur(acc, (9,9), 0)
    _, _, _, maxLoc = cv2.minMaxLoc(acc_blur)
    mx, my = maxLoc
    return (float(mx), float(my)), acc_blur

# ======== 椭圆拟合（中心约束，角度粗细搜，A,B 解析解） ========
def solve_AB_given_center_theta(pts, cx, cy, theta, weights=None):
    ct, st = math.cos(theta), math.sin(theta)
    X = pts - np.array([cx, cy], np.float32)
    xp =  X[:,0]*ct + X[:,1]*st
    yp = -X[:,0]*st + X[:,1]*ct
    X2, Y2 = xp*xp, yp*yp
    W = np.ones_like(X2, dtype=np.float32) if weights is None else weights.astype(np.float32)
    Sxx = np.sum(W*X2*X2); Syy = np.sum(W*Y2*Y2); Sxy = np.sum(W*X2*Y2)
    bx  = np.sum(W*X2);    by  = np.sum(W*Y2)
    M = np.array([[Sxx, Sxy],[Sxy, Syy]], dtype=np.float64)
    v = np.array([bx, by], dtype=np.float64)
    det = np.linalg.det(M)
    if abs(det) < 1e-9: return None
    a, b = np.linalg.solve(M, v)  # a=1/A^2, b=1/B^2
    if a<=1e-9 or b<=1e-9: return None
    A = 1.0 / math.sqrt(a); B = 1.0 / math.sqrt(b)
    return A, B, xp, yp

def ellipse_support_and_cover(xp, yp, A, B, band_tol=BAND_TOL):
    rho = np.sqrt((xp/(A+1e-6))**2 + (yp/(B+1e-6))**2)
    band = np.abs(rho - 1.0)
    support = float(np.mean(band < band_tol))
    t = np.arctan2(yp/(B+1e-6), xp/(A+1e-6))
    t = (t + 2*np.pi) % (2*np.pi)
    if t.size == 0: return support, 0.0, 360.0
    t = np.sort(t)
    gaps = np.diff(t)
    last_gap = (t[0] + 2*np.pi) - t[-1]
    gaps = np.concatenate([gaps, [last_gap]])
    max_gap_deg = float(np.max(gaps)) * 180.0 / math.pi
    ang_cover = 360.0 - max_gap_deg
    return support, ang_cover, max_gap_deg

def ellipse_metrics(e, pts, wh):
    """
    根据椭圆参数 e=((cx,cy),(MA,ma),ang) 和参与拟合点 pts，
    计算 A/B 半径、支持度、角度覆盖、最大缺口 等指标。
    带宽阈值使用全局 BAND_TOL。
    """
    (cx,cy),(MA,ma),ang = e
    A = max(MA,ma)/2.0
    B = min(MA,ma)/2.0
    if A < 1 or B < 1:
        return None

    th = math.radians(ang)
    c,s = math.cos(th), math.sin(th)
    R = np.array([[c, -s],[s, c]], np.float32)

    X  = pts.astype(np.float32) - np.array([cx,cy], np.float32)
    Xp = X @ R

    rho = np.sqrt((Xp[:,0]/(A+1e-6))**2 + (Xp[:,1]/(B+1e-6))**2)
    band = np.abs(rho - 1.0)
    support = float(np.mean(band < BAND_TOL))

    t = np.arctan2(Xp[:,1]/(B+1e-6), Xp[:,0]/(A+1e-6))
    t = (t + 2*np.pi) % (2*np.pi)
    if t.size >= 1:
        t = np.sort(t)
        gaps = np.diff(t)
        last_gap = (t[0] + 2*np.pi) - t[-1]
        gaps = np.concatenate([gaps, [last_gap]])
        max_gap = float(np.max(gaps)) * 180.0 / math.pi
        ang_cover = 360.0 - max_gap
    else:
        ang_cover, max_gap = 0.0, 360.0

    return dict(A=A, B=B, support=support, ang_cover=ang_cover, max_gap=max_gap)


def refine_center_local(pts, cx, cy, theta, A, B, steps=CENTER_FINE_ITR, step=CENTER_FINE_STEP):
    ct, st = math.cos(theta), math.sin(theta)
    def cost(cxx, cyy):
        X = pts - np.array([cxx, cyy], np.float32)
        xp =  X[:,0]*ct + X[:,1]*st
        yp = -X[:,0]*st + X[:,1]*ct
        rho = np.sqrt((xp/(A+1e-6))**2 + (yp/(B+1e-6))**2)
        return float(np.mean(np.abs(rho-1.0)))
    best = (cx, cy)
    best_cost = cost(cx, cy)
    for _ in range(steps):
        improved = False
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                if dx==0 and dy==0: continue
                ccx, ccy = best[0]+dx, best[1]+dy
                c = cost(ccx, ccy)
                if c < best_cost - 1e-6:
                    best_cost, best = c, (ccx, ccy)
                    improved = True
        if not improved:
            break
    return best

def fit_one_ellipse_center_constrained(pts, cen, hw):
    H, W = hw
    cx, cy = cen
    min_major_pix = max(32.0, MIN_MAJOR_REL*min(H,W))
    if pts.shape[0] < 20: return None
    best = None; best_err = 1e9
    for ang_deg in range(0, 180, 2):
        th = math.radians(ang_deg)
        sol = solve_AB_given_center_theta(pts, cx, cy, th)
        if sol is None: continue
        A, B, xp, yp = sol
        if max(A,B) < min_major_pix: continue
        rho = np.sqrt((xp/(A+1e-6))**2 + (yp/(B+1e-6))**2)
        err = float(np.mean(np.abs(rho - 1.0)))
        if err < best_err:
            best_err = err
            best = (th, A, B, xp, yp)
    if best is None: return None
    th, A, B, _, _ = best
    # 角度细调
    def local_search(th0, rng_deg=5.0, step_deg=0.5):
        best_local = (th0, A, B, best_err)
        th_cand = np.arange(th0 - math.radians(rng_deg),
                            th0 + math.radians(rng_deg)+1e-6,
                            math.radians(step_deg))
        for th1 in th_cand:
            sol = solve_AB_given_center_theta(pts, cx, cy, th1)
            if sol is None: continue
            A1, B1, xp1, yp1 = sol
            if max(A1,B1) < min_major_pix: continue
            rho = np.sqrt((xp1/(A1+1e-6))**2 + (yp1/(B1+1e-6))**2)
            err = float(np.mean(np.abs(rho - 1.0)))
            if err < best_local[3]:
                best_local = (th1, A1, B1, err)
        return best_local
    th, A, B, _ = local_search(th)
    # 中心微调 + 重新解 AB
    cx, cy = refine_center_local(pts, cx, cy, th, A, B)
    sol = solve_AB_given_center_theta(pts, cx, cy, th)
    if sol is None: return None
    A, B, xp, yp = sol
    # 质量
    support, ang_cover, max_gap = ellipse_support_and_cover(xp, yp, A, B, BAND_TOL)
    if max(A,B) < min_major_pix:           return None
    if support  < SUPPORT_MIN:             return None
    if ang_cover < ANG_COVER_MIN:          return None
    if max_gap  > MAX_GAP_DEG_MAX:         return None
    MA, ma = 2*max(A,B), 2*min(A,B)
    ang_deg = (th*180.0/math.pi) % 180.0
    ellipse = ((float(cx),float(cy)), (float(MA),float(ma)), float(ang_deg))
    metrics = dict(A=max(A,B), B=min(A,B), support=support, ang_cover=ang_cover, max_gap=max_gap)
    return ellipse, metrics

# ---------- 公用评分 ----------
def compute_fit_score(met):
    size_w = min(1.0, met['A'] / 40.0)
    cover  = max(0.0, min(1.0, met['ang_cover'] / 360.0))
    gap_w  = max(0.0, 1.0 - met['max_gap'] / 360.0)
    return float(met['support'] * cover * gap_w * size_w)

def grade_quality(met):
    if met['support'] >= 0.60 and met['ang_cover'] >= 260 and met['max_gap'] <= 100:
        return "A"
    if met['support'] >= 0.45 and met['ang_cover'] >= 200:
        return "B"
    return "C"

# ---------- 稳中心（弦垂直平分线交点投票） ----------
def bisector_center_vote(pts, n_pairs=BISECT_N_PAIRS, min_gap=BISECT_MIN_ARCIDX_GAP):
    N = pts.shape[0]
    if N < min_gap*2 + 2:
        return np.mean(pts, axis=0).tolist()
    centers = []
    for _ in range(n_pairs):
        i = random.randint(0, N-1); j = random.randint(0, N-1)
        if abs(i-j) < min_gap: continue
        p1 = pts[i].astype(np.float64); p2 = pts[j].astype(np.float64)
        m1 = 0.5*(p1+p2); d1 = p2 - p1; n1 = np.array([-d1[1], d1[0]], dtype=np.float64)
        k = random.randint(0, N-1); l = random.randint(0, N-1)
        if abs(k-l) < min_gap or len({i,j,k,l})<4: continue
        p3 = pts[k].astype(np.float64); p4 = pts[l].astype(np.float64)
        m2 = 0.5*(p3+p4); d2 = p4 - p3; n2 = np.array([-d2[1], d2[0]], dtype=np.float64)
        M = np.array([n1, -n2]).T; b = (m2 - m1)
        det = np.linalg.det(M)
        if abs(det) < 1e-8: continue
        ts = np.linalg.solve(M, b)
        cxy = m1 + ts[0]*n1
        centers.append(cxy)
    if not centers:
        return np.mean(pts, axis=0).tolist()
    centers = np.array(centers)
    c = np.median(centers, axis=0) if CENTER_USE_MEDIAN else np.mean(centers, axis=0)
    return [float(c[0]), float(c[1])]

# ---------- 残差/内点/角度均衡 ----------
def residuals_to_ellipse(e, pts):
    (cx,cy),(MA,ma),ang = e
    A = max(MA,ma)/2.0; B = min(MA,ma)/2.0
    th = math.radians(ang)
    c,s = math.cos(th), math.sin(th)
    R = np.array([[c,-s],[s,c]], np.float32)
    X = pts.astype(np.float32) - np.array([cx,cy], np.float32)
    Xp = X @ R
    rho = np.sqrt((Xp[:,0]/(A+1e-6))**2 + (Xp[:,1]/(B+1e-6))**2)
    return np.abs(rho - 1.0), Xp, A, B

def tukey_inlier_mask(resid, c=RESID_TUKEY_C):
    return (resid <= c)

def angular_balanced_subset(Xp, A, B, mask, per_bin=MAX_PER_BIN, bins=ANG_BINS):
    t = np.arctan2(Xp[:,1]/(B+1e-6), Xp[:,0]/(A+1e-6))
    t = (t + 2*np.pi) % (2*np.pi)
    idx = np.arange(len(t))
    idx = idx[mask]
    if idx.size == 0: return None
    bin_ids = (t[idx] / (2*np.pi) * bins).astype(np.int32)
    sel = []
    for b in range(bins):
        ids = idx[bin_ids==b]
        if ids.size == 0: continue
        if ids.size > per_bin:
            choice = np.random.choice(ids, size=per_bin, replace=False)
            sel.extend(choice.tolist())
        else:
            sel.extend(ids.tolist())
    if not sel: return None
    return np.array(sel, dtype=np.int32)

# ---------- 结果构建 ----------
def build_results_from_skel(bin0, skel):
    H, W = bin0.shape
    ys, xs = np.where(skel > 0)
    if len(xs) < 30:
        return []
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    # 1) 全局中心（FRST）用于半径聚类
    (cx0, cy0), _ = radial_symmetry_center(bin0)

    # 2) 半径聚类（外→内）
    v = pts - np.array([cx0,cy0], np.float32)
    rad = np.sqrt(np.sum(v*v, axis=1))
    K_try = [3,2,1]
    out = []
    for K in K_try:
        if len(rad) < 30 or K==1:
            clusters = [np.arange(len(rad))]
        else:
            r = rad.reshape(-1,1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            _ret, labels, centers = cv2.kmeans(r, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            clusters = [np.where(labels.ravel()==k)[0] for k in range(K)]
            order = np.argsort(centers.ravel())[::-1]  # 外到内
            clusters = [clusters[i] for i in order]

        out = []
        for idx in clusters:
            if idx.size < 20:
                continue
            pts_k = pts[idx]

            # 2.1 中心约束拟合（FRST 中心）
            cand0 = fit_one_ellipse_center_constrained(pts_k, (cx0,cy0), (H,W))

            # 2.2 稳中心：弦平分线投票，再中心约束拟合（二选一）
            cx_bi, cy_bi = bisector_center_vote(pts_k)
            cand1 = fit_one_ellipse_center_constrained(pts_k, (cx_bi,cy_bi), (H,W)) if cand0 is not None else None

            use_cand = None
            if cand0 is None and cand1 is not None:
                use_cand = cand1
            elif cand0 is not None and cand1 is None:
                use_cand = cand0
            elif cand0 is not None and cand1 is not None:
                m0, m1 = cand0[1], cand1[1]
                use_cand = cand1 if compute_fit_score(m1) > compute_fit_score(m0) else cand0
            else:
                continue

            e_use, met_use = use_cand

            # 2.3 鲁棒重拟合：Tukey 截断 + 角度均衡子采样 → 非约束 fitEllipse
            resid, Xp, A0, B0 = residuals_to_ellipse(e_use, pts_k)
            mask = tukey_inlier_mask(resid, c=RESID_TUKEY_C)
            sel = angular_balanced_subset(Xp, A0, B0, mask, per_bin=MAX_PER_BIN, bins=ANG_BINS)
            if sel is not None and sel.size >= 12:
                e_ref = cv2.fitEllipse(pts_k[sel].astype(np.float32))
                if e_ref is not None:
                    met_ref = ellipse_metrics(e_ref, pts_k, (W, H))
                    if met_ref is not None:
                        fs_new, fs_old = compute_fit_score(met_ref), compute_fit_score(met_use)
                        cover_inc = met_ref['ang_cover'] - met_use['ang_cover']
                        gap_worse = met_ref['max_gap']   - met_use['max_gap']
                        if (fs_new - fs_old >= REFIT_ACCEPT_FITSCORE_DELTA) or \
                           (cover_inc >= REFIT_ACCEPT_COVER_INC_DEG and gap_worse <= REFIT_ALLOW_MAXGAP_WORSEN):
                            e_use, met_use = e_ref, met_ref  # 接受更优的非约束重拟合

            out.append((e_use, met_use, int(idx.size)))

        if out:
            break

    out.sort(key=lambda t: t[1]['A'], reverse=True)
    return out[:MAX_OUT_ELLIPSES]

def best_tuple(results):
    return None if not results else results[0][1]  # 返回 metrics

def growth_is_better(res_grow, res_base):
    if res_base and not res_grow: return False
    if res_grow and not res_base: return True
    if not res_grow and not res_base: return False
    mg = best_tuple(res_grow); mb = best_tuple(res_base)
    fs_g = compute_fit_score(mg); fs_b = compute_fit_score(mb)
    cover_inc = mg['ang_cover'] - mb['ang_cover']
    gap_worse = mg['max_gap']   - mb['max_gap']
    return (fs_g - fs_b >= GROW_ACCEPT_FITSCORE_DELTA) or \
           (cover_inc >= GROW_ACCEPT_COVER_INC_DEG and gap_worse <= GROW_ALLOW_MAXGAP_WORSEN)

# ======== 主处理：读一张二轮图 → 骨架（含弧段生长） → KMeans 分层 → 拟合/重拟合 ========
def process_one_image(bgr):
    bin0 = to_binary_from_green(bgr)
    if bin0 is None or cv2.countNonZero(bin0)==0:
        return []
    skel = thinning(bin0)
    skel_grow = grow_arcs_angle_aware(skel)
    res_base = build_results_from_skel(bin0, skel)
    res_grow = build_results_from_skel(bin0, skel_grow)
    return res_grow if growth_is_better(res_grow, res_base) else res_base

def draw_overlay(HW, ellipses):
    """
    输出：白底 + 黑线椭圆。
    """
    H, W = HW
    # 白底
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i,(e,met,_) in enumerate(ellipses):
        (cx,cy),(MA,ma),ang = e
        # 椭圆用黑色
        cv2.ellipse(canvas, (int(cx),int(cy)),
                    (max(1,int(MA/2)), max(1,int(ma/2))),
                    float(ang), 0, 360,
                    (0, 0, 0), DRAW_THICK, lineType=cv2.LINE_AA)
        # 中心点也画成黑色小点
        cv2.circle(canvas, (int(cx),int(cy)), 2, (0, 0, 0), -1)
    return canvas

def pad_to_fixed(results, n=FORCE_OUTPUT_COUNT):
    if not results: return []
    records = []
    for (e, met, fit_pts) in results:
        rec = {
            "ellipse": e,
            "metrics": met,
            "fit_pts": int(fit_pts),
            "fit_score": compute_fit_score(met),
            "quality":   grade_quality(met),
        }
        records.append(rec)
    records.sort(key=lambda d: d["fit_score"], reverse=True)
    records = records[:MAX_OUT_ELLIPSES]
    while len(records) < n:
        records.append(records[0])
    return records[:n]

# ======== 主程序 ========
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    KEEP_DIR.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(EXTS)]
    files.sort()
    print(f"待处理 {len(files)} 张图…")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image","roi_id","method","cx","cy","MA","ma","angle","fit_score","quality","fit_pts","roi_x1","roi_y1"])

        ok, fail = 0, 0
        for i,name in enumerate(files,1):
            path = str(IN_DIR / name)
            bgr  = cv2.imread(path)
            if bgr is None:
                print(f"[{i}/{len(files)}] {name}: 读取失败")
                continue

            results = process_one_image(bgr)
            out_stem = make_output_basename(name)

            if results:
                picked5 = pad_to_fixed(results, n=FORCE_OUTPUT_COUNT)
                draw_items = [(r["ellipse"], r["metrics"], r["fit_pts"]) for r in picked5]
                out = draw_overlay(bgr.shape[:2], draw_items)
                out_path = safe_out_path(OUT_DIR, out_stem, OUT_EXT)
                cv2.imwrite(str(out_path), out)

                msg = " | ".join([
                    f"A={r['metrics']['A']:.1f},sup={r['metrics']['support']:.2f},cov={r['metrics']['ang_cover']:.0f}°,score={r['fit_score']:.3f},Q={r['quality']}"
                    for r in picked5
                ])
                print(f"[{i}/{len(files)}] {name} -> {out_path.name}: 输出{FORCE_OUTPUT_COUNT}个（含重复高质） -> {msg}")
                ok += 1

                for r in picked5:
                    (cx,cy),(MA,ma),ang = r["ellipse"]
                    writer.writerow([
                        out_path.name,
                        0,
                        "fourth_stage_center_constrained+arc_grow+bisector_center+robust_refit",
                        f"{float(cx):.3f}", f"{float(cy):.3f}",
                        f"{float(MA):.3f}", f"{float(ma):.3f}",
                        f"{float(ang):.3f}",
                        f"{float(r['fit_score']):.6f}",
                        r["quality"],
                        int(r["fit_pts"]),
                        0, 0
                    ])
            else:
                save_path = safe_out_path(KEEP_DIR, out_stem, OUT_EXT)
                cv2.imwrite(str(save_path), bgr)
                print(f"[{i}/{len(files)}] {name} -> {save_path.name}: 仍无可靠椭圆 → 已归档到 still_noellipse")
                fail += 1

    print(f"\nCSV 已生成：{CSV_PATH}")
    print(f"完成：成功 {ok}，仍无椭圆 {fail}（已归档到 {KEEP_DIR}）。")

if __name__ == "__main__":
    main()
