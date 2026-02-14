# -*- coding: utf-8 -*-
import os, math, re, csv, copy, random
from pathlib import Path
import numpy as np
import cv2

# ============ 路径 ============
IN_DIR  = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3_refined_v2")
OUT_DIR = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3_refined_v3_healed")
NO_ELLIPSE_DIR = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3_refined_v2_noellipse")
CSV_PATH = OUT_DIR / "ellipses_third_stage.csv"

OVERLAY_ON_INPUT = False

# ============ 输出命名规则 ============
NAME_MODE = "auto"
OUT_EXT   = ".png"

# ============ 通用参数 ============
MAX_ELLIPSES = 5
DRAW_THICK   = 2

STRICT = dict(
    min_major_rel   = 0.12,
    support_min     = 0.55,
    ang_cover_min   = 260.0,
    max_gap_deg     = 90,
    center_eps      = 18.0
)
RELAX1 = dict(
    min_major_rel   = 0.10,
    support_min     = 0.30,
    ang_cover_min   = 150.0,
    max_gap_deg     = 150.0,
    center_eps      = 22.0
)
RELAX2 = dict(
    min_major_rel   = 0.08,
    support_min     = 0.40,
    ang_cover_min   = 150.0,
    max_gap_deg     = 170.0,
    center_eps      = 26.0
)
EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

# ============ 弧段生长参数 ============
GROW_D_FRAC = 0.03
GROW_D_MAX_PX = 10
GROW_THETA_ALIGN_DEG = 25.0
GROW_MAX_BRIDGES = 64
GROW_ACCEPT_FITSCORE_DELTA = 0.05
GROW_ACCEPT_COVER_INC_DEG  = 15.0
GROW_ALLOW_MAXGAP_WORSEN   = 10.0

# ============ 稳中心/鲁棒重拟合参数 ============
BISECT_N_PAIRS = 200          # 弦对数量（随机）
BISECT_MIN_ARCIDX_GAP = 8     # 同一弧段采样点下标至少相隔
CENTER_USE_MEDIAN = True      # 用中值聚合（稳健）
RESID_TUKEY_C = 0.25          # Tukey 截断常数（|rho-1| 阈值）
ANG_BINS = 36                 # 角度分箱数
MAX_PER_BIN = 12              # 每箱最多样本

# ============ 工具函数 ============
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
    if num is None: return s_low
    if NAME_MODE == "spout": return f"spout_{num}"
    if NAME_MODE == "num":   return f"{num}"
    return f"spout_{num}" if "spout" in s_low else f"{num}"

def to_binary_from_green(bgr):
    """
    统一把轮廓转成二值：
    - 若图像中存在明显绿色：按 HSV 提取绿色（兼容黑底绿线版本）；
    - 否则认为是白底黑线：用灰度+反阈值提取“黑线”。
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
        # 2) 没有绿色：默认白底黑线 → 取“暗像素”为前景
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 阈值 200 可按需要微调（越小越“宽松”）
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 3) 轻微闭运算，去小孔
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def thinning(binary):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    img = binary.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = (cv2.countNonZero(img) == 0)
    return skel

def fit_ellipse(points):
    if points.shape[0] < 5: return None
    try:
        return cv2.fitEllipse(points.astype(np.float32))
    except:
        return None

def ellipse_metrics(e, pts, wh):
    (cx,cy),(MA,ma),ang = e
    A = max(MA,ma)/2.0
    B = min(MA,ma)/2.0
    if A < 1 or B < 1: return None
    th = math.radians(ang)
    c,s = math.cos(th), math.sin(th)
    R = np.array([[c, -s],[s, c]], np.float32)
    X = pts.astype(np.float32) - np.array([cx,cy], np.float32)
    Xp = X @ R
    rho = np.sqrt((Xp[:,0]/(A+1e-6))**2 + (Xp[:,1]/(B+1e-6))**2)
    band = np.abs(rho - 1.0)
    support = float(np.mean(band < 0.22))
    t = np.arctan2(Xp[:,1]/(B+1e-6), Xp[:,0]/(A+1e-6))
    t = (t + 2*np.pi) % (2*np.pi)
    t = np.sort(t)
    if t.size >= 1:
        gaps = np.diff(t)
        last_gap = (t[0] + 2*np.pi) - t[-1]
        gaps = np.concatenate([gaps, [last_gap]])
        max_gap = float(np.max(gaps)) * 180.0 / math.pi
        ang_cover = 360.0 - max_gap
    else:
        ang_cover, max_gap = 0.0, 360.0
    perim = math.pi*(3*(A+B) - math.sqrt((3*A+B)*(A+3*B)))
    return dict(A=A, B=B, support=support, ang_cover=ang_cover, max_gap=max_gap, perim=perim)

# ---------- 弧段生长（小缺口桥接+细化） ----------
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
    if endpoints.shape[0] < 2: return skel_bin
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
            if dist < 2 or dist > d_max: continue
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

# ---------- 候选生成/挑选 ----------
def pts_sets_from_skel(skel_bin):
    contours, _ = cv2.findContours(skel_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts_sets = []
    for c in contours:
        if c is None or len(c) < 12: continue
        pts = c.reshape(-1, 2)
        pts_sets.append(pts)
    return pts_sets

# ---------- 稳中心（弦垂直平分线交点投票） ----------
def bisector_center_vote(pts, n_pairs=BISECT_N_PAIRS, min_gap=BISECT_MIN_ARCIDX_GAP):
    N = pts.shape[0]
    if N < min_gap*2 + 2:
        return np.mean(pts, axis=0).tolist()
    centers = []
    for _ in range(n_pairs):
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if abs(i-j) < min_gap: continue
        p1 = pts[i].astype(np.float64); p2 = pts[j].astype(np.float64)
        m1 = 0.5*(p1+p2)
        d1 = p2 - p1
        n1 = np.array([-d1[1], d1[0]], dtype=np.float64)
        k = random.randint(0, N-1)
        l = random.randint(0, N-1)
        if abs(k-l) < min_gap or len({i,j,k,l})<4: continue
        p3 = pts[k].astype(np.float64); p4 = pts[l].astype(np.float64)
        m2 = 0.5*(p3+p4)
        d2 = p4 - p3
        n2 = np.array([-d2[1], d2[0]], dtype=np.float64)
        M = np.array([n1, -n2]).T  # m1 + t n1 = m2 + s n2
        b = (m2 - m1)
        det = np.linalg.det(M)
        if abs(det) < 1e-8: continue
        ts = np.linalg.solve(M, b)
        cxy = m1 + ts[0]*n1
        centers.append(cxy)
    if not centers:
        return np.mean(pts, axis=0).tolist()
    centers = np.array(centers)
    if CENTER_USE_MEDIAN:
        c = np.median(centers, axis=0)
    else:
        c = np.mean(centers, axis=0)
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

# ---------- 聚类（支持“稳中心”） ----------
def cluster_same_center(cands, center_eps):
    if not cands: return []
    remain = set(range(len(cands)))
    clusters = []
    while remain:
        idx = max(remain, key=lambda i: cands[i]['metrics']['A'])
        cx,cy = cands[idx].get('cluster_center', cands[idx]['ellipse'][0])
        members = []
        for j in list(remain):
            cx2,cy2 = cands[j].get('cluster_center', cands[j]['ellipse'][0])
            if math.hypot(cx-cx2, cy-cy2) <= center_eps:
                members.append(j)
        if not members: members = [idx]
        for j in members:
            if j in remain: remain.remove(j)
        clusters.append(members)
    return clusters

def pick_cluster_and_rank(cands, center_eps):
    if not cands: return []
    clusters = cluster_same_center(cands, center_eps)
    if not clusters: return []
    best = None; best_score = -1.0
    for cl in clusters:
        sumR = sum([cands[i]['metrics']['A'] for i in cl])
        mean_sup = float(np.mean([cands[i]['metrics']['support'] for i in cl]))
        sc = sumR * (0.5 + 0.5*mean_sup)
        if sc > best_score:
            best_score = sc; best = cl
    if best is None: return []
    sel = [cands[i] for i in best]
    sel.sort(key=lambda d: d['metrics']['A'], reverse=True)
    return sel

# ---------- 候选生成（加入：稳中心 + 鲁棒重拟合） ----------
def pick_from_pts_sets(pts_sets, cfg, W, H):
    cands = []
    for pts in pts_sets:
        e0 = fit_ellipse(pts)
        if e0 is None: continue
        # 1) 稳中心（弦平分线交点投票）
        cx_ref, cy_ref = bisector_center_vote(pts)
        # 2) 残差 -> 内点 -> 角均衡子采样
        resid, Xp, A0, B0 = residuals_to_ellipse(e0, pts)
        mask = tukey_inlier_mask(resid, c=RESID_TUKEY_C)
        sel = angular_balanced_subset(Xp, A0, B0, mask, per_bin=MAX_PER_BIN, bins=ANG_BINS)
        if sel is not None and sel.size >= 12:
            e1 = fit_ellipse(pts[sel])
            e_use = e1 if e1 is not None else e0
        else:
            e_use = e0
        met = ellipse_metrics(e_use, pts, (W, H))
        if met is None: continue
        min_major_pix = max(32.0, min(W, H) * cfg['min_major_rel'])
        if met['A'] < min_major_pix:          continue
        if met['support']   < cfg['support_min']:   continue
        if met['ang_cover'] < cfg['ang_cover_min']: continue
        if met['max_gap']   > cfg['max_gap_deg']:   continue
        cand = dict(ellipse=e_use, metrics=met, pts=pts)
        cand['cluster_center'] = (cx_ref, cy_ref)  # 仅用于同心聚类
        cands.append(cand)
    return pick_cluster_and_rank(cands, cfg['center_eps'])

# === 评分/分级 ===
def compute_fit_score(met):
    size_w = min(1.0, met['A'] / 40.0)
    cover = max(0.0, min(1.0, met['ang_cover'] / 360.0))
    gap_w = max(0.0, 1.0 - met['max_gap'] / 360.0)
    return float(met['support'] * cover * gap_w * size_w)

def grade_quality(met):
    if met['support'] >= 0.60 and met['ang_cover'] >= 280 and met['max_gap'] <= 80:
        return "A"
    if met['support'] >= 0.45 and met['ang_cover'] >= 200:
        return "B"
    return "C"

def best_fit_score(picked):
    if not picked: return -1.0
    return max([compute_fit_score(p['metrics']) for p in picked])

def is_growth_better(p_grow, p_base):
    if p_base and not p_grow: return False
    if p_grow and not p_base: return True
    if not p_grow and not p_base: return False
    m_g, m_b = p_grow[0]['metrics'], p_base[0]['metrics']
    fs_g = compute_fit_score(m_g)
    fs_b = compute_fit_score(m_b)
    cover_inc = (m_g['ang_cover'] - m_b['ang_cover'])
    gap_worsen = (m_g['max_gap'] - m_b['max_gap'])
    return (fs_g - fs_b >= GROW_ACCEPT_FITSCORE_DELTA) or \
           (cover_inc >= GROW_ACCEPT_COVER_INC_DEG and gap_worsen <= GROW_ALLOW_MAXGAP_WORSEN)

# ---------- 主筛选 ----------
def process_one(img_bgr, strict_cfg, relax_cfgs):
    H, W = img_bgr.shape[:2]
    bin0 = to_binary_from_green(img_bgr)
    if bin0 is None: return None, []
    skel = thinning(bin0)
    # 弧段生长（小缺口桥接）+ 验收/回退
    skel_grown = grow_arcs_angle_aware(skel)

    pts_sets_base  = pts_sets_from_skel(skel)
    pts_sets_grown = pts_sets_from_skel(skel_grown)

    stages = [strict_cfg] + list(relax_cfgs)
    for cfg in stages:
        picked_base = pick_from_pts_sets(pts_sets_base,  cfg, W, H)
        picked_grow = pick_from_pts_sets(pts_sets_grown, cfg, W, H)
        if picked_grow and (not picked_base or is_growth_better(picked_grow, picked_base)):
            sel = picked_grow[:MAX_ELLIPSES]
            return (skel_grown if OVERLAY_ON_INPUT else None), sel
        if picked_base:
            sel = picked_base[:MAX_ELLIPSES]
            return (skel if OVERLAY_ON_INPUT else None), sel
    return (skel if OVERLAY_ON_INPUT else None), []

def draw_result(base_canvas, size_hw, picked):
    """
    输出图像：白底 + 黑色椭圆。
    """
    H, W = size_hw
    if base_canvas is None:
        # 白底
        canvas = np.full((H, W, 3), 255, np.uint8)
    else:
        # 叠加模式：用骨架灰度作为背景（可按需要改成 255-base_canvas 做反色）
        canvas = cv2.cvtColor(base_canvas, cv2.COLOR_GRAY2BGR)

    for i, c in enumerate(picked):
        (cx,cy),(MA,ma),ang = c['ellipse']
        cv2.ellipse(canvas, (int(cx),int(cy)),
                    (max(1, int(MA/2)), max(1, int(ma/2))),
                    float(ang), 0, 360,
                    (0, 0, 0),  # 黑色线
                    DRAW_THICK, lineType=cv2.LINE_AA)
    return canvas

def pad_to_five(picked):
    if not picked: return []
    scored = []
    for c in picked:
        met = c['metrics']
        cc = dict(**c)
        cc['fit_score'] = compute_fit_score(met)
        cc['quality'] = grade_quality(met)
        scored.append(cc)
    scored.sort(key=lambda d: d['fit_score'], reverse=True)
    scored = scored[:MAX_ELLIPSES]
    while len(scored) < MAX_ELLIPSES:
        scored.append(copy.deepcopy(scored[0]))
    return scored

# ============ 主流程 ============
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(NO_ELLIPSE_DIR)
    files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(EXTS)]
    files.sort()
    print(f"待处理 {len(files)} 张图…")

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image","roi_id","method","cx","cy","MA","ma","angle","fit_score","quality","fit_pts","roi_x1","roi_y1"])

        for i, name in enumerate(files, 1):
            path = str(IN_DIR / name)
            bgr  = cv2.imread(path)
            if bgr is None:
                print(f"[{i}/{len(files)}] {name}: 读取失败")
                continue

            base, picked = process_one(bgr, STRICT, [RELAX1, RELAX2])
            out_stem = make_output_basename(name)

            if picked:
                picked5 = pad_to_five(picked)
                out = draw_result(base, bgr.shape[:2], picked5)
                out_path = safe_out_path(OUT_DIR, out_stem, OUT_EXT)
                cv2.imwrite(str(out_path), out)

                msg = " | ".join([
                    f"A={p['metrics']['A']:.1f},sup={p['metrics']['support']:.2f},cov={p['metrics']['ang_cover']:.0f}°,score={p['fit_score']:.3f},Q={p['quality']}"
                    for p in picked5
                ])
                print(f"[{i}/{len(files)}] {name} -> {out_path.name}: 输出5个椭圆（含重复高质） -> {msg}")

                for p in picked5:
                    (cx,cy),(MA,ma),ang = p['ellipse']
                    fit_pts = int(len(p['pts'])) if 'pts' in p and p['pts'] is not None else 0
                    writer.writerow([
                        out_path.name,
                        0,
                        "third_stage_fitEllipse_cluster+arc_grow+bisector_center+robust_refit",
                        float(cx), float(cy),
                        float(MA), float(ma),
                        float(ang),
                        float(p['fit_score']),
                        p['quality'],
                        fit_pts,
                        0, 0
                    ])
            else:
                save_path = safe_out_path(NO_ELLIPSE_DIR, out_stem, OUT_EXT)
                cv2.imwrite(str(save_path), bgr)
                print(f"[{i}/{len(files)}] {name} -> {save_path.name}: 无椭圆，已移至 noellipse 目录")

    print(f"\nCSV 已生成：{CSV_PATH}")

if __name__ == "__main__":
    main()
