# -*- coding: utf-8 -*-
"""
从原图中用 YOLO 定位 ROI，在 ROI 内做颜色梯度边缘 → 细化 → 拓扑净化，
并行进行：
  A) 骨架组件拟合 + 打分
  B) 等厚轮廓(contour)拟合 + 打分
若两路都空，回退到 FRST(中心投票+半径峰)。
最终在原图尺寸的黑底图上，仅渲染“接近椭圆”的轮廓（绿色），可选叠加黄椭圆。
"""

import os, cv2, math, numpy as np
from pathlib import Path

# ===================== 路径与基本设置 =====================
IMAGE_FOLDER = Path(r"F:\YOLO\DAV2\image")
OUTPUT_DIR   = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3")
MODEL_PATH   = Path(r"F:\YOLO\yolov11(GAM)\runs\train_stable\exp_yolov11m_safe5\weights\best.pt")

CONF_THRESH        = 0.35
BASE_ROI_PAD       = 20
MAX_ROIS_PER_IMAGE = 5
DRAW_FIT_ELLIPSE   = False       # 调试：在输出图上叠加黄椭圆
FORCE_AT_LEAST_ONE = True        # 没有候选时强行拟合最大组件一次（保底）

# ===================== 预处理/边缘提取 =====================
CLAHE_CLIP  = 2.5
BILATERAL_D  = 7
BILATERAL_SC = 50
BILATERAL_SS = 50
CLOSE_ITERS  = 2

MIN_COMPONENT_AREA_RATIO = 1.2e-4
MIN_COMPONENT_AREA_ABS   = 60
MIN_BBOX_SHORT_SIDE      = 14

# ===================== 骨架净化（ spur / 小闭环 ） =====================
SPUR_LEN_MIN   = 10
MIN_CYCLE_PIX  = 80
MIN_PATH_PIX   = 50

# ===================== 椭圆一致性与几何约束（初始） =====================
MIN_AXIS_PIX       = 12
MIN_REL_RAD        = 0.08
MAX_DIAM_RATIO_ROI = 0.98
AXIS_RATIO_MIN     = 0.40
AXIS_RATIO_MAX     = 1.45
CENTER_IN_ROI_ONLY = True

SUPPORT_DIST       = 0.30
SUPPORT_RATIO_MIN  = 0.45
COVERAGE_RATIO_MIN = 0.18
RADIAL_STD_MAX_REL = 0.35

# 自适应放宽阶梯（逐级放宽，直到出现候选）
RELAX_STAGES = [
    dict(min_rel_rad=MIN_REL_RAD,  ratio_min=AXIS_RATIO_MIN, ratio_max=AXIS_RATIO_MAX,
         support=SUPPORT_RATIO_MIN, coverage=COVERAGE_RATIO_MIN, rstd=RADIAL_STD_MAX_REL),
    dict(min_rel_rad=0.06, ratio_min=0.35, ratio_max=1.55, support=0.40, coverage=0.14, rstd=0.42),
    dict(min_rel_rad=0.05, ratio_min=0.30, ratio_max=1.65, support=0.36, coverage=0.10, rstd=0.50),
]

# 渲染（把稀疏点连成连续线）
RENDER_RES_THR     = 0.40
TANGENT_PERP_THR   = 0.55
DOT_R            = 2
CLOSE_KSIZE      = 5
DILATE_KSIZE     = 3

# 去重/排序
CENTER_TOL    = 10.0
ANGLE_TOL_DEG = 12.0
DELTA_A_MIN   = 8.0
TOP_KEEP_PER_ROI = 10

# ============ FRST 风格回退（中心投票 + 半径峰） ============
VOTE_R_MIN_REL  = 0.10
VOTE_R_MAX_REL  = 0.65
VOTE_STEP_PX    = 2
CENTER_NMS      = 20
CENTER_TOPK     = 5
RADIUS_BAND     = 3.0
RADIUS_MIN_INLIER = 50

# ===================== 实用函数 =====================
def auto_gamma(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray))
    if m < 60:      gamma = 1.4
    elif m < 120:   gamma = 0.8
    else:           gamma = 0.6
    inv = 1.0 / gamma
    table = np.array([((i/255.0)**inv)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, table)

def clahe_bgr(img, clip=CLAHE_CLIP):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    p2, p98 = np.percentile(l, (2, 98))
    l = np.clip((l - p2) * 255.0 / (p98 - p2 + 1e-6), 0, 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([cl,a,b]), cv2.COLOR_LAB2BGR)

def thinning(binary):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)
    # Guo–Hall 回退
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for it in (0, 1):
            marker = np.zeros_like(img)
            for y in range(1, img.shape[0]-1):
                rowm1, row, rowp1 = img[y-1], img[y], img[y+1]
                for x in range(1, img.shape[1]-1):
                    p1 = row[x]
                    if p1 == 0: continue
                    p2,p3,p4 = rowm1[x], rowm1[x+1], row[x+1]
                    p5,p6,p7 = rowp1[x+1], rowp1[x], rowp1[x-1]
                    p8,p9    = row[x-1], rowm1[x-1]
                    nb = p2+p3+p4+p5+p6+p7+p8+p9
                    if nb < 2 or nb > 6: continue
                    A = int((p2==0 and p3==1))+int((p3==0 and p4==1))+int((p4==0 and p5==1))+ \
                        int((p5==0 and p6==1))+int((p6==0 and p7==1))+int((p7==0 and p8==1))+ \
                        int((p8==0 and p9==1))+int((p9==0 and p2==1))
                    if A != 1: continue
                    if it == 0:
                        if p2*p4*p6 != 0: continue
                        if p4*p6*p8 != 0: continue
                    else:
                        if p2*p4*p8 != 0: continue
                        if p2*p6*p8 != 0: continue
                    marker[y,x] = 1
            if marker.any():
                img[marker==1] = 0
                changed = True
    return (img * 255).astype(np.uint8)

def edges_from_roi(bgr_roi):
    H, W = bgr_roi.shape[:2]
    img = auto_gamma(bgr_roi)
    img = clahe_bgr(img)
    img = cv2.bilateralFilter(img, d=BILATERAL_D, sigmaColor=BILATERAL_SC, sigmaSpace=BILATERAL_SS)

    # 颜色梯度 → 二值
    mags = []
    for ch in cv2.split(img):
        ch_b = cv2.GaussianBlur(ch, (5,5), 0)
        gx = cv2.Sobel(ch_b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch_b, cv2.CV_32F, 0, 1, ksize=3)
        mags.append(cv2.magnitude(gx, gy))
    color_grad = np.maximum.reduce(mags)
    color_grad = (255 * color_grad / (np.max(color_grad) + 1e-6)).astype(np.uint8)

    _, th_otsu = cv2.threshold(color_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_adp = cv2.adaptiveThreshold(color_grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, -5)
    edges = cv2.bitwise_or(th_otsu, th_adp)

    # 形态学
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if CLOSE_ITERS > 0:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    edges = cv2.dilate(edges, k, iterations=1)

    # 小域过滤
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    area = H * W
    min_area = max(int(MIN_COMPONENT_AREA_RATIO * area), MIN_COMPONENT_AREA_ABS)
    clean = np.zeros_like(edges)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a >= min_area and min(w, h) >= MIN_BBOX_SHORT_SIDE:
            clean[labels == i] = 255

    skel = thinning(clean)

    # 供 FRST 使用的“灰度”（来自增强后的亮度通道）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return clean, skel, gray

# ========= 拓扑净化/组件提取 =========
NBRS8 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)

def degree_map(skel):
    nb = cv2.filter2D((skel>0).astype(np.uint8), -1, NBRS8)
    return (skel>0).astype(np.uint8) * nb

def prune_spurs(skel, spur_len=SPUR_LEN_MIN):
    sk = skel.copy()
    H,W = sk.shape
    while True:
        deg = degree_map(sk)
        endpoints = np.column_stack(np.where((sk>0) & (deg==1)))
        if len(endpoints) == 0:
            break
        removed_any = False
        for y,x in endpoints:
            if sk[y,x] == 0: 
                continue
            path = [(y,x)]
            yp, xp = y, x
            for _ in range(spur_len):
                found = None
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dy==0 and dx==0: continue
                        yy,xx = yp+dy, xp+dx
                        if 0<=yy<H and 0<=xx<W and sk[yy,xx]>0:
                            if len(path)==1 or (yy,xx)!=(path[-2]):
                                found = (yy,xx)
                                break
                    if found: break
                if not found: break
                path.append(found)
                yp,xp = found
                d = degree_map(sk)
                if d[yp,xp] != 2:
                    break
            if len(path) <= spur_len:
                for (yy,xx) in path: sk[yy,xx] = 0
                removed_any = True
        if not removed_any: break
    return sk

def extract_components(skel_pruned):
    comps = []
    visited = set()
    H,W = skel_pruned.shape
    ys,xs = np.where(skel_pruned>0)
    for y,x in zip(ys,xs):
        if (y,x) in visited: continue
        stack=[(y,x)]; pts=[]
        while stack:
            yy,xx = stack.pop()
            if (yy,xx) in visited: continue
            visited.add((yy,xx))
            pts.append((xx,yy))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==0 and dx==0: continue
                    y2,x2 = yy+dy, xx+dx
                    if 0<=y2<H and 0<=x2<W and skel_pruned[y2,x2]>0 and (y2,x2) not in visited:
                        stack.append((y2,x2))
        if not pts: continue
        pts_np = np.array(pts, np.int32)
        sub = np.zeros_like(skel_pruned); sub[pts_np[:,1], pts_np[:,0]] = 255
        deg = degree_map(sub)
        n_end = int(np.sum((sub>0) & (deg==1)))
        closed = (n_end == 0)
        if closed and len(pts_np) < MIN_CYCLE_PIX: continue
        if (not closed) and len(pts_np) < MIN_PATH_PIX: continue
        x,y,w,h = cv2.boundingRect(pts_np.reshape(-1,1,2))
        if min(w,h) < MIN_BBOX_SHORT_SIDE: continue
        comps.append((pts_np, closed))
    return comps

# ========= 椭圆拟合与打分 =========
def ellipse_fit_metrics(pts):
    if len(pts) < 5: return None
    try:
        e = cv2.fitEllipse(pts.astype(np.float32))
    except: return None
    (cx,cy),(MA,ma),ang = e
    A,B = max(MA,ma)/2.0, min(MA,ma)/2.0
    if A < 1 or B < 1: return None
    th = math.radians(ang)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]], np.float32)
    X = pts.astype(np.float32) - np.array([cx,cy], np.float32)
    Xp = (X @ R)
    res = np.abs((Xp[:,0]**2)/(A**2 + 1e-6) + (Xp[:,1]**2)/(B**2 + 1e-6) - 1.0)
    support = float(np.mean(res < SUPPORT_DIST))
    peri_cnt = float(len(pts))
    peri_ell = math.pi*(3*(A+B) - math.sqrt((3*A+B)*(A+3*B)))
    coverage = float(peri_cnt / max(peri_ell, 1.0))
    r = np.sqrt((Xp[:,0]**2) + (Xp[:,1]**2))
    r_mean = np.mean(r) + 1e-6
    radial_rel_std = float(np.std(r) / r_mean)
    return e, support, coverage, radial_rel_std, res

def keep_by_geometry(e, roi_w, roi_h, roi_rect, min_rel_rad, ratio_min, ratio_max):
    (cx,cy),(MA,ma),ang = e
    A,B = max(MA,ma)/2.0, min(MA,ma)/2.0
    if A < MIN_AXIS_PIX or B < MIN_AXIS_PIX: return False
    if A < min(roi_w, roi_h) * min_rel_rad: return False
    if 2*A > MAX_DIAM_RATIO_ROI*roi_w or 2*B > MAX_DIAM_RATIO_ROI*roi_h: return False
    ratio = B / A
    if not (ratio_min <= ratio <= ratio_max): return False
    if CENTER_IN_ROI_ONLY:
        x1,y1,x2,y2 = roi_rect
        if not (x1 <= cx <= x2 and y1 <= cy <= y2): return False
    return True

def score_ellipse(support, coverage, radial_rel_std, A):
    size_w = min(1.0, A / 40.0)
    smooth_w = max(0.1, 1.0 - radial_rel_std)
    return float(support * coverage * size_w * smooth_w)

def dedup(results):
    if len(results) <= 1: return results
    results.sort(key=lambda r: r["score"], reverse=True)
    keep = []
    for r in results:
        (cx,cy),(MA,ma),ang = r["ellipse"]
        A = max(MA,ma)/2.0
        ok = True
        for rr in keep:
            (x2,y2),(MA2,ma2),ang2 = rr["ellipse"]
            A2 = max(MA2,ma2)/2.0
            dc = math.hypot(cx - x2, cy - y2)
            da = abs(((ang - ang2 + 90) % 180) - 90)
            if dc <= CENTER_TOL and da <= ANGLE_TOL_DEG and abs(A - A2) < DELTA_A_MIN:
                ok = False; break
        if ok: keep.append(r)
    return keep

# ========= FRST 风格回退：用“原始灰度”的梯度 =========
def frst_vote_centers(gray, skel, rmin, rmax, step):
    H,W = gray.shape
    acc = np.zeros((H,W), np.uint32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-6
    ys,xs = np.where(skel>0)
    for y,x in zip(ys,xs):
        vx = gx[y,x] / mag[y,x]
        vy = gy[y,x] / mag[y,x]
        if not np.isfinite(vx) or not np.isfinite(vy): 
            continue
        for sgn in (-1, 1):
            for t in range(rmin, rmax, step):
                cx = int(round(x + sgn * vx * t))
                cy = int(round(y + sgn * vy * t))
                if 0 <= cx < W and 0 <= cy < H:
                    acc[cy, cx] += 1
    return acc

def topk_centers_from_acc(acc, k=5, nms_rad=20):
    acc_copy = acc.astype(np.float32)
    centers=[]
    for _ in range(k):
        idx = np.argmax(acc_copy)
        v = float(acc_copy.flat[idx])
        if v <= 0: break
        y, x = np.unravel_index(idx, acc_copy.shape)
        centers.append((x,y,int(v)))
        cv2.circle(acc_copy, (x,y), nms_rad, 0.0, -1)
    return centers

def radius_peaks(pts_xy, center, rmin, rmax, band, min_inlier):
    cx,cy = center
    vec = pts_xy - np.array([cx,cy], np.float32)
    dist = np.sqrt((vec**2).sum(1))
    hist_bins = np.arange(rmin, rmax+1, 1)
    hist, _ = np.histogram(dist, bins=hist_bins)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] >= hist[i-1] and hist[i] >= hist[i+1] and hist[i] >= min_inlier:
            r = int(hist_bins[i])
            peaks.append((r, hist[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in peaks[:3]]

# ========= 轮廓(contour)候选（新增 B 路） =========
def contour_candidates(clean, x1, y1, roi_w, roi_h, roi_rect, stage):
    cnts, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if c.shape[0] < 30: 
            continue
        # 转为全局坐标
        pts = c.reshape(-1,2).astype(np.float32)
        pts[:,0] += x1
        pts[:,1] += y1

        met = ellipse_fit_metrics(pts)
        if met is None: 
            continue
        e, support, coverage, rstd, _ = met
        if not keep_by_geometry(e, roi_w, roi_h, roi_rect, 
                                stage["min_rel_rad"], stage["ratio_min"], stage["ratio_max"]):
            continue
        if support < stage["support"] or coverage < stage["coverage"] or rstd > stage["rstd"]:
            continue
        A = max(e[1]) / 2.0
        sc = score_ellipse(support, coverage, rstd, A)
        out.append(dict(ellipse=e, score=sc))
    return out

# ===================== YOLO ROI =====================
def detect_rois(model, bgr, conf=CONF_THRESH):
    res = model.predict(bgr, conf=conf, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy is None:
        return []
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    H, W = bgr.shape[:2]
    rois = []
    for (x1,y1,x2,y2) in boxes:
        x1p = max(0, x1 - BASE_ROI_PAD)
        y1p = max(0, y1 - BASE_ROI_PAD)
        x2p = min(W-1, x2 + BASE_ROI_PAD)
        y2p = min(H-1, y2 + BASE_ROI_PAD)
        if x2p - x1p >= 10 and y2p - y1p >= 10:
            rois.append([x1p, y1p, x2p, y2p])
    rois.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return rois[:MAX_ROIS_PER_IMAGE]

# ===================== 主处理：在原图尺寸上渲染 =====================
def process_one(bgr, rois):
    H, W = bgr.shape[:2]
    canvas = np.full((H, W, 3), 255, np.uint8)

    for (x1,y1,x2,y2) in rois:
        roi = bgr[y1:y2, x1:x2]
        clean, skel, roi_gray = edges_from_roi(roi)
        roi_h, roi_w = roi.shape[:2]

        skel_pruned = prune_spurs(skel, SPUR_LEN_MIN)

        # ---------- 逐级放宽，直到拿到候选 ----------
        cand = []
        for stage in RELAX_STAGES:
            # A) 骨架组件拟合
            comps = extract_components(skel_pruned)
            from_skel = []
            for pts_xy, _is_closed in comps:
                pts_global = pts_xy.astype(np.float32).copy()
                pts_global[:,0] += x1
                pts_global[:,1] += y1
                met = ellipse_fit_metrics(pts_global)
                if met is None: 
                    continue
                e, support, coverage, rstd, _ = met
                if not keep_by_geometry(e, roi_w, roi_h, (x1,y1,x2,y2),
                                        stage["min_rel_rad"], stage["ratio_min"], stage["ratio_max"]):
                    continue
                if support < stage["support"] or coverage < stage["coverage"] or rstd > stage["rstd"]:
                    continue
                A = max(e[1]) / 2.0
                sc = score_ellipse(support, coverage, rstd, A)
                from_skel.append(dict(ellipse=e, score=sc))

            # B) 轮廓候选（对闭环更稳）
            from_cnt = contour_candidates(clean, x1, y1, roi_w, roi_h, (x1,y1,x2,y2), stage)

            cand = dedup(from_skel + from_cnt)
            cand = cand[:TOP_KEEP_PER_ROI]
            if len(cand) > 0:
                break  # 本级已足够

        # C) 回退：FRST（若 A+B 都没有）
        if len(cand) == 0:
            rmin = int(max(8, min(roi_h, roi_w) * VOTE_R_MIN_REL))
            rmax = int(max(rmin+8, min(roi_h, roi_w) * VOTE_R_MAX_REL))
            acc = frst_vote_centers(roi_gray, skel_pruned, rmin, rmax, VOTE_STEP_PX)
            centers = topk_centers_from_acc(acc, k=CENTER_TOPK, nms_rad=CENTER_NMS)

            ys, xs = np.where(skel_pruned>0)
            if len(xs) > 0:
                pts_all = np.stack([xs + x1, ys + y1], axis=1).astype(np.float32)
                for (cx,cy,_v) in centers:
                    radii = radius_peaks(pts_all, (cx,cy), rmin, rmax, RADIUS_BAND, RADIUS_MIN_INLIER)
                    for r in radii:
                        vec = pts_all - np.array([cx,cy], np.float32)
                        dist = np.sqrt((vec**2).sum(1))
                        mask = np.abs(dist - r) < RADIUS_BAND
                        inliers = pts_all[mask]
                        if len(inliers) < max(RADIUS_MIN_INLIER, 30): 
                            continue
                        met = ellipse_fit_metrics(inliers)
                        if met is None: 
                            continue
                        e, support, coverage, rstd, _ = met
                        if not keep_by_geometry(e, roi_w, roi_h, (x1,y1,x2,y2),
                                                RELAX_STAGES[-1]["min_rel_rad"],
                                                RELAX_STAGES[-1]["ratio_min"],
                                                RELAX_STAGES[-1]["ratio_max"]):
                            continue
                        A = max(e[1]) / 2.0
                        sc = score_ellipse(max(support, 0.35),
                                           max(coverage, 0.10),
                                           min(rstd, 0.50), A)
                        cand.append(dict(ellipse=e, score=sc))
            cand = dedup(cand)[:TOP_KEEP_PER_ROI]

        # D) 仍无：保底拟合最大组件一次
        if len(cand) == 0 and FORCE_AT_LEAST_ONE:
            comps = extract_components(skel_pruned)
            if len(comps) > 0:
                comps.sort(key=lambda z: len(z[0]), reverse=True)
                pts0 = comps[0][0].astype(np.float32)
                pts0[:,0] += x1; pts0[:,1] += y1
                met = ellipse_fit_metrics(pts0)
                if met is not None:
                    e, support, coverage, rstd, _ = met
                    A = max(e[1])/2.0
                    sc = score_ellipse(support, coverage, rstd, A) * 0.5
                    cand = [dict(ellipse=e, score=sc)]

        # ------- 渲染：用骨架 + 椭圆残差 + 切向一致性，把点连成线 -------
        ys, xs = np.where(skel_pruned > 0)
        if len(xs) > 0 and len(cand) > 0:
            skel_f = cv2.GaussianBlur((skel_pruned > 0).astype(np.float32), (5,5), 0)
            gx = cv2.Sobel(skel_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(skel_f, cv2.CV_32F, 0, 1, ksize=3)
            tx = -gy; ty = gx

            pts = np.stack([xs + x1, ys + y1], axis=1).astype(np.float32)
            keep_mask = np.zeros(len(pts), dtype=bool)

            for r in cand:
                (cx,cy),(MA,ma),ang = r["ellipse"]
                A,B = max(MA,ma)/2.0, min(MA,ma)/2.0
                th = math.radians(ang)
                Rm = np.array([[math.cos(th), -math.sin(th)],
                               [math.sin(th),  math.cos(th)]], np.float32)
                X  = pts - np.array([cx,cy], np.float32)
                Xp = (X @ Rm)
                res = np.abs((Xp[:,0]**2)/(A**2 + 1e-6) + (Xp[:,1]**2)/(B**2 + 1e-6) - 1.0)

                rx = pts[:,0] - cx;  ry = pts[:,1] - cy
                rnorm = np.sqrt(rx*rx + ry*ry) + 1e-6
                txx = tx[ys, xs];    tyy = ty[ys, xs]
                tnorm = np.sqrt(txx*txx + tyy*tyy) + 1e-6
                dotv = (rx*txx + ry*tyy) / (rnorm * tnorm)
                tang_ok = np.abs(dotv) < TANGENT_PERP_THR

                keep_mask |= (res < RENDER_RES_THR) & tang_ok

            kept = pts[keep_mask].astype(np.int32)

            roi_canvas = np.zeros((roi_h, roi_w), np.uint8)
            if len(kept) > 0:
                for (xx,yy) in kept:
                    xxr = int(xx - x1); yyr = int(yy - y1)
                    if 0 <= xxr < roi_w and 0 <= yyr < roi_h:
                        cv2.circle(roi_canvas, (xxr, yyr), DOT_R, 255, -1)

                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
                k_dil   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KSIZE, DILATE_KSIZE))
                roi_canvas = cv2.morphologyEx(roi_canvas, cv2.MORPH_CLOSE, k_close, iterations=1)
                roi_canvas = cv2.dilate(roi_canvas, k_dil, iterations=1)

            roi_region = canvas[y1:y2, x1:x2]
            mask = roi_canvas > 0
            roi_region[mask] = (0, 0, 0)
            canvas[y1:y2, x1:x2] = roi_region

            if DRAW_FIT_ELLIPSE:
                for r in cand:
                    (cx,cy),(MA,ma),ang = r["ellipse"]
                    cv2.ellipse(canvas, (int(cx),int(cy)), (int(MA/2),int(ma/2)),
                                float(ang), 0, 360, (0,255,255), 2)

    return canvas

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
    imgs.sort()
    if not imgs:
        print("未在目录中找到图像：", IMAGE_FOLDER); return

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    for i, name in enumerate(imgs, 1):
        p = str(IMAGE_FOLDER / name)
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[{i}/{len(imgs)}] {name} 读取失败"); continue

        rois = detect_rois(model, bgr, conf=CONF_THRESH)
        if not rois:
            H,W = bgr.shape[:2]
            rois = [[0, 0, W-1, H-1]]

        out = process_one(bgr, rois)
        out_path = OUTPUT_DIR / f"{Path(name).stem}_contour.png"
        cv2.imwrite(str(out_path), out)
        print(f"[{i}/{len(imgs)}] {name} 完成，ROI数={len(rois)} → {out_path.name}")

if __name__ == "__main__":
    main()
