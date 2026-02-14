# -*- coding: utf-8 -*-
import os, math
from pathlib import Path
import numpy as np
import cv2

# ===================== 路径 =====================
INPUT_DIR  = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3")   # 第一轮后的 ROI 轮廓图
OUTPUT_DIR = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3_refined_v2")

# ===================== 参数 =====================
# 绿色阈值（优先 HSV，BGR 备选，最后用“暗像素”（黑线）兜底）
HSV_LO = (35, 60, 60)
HSV_HI = (85, 255, 255)
BGR_LO = (0, 100, 0)
BGR_HI = (120, 255, 120)

USE_XIMGPROC = True  # 细化优先用 ximgproc

MIN_CC_AREA_FOR_CENTER = 40
CENTER_MERGE_RAD = 18

TOP_RADII = 6
R_MIN_GAP = 8
BASE_BAND = 8
BAND_FRAC = 0.03
R_MAX_FRAC = 0.8

MIN_CC_AREA_KEEP = 60
MIN_ARC_DEG      = 130
ECC_MIN, ECC_MAX = 0.45, 1.00
CURV_STD_MAX     = 1.00

MIN_PIXELS_ACCEPT = 250
RESCUE_TOPK = 8
RELAX_STAGES = [
    dict(),  # 基线
    dict(MIN_ARC_DEG=90,  MIN_CC_AREA_KEEP=28, ECC_MIN=0.30, CURV_STD_MAX=1.15),
    dict(MIN_ARC_DEG=70,  MIN_CC_AREA_KEEP=20, ECC_MIN=0.25, CURV_STD_MAX=1.30),
    dict(MIN_ARC_DEG=50,  MIN_CC_AREA_KEEP=12, ECC_MIN=0.20, CURV_STD_MAX=1.50),
]

# ===================== 工具 =====================
def load_mask_from_green(img):
    """
    尽可能稳地把轮廓转成二值：
    - 优先：黑底绿线 → 用 HSV/BGR 提绿；
    - 如果没有绿色：默认白底黑线 → 用灰度 + 反阈值取“黑线”。
    """
    # 单通道输入（很少用到，保持原逻辑）
    if img.ndim == 2 or img.shape[2] == 1:
        return (img > 0).astype(np.uint8) * 255

    # 1) 先尝试按“绿色”提取（兼容老的黑底绿线结果）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv,
                           np.array(HSV_LO, np.uint8),
                           np.array(HSV_HI, np.uint8))
    mask_bgr = cv2.inRange(img,
                           np.array(BGR_LO, np.uint8),
                           np.array(BGR_HI, np.uint8))
    mask = cv2.bitwise_or(mask_hsv, mask_bgr)

    # 2) 若没有绿色，再按“暗像素”提取黑线（白底黑线）
    if mask.sum() == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 阈值可以根据你图像亮度情况微调，比如 200/220 等
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 3) 轻微形态学闭运算，去小缝隙
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def thin(binary):
    if USE_XIMGPROC and hasattr(cv2, "ximgproc"):
        try:
            return cv2.ximgproc.thinning(binary)
        except Exception:
            pass
    k = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    return cv2.erode(binary, k, iterations=1)

def fit_ellipse_safe(pts):
    if len(pts) < 5: return None
    try:
        return cv2.fitEllipse(pts.astype(np.float32))
    except: return None

def curvature_std(pts):
    if len(pts) < 8: return 1e9
    pts = pts[np.argsort(pts[:,0])]
    ds  = np.diff(pts, axis=0)
    ang = np.arctan2(ds[:,1], ds[:,0])
    dang = np.diff(ang, prepend=ang[:1])
    dang = (dang + np.pi)%(2*np.pi) - np.pi
    return float(np.std(dang))

def angular_coverage(pts, cx, cy):
    ang = np.degrees(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))
    ang = np.sort((ang+360)%360)
    dif = np.diff(np.r_[ang, ang[0]+360.0])
    return 360.0 - np.max(dif)

def cluster_centers(centers, merge_rad=CENTER_MERGE_RAD):
    used=[False]*len(centers); out=[]
    for i,(x,y) in enumerate(centers):
        if used[i]: continue
        group=[(x,y)]; used[i]=True
        for j,(xx,yy) in enumerate(centers[i+1:], start=i+1):
            if used[j]: continue
            if (x-xx)**2 + (y-yy)**2 <= merge_rad*merge_rad:
                group.append((xx,yy)); used[j]=True
        gx,gy = np.mean(group, axis=0)
        out.append((float(gx), float(gy), len(group)))
    out.sort(key=lambda t:-t[2])
    return out

def choose_center_hybrid(bin_img):
    """先用各连通域拟合中心做聚类，失败则用距离变换峰值兜底"""
    h,w = bin_img.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    centers=[]
    for i in range(1,num):
        if stats[i, cv2.CC_STAT_AREA] < MIN_CC_AREA_FOR_CENTER:
            continue
        ys,xs = np.where(labels==i)
        pts = np.stack([xs,ys],1)
        e = fit_ellipse_safe(pts)
        if e is None: continue
        (cx,cy),_,_ = e
        if 0<=cx<w and 0<=cy<h:
            centers.append((cx,cy))
    if centers:
        c = cluster_centers(centers)[0]
        return (c[0], c[1])

    inv = cv2.bitwise_not((bin_img>0).astype(np.uint8)*255)
    dt  = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    y,x = np.unravel_index(np.argmax(dt), dt.shape)
    return (float(x), float(y))

def radial_hist(pts, cx, cy, rmax):
    rr = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    rr_in = rr[(rr>=0) & (rr<=max(1, rmax))]
    if rr_in.size==0:
        return None, rr  # 注意：这里返回全部 rr 作为兜底
    rr_i = rr_in.astype(np.int32)
    hist = np.bincount(rr_i, minlength=int(rmax)+1).astype(np.float32)
    k = np.array([1,2,3,2,1], np.float32); k/=k.sum()
    hist = np.convolve(hist, k, mode='same')
    return hist, rr_in

def pick_peaks(hist, top=5, min_gap=8):
    if hist is None: return []
    h = hist.copy()
    peaks=[]
    for _ in range(top*3):
        i = int(np.argmax(h)); v=float(h[i])
        if v<=0: break
        peaks.append((i,v))
        l=max(0, i-min_gap); r=min(len(h)-1, i+min_gap)
        h[l:r+1]=0
    peaks.sort(key=lambda x:-x[1])
    return [p[0] for p in peaks[:top]]

def build_belt_mask(shape, cx, cy, radii, base_band=BASE_BAND, frac=BAND_FRAC):
    h,w = shape
    Y,X = np.mgrid[0:h, 0:w]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    belt = np.zeros((h,w), np.uint8)
    for r in radii:
        band = max(base_band, int(frac*r))
        belt |= (np.abs(R-r) <= band).astype(np.uint8)
    return (belt*255).astype(np.uint8)

# ===================== 核心：二次筛选 =====================
def refine_once(bin0, stage):
    h,w = bin0.shape
    arc_min   = stage.get("MIN_ARC_DEG", MIN_ARC_DEG)
    area_min  = stage.get("MIN_CC_AREA_KEEP", MIN_CC_AREA_KEEP)
    ecc_min   = stage.get("ECC_MIN", ECC_MIN)
    ecc_max   = stage.get("ECC_MAX", ECC_MAX)
    kstd_max  = stage.get("CURV_STD_MAX", CURV_STD_MAX)

    # 细化 + 轻闭运算
    edges = thin((bin0>0).astype(np.uint8)*255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

    # 边点
    ys,xs = np.where(edges>0)
    if len(xs)==0:
        return np.zeros_like(edges)

    pts = np.stack([xs,ys],1)
    cx,cy = choose_center_hybrid(edges)

    rmax = int(min(h,w)*R_MAX_FRAC)

    # —— 稳健半径推断 —— #
    hist, rr_in = radial_hist(pts, cx, cy, rmax)  # rr_in 可能为空
    rr_all = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)

    radii = pick_peaks(hist, top=TOP_RADII, min_gap=R_MIN_GAP)
    if not radii:
        # 峰值没有：用分位数稳健给 3~4 个候选半径
        qs = np.quantile(rr_all, [0.35, 0.55, 0.72, 0.85])
        radii = [int(q) for q in qs]

    # 半径清洗：去重、限幅、去太小/太大、间隔约束
    rmin_allow = 5
    radii = [int(max(rmin_allow, min(r, rmax))) for r in radii]
    radii = sorted(list(dict.fromkeys(radii)))
    merged = []
    for r in radii:
        if not merged or abs(r - merged[-1]) >= max(R_MIN_GAP//2, 4):
            merged.append(r)
    radii = merged or [max(rmin_allow, int(min(rr_all)))]
    # —— 稳健半径推断结束 —— #

    belt = build_belt_mask((h,w), cx, cy, radii)
    kept = cv2.bitwise_and(edges, belt)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(kept, 8)
    out = np.zeros_like(kept)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < area_min: 
            continue
        ys_i, xs_i = np.where(labels==i)
        comp = np.stack([xs_i,ys_i],1).astype(np.float32)

        if angular_coverage(comp, cx, cy) < arc_min:
            continue

        e = fit_ellipse_safe(comp)
        if e is not None:
            (ecx,ecy),(MA,ma),_ = e
            a,b = max(MA,ma)/2.0, min(MA,ma)/2.0
            if a<1e-3 or b<1e-3: 
                continue
            ratio = b/a
            if not (ecc_min <= ratio <= ecc_max):
                continue

        if curvature_std(comp) > kstd_max:
            continue

        out[labels==i] = 255

    # 兜底1：为空则选与任意半径带相交的面积前 K 大
    if np.count_nonzero(out)==0 and num>1:
        inter_areas = []
        for i in range(1,num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < max(8, area_min//2): 
                continue
            ys_i, xs_i = np.where(labels==i)
            comp_mask = np.zeros_like(kept); comp_mask[ys_i, xs_i] = 255
            inter = cv2.countNonZero(cv2.bitwise_and(comp_mask, belt))
            if inter>0:
                inter_areas.append((i, inter))
        inter_areas.sort(key=lambda t:-t[1])
        for i,_ in inter_areas[:RESCUE_TOPK]:
            out[labels==i]=255

    # 兜底2：仍为空则用 kept
    if np.count_nonzero(out)==0:
        out = kept.copy()

    return out

def refine_with_relax(bin0):
    for si, st in enumerate(RELAX_STAGES):
        mask = refine_once(bin0, st)
        pix = int(np.count_nonzero(mask))
        if pix >= MIN_PIXELS_ACCEPT:
            return mask, si, pix
    mask = refine_once(bin0, {**RELAX_STAGES[-1], "MIN_ARC_DEG":0, "MIN_CC_AREA_KEEP":1})
    return mask, len(RELAX_STAGES)-1, int(np.count_nonzero(mask))

# ===================== 主流程 =====================
def save_mask(mask, out_path):
    """
    输出为：白底 + 黑线
    """
    h, w = mask.shape[:2]
    canvas = np.full((h, w, 3), 255, np.uint8)   # 白底
    canvas[mask > 0] = (0, 0, 0)                 # 线条画成黑色
    cv2.imwrite(str(out_path), canvas)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]
    files.sort()
    for i,name in enumerate(files,1):
        src = cv2.imread(str(INPUT_DIR/name), cv2.IMREAD_COLOR)
        if src is None:
            print(f"[{i}/{len(files)}] {name}: 读图失败")
            continue
        bin0 = load_mask_from_green(src)
        mask, stage_id, pix = refine_with_relax(bin0)
        out_path = OUTPUT_DIR / f"{Path(name).stem}_refined.png"
        save_mask(mask, out_path)
        print(f"[{i}/{len(files)}] {name} → stage{stage_id}, pixels={pix} → {out_path}")

if __name__ == "__main__":
    main()
