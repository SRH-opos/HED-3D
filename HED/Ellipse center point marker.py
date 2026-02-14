import os
import math
import csv
import cv2
import numpy as np


TARGET_RATIO = 0.8          
SAMPLE_POINTS = 180           
PERIM_THICKNESS = 4           
WEIGHT_EDGE = 0.7             
WEIGHT_CENTER = 0.6         
MIN_AREA_RATIO = 0.55        
MAX_AREA_RATIO = 0.95         
VERBOSE = False               


def clahe_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def edges_for_scoring(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.Canny(blur, 30, 150)

def ellipse_area_from_params(MA, ma):
    return math.pi * (MA/2.0) * (ma/2.0)

def sample_ellipse_perimeter(ellipse, samples=SAMPLE_POINTS):
    # ellipse: ((cx,cy),(MA,ma),angle_deg)
    (cx, cy), (MA, ma), angle = ellipse
    a = MA / 2.0
    b = ma / 2.0
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    pts = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        x_ = a * math.cos(t)
        y_ = b * math.sin(t)
        # rotate
        x = cx + x_ * cos_t - y_ * sin_t
        y = cy + x_ * sin_t + y_ * cos_t
        pts.append((int(round(x)), int(round(y))))
    return pts

def perimeter_match_ratio(ellipse, edges_map, thickness=PERIM_THICKNESS, samples=SAMPLE_POINTS):
    h, w = edges_map.shape
    pts = sample_ellipse_perimeter(ellipse, samples)
    match = 0
    for (x, y) in pts:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        x0 = max(0, x-thickness)
        x1 = min(w, x+thickness+1)
        y0 = max(0, y-thickness)
        y1 = min(h, y+thickness+1)
        if np.any(edges_map[y0:y1, x0:x1]):
            match += 1
    denom = samples if samples > 0 else 1
    return float(match) / denom


def detect_by_hough(img, target_ratio=TARGET_RATIO, debug=False):
    """
    基于 DoG + HoughCircles（回复1 类型）
    返回：(success, method_name, ellipse_tuple, area_ratio, extra_dict)
    ellipse_tuple 使用 fitEllipse 风格： ((cx,cy),(MA,ma),angle)
    """
    gray = clahe_gray(img)
    blur1 = cv2.GaussianBlur(gray, (5,5), 1)
    blur2 = cv2.GaussianBlur(gray, (15,15), 3)
    dog = cv2.subtract(blur1, blur2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    # try HoughCircles on the binary image; tune params lightly
    h, w = img.shape[:2]
    maxR = int(min(h, w) / 2)
    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=80, param2=28, minRadius=10, maxRadius=maxR)
    if circles is None:
        return (False, 'hough', None, None, {})
    circles = np.uint16(np.around(circles[0,:]))
    # pick circle closest to target_ratio
    best = None
    best_diff = 1e9
    for (x, y, r) in circles:
        area_ratio = math.pi * r * r / (h * w)
        diff = abs(area_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            # represent as ellipse (MA,ma) = (2r,2r)
            ellipse = ((float(x), float(y)), (2.0*r, 2.0*r), 0.0)
            best = (ellipse, area_ratio)
    if best is None:
        return (False, 'hough', None, None, {})
    ellipse, area_ratio = best
    return (True, 'hough', ellipse, area_ratio, {'binary': binary})

def detect_by_fitellipse(img, target_ratio=TARGET_RATIO, debug=False):
    """
    基于边缘增强 + findContours + fitEllipse（你的后续fitEllipse方法）
    返回相同格式
    """
    gray = clahe_gray(img)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    lap = cv2.Laplacian(blur, cv2.CV_8U)
    _, binary = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7,7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    img_area = h * w
    best = None
    best_diff = 1e9
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_AREA_RATIO:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), ang = ellipse
        ellipse_area = ellipse_area_from_params(MA, ma)
        area_ratio = ellipse_area / img_area
        diff = abs(area_ratio - target_ratio)
        # prefer center near image center (soft)
        if MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO and diff < best_diff:
            best_diff = diff
            best = (ellipse, area_ratio)
    if best is None:
        return (False, 'fitellipse', None, None, {'binary': binary})
    ellipse, area_ratio = best
    return (True, 'fitellipse', ellipse, area_ratio, {'binary': binary})

def detect_by_dogfallback(img, target_ratio=TARGET_RATIO, debug=False):
    """
    DoG + slightly different Hough / fallback. （作为第三策略）
    """
    # reuse edges for scoring and try looser Hough parameters
    gray = clahe_gray(img)
    blur1 = cv2.GaussianBlur(gray, (3,3), 0.5)
    blur2 = cv2.GaussianBlur(gray, (11,11), 2)
    dog = cv2.subtract(blur1, blur2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = img.shape[:2]
    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20,
                               param1=60, param2=25, minRadius=8, maxRadius=int(min(h,w)/2))
    if circles is None:
        return (False, 'dog_hough', None, None, {'binary': binary})
    circles = np.uint16(np.around(circles[0,:]))
    best = None
    best_diff = 1e9
    for (x,y,r) in circles:
        area_ratio = math.pi * r * r / (h*w)
        diff = abs(area_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            ellipse = ((float(x), float(y)), (2.0*r, 2.0*r), 0.0)
            best = (ellipse, area_ratio)
    if best is None:
        return (False, 'dog_hough', None, None, {'binary': binary})
    ellipse, area_ratio = best
    return (True, 'dog_hough', ellipse, area_ratio, {'binary': binary})

# ========== 主流程（ensemble） ==========
def choose_best_candidate(img, candidates, target_ratio=TARGET_RATIO):
    """
    candidates: list of tuples (method_name, ellipse, area_ratio, extra)
    returns chosen tuple and scoring dict
    """
    h, w = img.shape[:2]
    img_center = (w/2.0, h/2.0)
    diag = math.hypot(w, h)

    # compute edges_map for perimeter matching (uniform)
    gray = clahe_gray(img)
    edges_map = edges_for_scoring(gray)

    scored = []
    for method, ellipse, area_ratio, extra in candidates:
        if ellipse is None:
            continue
        # compute diff and center distance
        (cx, cy), (MA, ma), ang = ellipse
        diff = abs(area_ratio - target_ratio)
        center_dist = math.hypot(cx - img_center[0], cy - img_center[1])
        center_norm = center_dist / diag  # 0..~1
        # compute perimeter match
        match = perimeter_match_ratio(ellipse, edges_map, thickness=PERIM_THICKNESS, samples=SAMPLE_POINTS)
        # final score: smaller better
        final_score = diff - WEIGHT_EDGE * match + WEIGHT_CENTER * center_norm
        scored.append({
            'method': method,
            'ellipse': ellipse,
            'area_ratio': area_ratio,
            'diff': diff,
            'center_norm': center_norm,
            'match': match,
            'final_score': final_score
        })

    if not scored:
        return None, None

    # choose smallest final_score
    scored_sorted = sorted(scored, key=lambda x: x['final_score'])
    best = scored_sorted[0]
    # if top two close and centers agree, average centers (optional)
    if len(scored_sorted) > 1:
        second = scored_sorted[1]
        cen_dist = math.hypot(best['ellipse'][0][0] - second['ellipse'][0][0],
                              best['ellipse'][0][1] - second['ellipse'][0][1])
        if cen_dist < 10.0:
            # average centers, keep ellipse size of best
            (cx,cy),(MA,ma),ang = best['ellipse']
            sx = (best['ellipse'][0][0] + second['ellipse'][0][0]) / 2.0
            sy = (best['ellipse'][0][1] + second['ellipse'][0][1]) / 2.0
            best['ellipse'] = ((sx,sy),(MA,ma),ang)
    return best, scored_sorted

def process_folder_ensemble(input_folder, output_folder, target_ratio=TARGET_RATIO):
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "ellipse_centers_ensemble.csv")
    no_detect_folder = os.path.join(output_folder, "no_detection")
    os.makedirs(no_detect_folder, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename','chosen_method','center_x','center_y','area_ratio','match','final_score','all_candidates'])
        for fname in sorted(os.listdir(input_folder)):
            if not fname.lower().endswith(valid_exts):
                continue
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)

            img = cv2.imread(in_path)
            if img is None:
                print("无法读取：", in_path)
                continue

            # run detectors
            cands = []
            for func in (detect_by_hough, detect_by_fitellipse, detect_by_dogfallback):
                success, method, ellipse, area_ratio, extra = func(img, target_ratio)
                if success:
                    cands.append((method, ellipse, area_ratio, extra))
                else:
                    # we still pass candidate with None to extra for debugging if needed
                    pass

            # choose best
            best, all_scored = choose_best_candidate(img, cands, target_ratio)
            if best is None:
                # save original into no_detection
                cv2.imwrite(os.path.join(no_detect_folder, fname), img)
                print(f"[NO DET] {fname}")
                writer.writerow([fname, 'none', '', '', '', '', '', str([])])
                continue

            # draw chosen result
            ellipse = best['ellipse']
            (cx, cy), (MA, ma), ang = ellipse
            cx_i, cy_i = int(round(cx)), int(round(cy))
            # draw ellipse & center & text
            out_img = img.copy()
            cv2.ellipse(out_img, ( (int(round(cx)), int(round(cy))), (int(round(MA)), int(round(ma))), float(ang) ), (0,255,0), 2)
            cv2.circle(out_img, (cx_i, cy_i), 5, (0,0,255), -1)
            method_name = best['method']
            area_ratio = best['area_ratio']
            match = best['match']
            final_score = best['final_score']
            txt1 = f"{method_name}  Center: ({cx_i},{cy_i})"
            txt2 = f"AreaRatio:{area_ratio:.3f} Match:{match:.3f}"
            cv2.putText(out_img, txt1, (max(5, cx_i-140), min(out_img.shape[0]-10, cy_i+35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.putText(out_img, txt2, (max(5, cx_i-140), min(out_img.shape[0]-10, cy_i+55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.imwrite(out_path, out_img)
            # record
            # also record other candidates summaries for debugging
            cand_summary = []
            for s in (all_scored or []):
                cand_summary.append(f"{s['method']}|ratio:{s['area_ratio']:.3f}|match:{s['match']:.3f}|score:{s['final_score']:.3f}")
            writer.writerow([fname, method_name, f"{cx:.2f}", f"{cy:.2f}", f"{area_ratio:.4f}", f"{match:.4f}", f"{final_score:.4f}", ";".join(cand_summary)])
            print(f"[OK] {fname} -> {method_name}, center=({cx_i},{cy_i}), ratio={area_ratio:.3f}, match={match:.3f}")

    print("\n完成。结果保存到：", output_folder)
    print("CSV:", csv_path)


# ========== 运行入口 ==========
if __name__ == "__main__":
    # 改成你自己的路径
    input_folder = r"F:/YOLO/videos/cropped_frames_refined/spout"
    output_folder = r"F:/YOLO/videos/cropped_frames_refined/out_ensemble"
    os.makedirs(output_folder, exist_ok=True)
    process_folder_ensemble(input_folder, output_folder, target_ratio=TARGET_RATIO)
