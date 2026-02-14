# -*- coding: utf-8 -*-
import os
import re
import shutil
from pathlib import Path
import cv2
import numpy as np

# ================== 路径配置 ==================
THIRD_DIR   = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v3_refined_v3_healed")                          # 第三轮输出（黑底）
FOURTH_DIR  = Path(r"F:\YOLO\DAV2\contour_keep_ellipses_roi_v4_recovered")                          # 第四轮输出（黑底）
ORIG_DIR    = Path(r"F:\YOLO\DAV2\image")                      # 原图

MERGED_DIR         = Path(r"F:\YOLO\DAV2\5")                   # 合并后的黑底结果
MERGED_OVERLAY_DIR = Path(r"F:\YOLO\DAV2\5.1")                 # 合并结果叠加在原图上的可视化

# 支持的图像后缀
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ================== 工具函数 ==================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    return {f for f in os.listdir(folder) if f.lower().endswith(EXTS)} if folder.exists() else set()

def extract_candidate_stems(result_name: str):
    """
    从结果文件名中抽取可能的原图 stem：
    - 优先提取 'spout_<num>'
    - 若无，则取最后一个数字 token
    返回列表（去重，保持稳定顺序），例如：
      'spout_280_contour_refined' -> ['spout_280', '280']
      'foo_001_bar'               -> ['spout_1', '1']
    """
    s = Path(result_name).stem
    cands = []

    # 1) 匹配 spout_<num>
    m = re.findall(r'(spout_(\d+))', s, flags=re.IGNORECASE)
    if m:
        full, num = m[-1]
        cands.append(full.lower())       # 'spout_280'
        cands.append(str(int(num)))      # '280' (去前导零)

    # 2) 若没有，则匹配最后一个数字
    if not m:
        m2 = re.findall(r'(\d+)', s)
        if m2:
            num = str(int(m2[-1]))
            cands.append(f"spout_{num}")  # 'spout_280'
            cands.append(num)             # '280'

    # 3) 最后兜底：把原 stem 也放进去（极端情况）
    cands.append(s.lower())

    # 去重保序
    seen = set()
    out = []
    for t in cands:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def try_exists_with_ext(base_dir: Path, stem: str):
    """尝试 stem+任意后缀 是否存在，存在则返回路径，否则 None"""
    for ext in EXTS:
        p = base_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def find_original_for_result(orig_dir: Path, result_name: str):
    """
    更智能的原图定位：
    1) 先尝试完全同名（不同后缀）
    2) 再用 'spout_<num>' 和 '<num>' 两种 stem 尝试
    3) 若仍找不到，扫描原图目录，对提取到的 <num> 做“数值等价”匹配（忽略前导零）
    """
    stem = Path(result_name).stem.lower()

    # (1) 同名尝试
    p = try_exists_with_ext(orig_dir, stem)
    if p is not None:
        return p

    # (2) 候选 stems 尝试
    cand_stems = extract_candidate_stems(result_name)
    for st in cand_stems:
        p = try_exists_with_ext(orig_dir, st)
        if p is not None:
            return p

    # (3) 兜底：按数字等价扫描
    nums = re.findall(r'(\d+)', stem)
    if nums:
        target_num = int(nums[-1])
        for f in os.listdir(orig_dir):
            q = Path(f)
            if q.suffix.lower() not in EXTS:
                continue
            qnums = re.findall(r'(\d+)', q.stem.lower())
            if qnums and int(qnums[-1]) == target_num:
                return orig_dir / f

    return None

def overlay_on_original(orig_bgr, canvas_bgr):
    """
    只把 canvas 上的彩色线条（非黑）叠到原图上。
    如果大小不一致，先把 canvas resize 成原图大小。
    """
    H, W = orig_bgr.shape[:2]
    if canvas_bgr.shape[:2] != (H, W):
        canvas_bgr = cv2.resize(canvas_bgr, (W, H), interpolation=cv2.INTER_NEAREST)

    # 非黑像素作为前景（线条）
    mask = np.any(canvas_bgr > 15, axis=2)  # bool HxW
    out = orig_bgr.copy()
    out[mask] = canvas_bgr[mask]
    return out

# ================== 主流程 ==================
def main():
    ensure_dir(MERGED_DIR)
    ensure_dir(MERGED_OVERLAY_DIR)

    third_names  = list_images(THIRD_DIR)
    fourth_names = list_images(FOURTH_DIR)
    all_names    = sorted(third_names | fourth_names)

    used_third = used_fourth = skipped = 0

    print(f"第三轮已有 {len(third_names)}，第四轮补充 {len(fourth_names)}，合计候选 {len(all_names)}")

    for i, name in enumerate(all_names, 1):
        # 选择优先级：第三轮 > 第四轮
        if name in third_names:
            src = THIRD_DIR / name
            source_tag = "3rd"
            used_third += 1
        elif name in fourth_names:
            src = FOURTH_DIR / name
            source_tag = "4th"
            used_fourth += 1
        else:
            skipped += 1
            print(f"[{i}/{len(all_names)}] {name}: 未在第三/第四轮中找到，跳过")
            continue

        # 拷贝黑底结果到合并目录
        dst_canvas = MERGED_DIR / name
        shutil.copy2(str(src), str(dst_canvas))

        # 在原图上叠加
        orig_path = find_original_for_result(ORIG_DIR, name)

        if orig_path and orig_path.exists():
            orig   = cv2.imread(str(orig_path))
            canvas = cv2.imread(str(src))
            if orig is None or canvas is None:
                print(f"[{i}/{len(all_names)}] {name}: 读图失败（原图或画布），仅完成黑底拷贝")
                continue

            merged = overlay_on_original(orig, canvas)
            cv2.imwrite(str(MERGED_OVERLAY_DIR / name), merged)
            print(f"[{i}/{len(all_names)}] {name}: 合并({source_tag})并叠加完成 -> 原图: {orig_path.name}")
        else:
            # 给出更明确的失败提示：展示匹配到的候选 stem
            cand = extract_candidate_stems(name)
            print(f"[{i}/{len(all_names)}] {name}: 找不到对应原图（试过 {cand}），仅完成黑底拷贝")

    print(f"\n完成：第三轮直接采用 {used_third}，第四轮补充 {used_fourth}，未处理/跳过 {skipped}")
    print(f"黑底合并输出：{MERGED_DIR}")
    print(f"原图叠加输出：{MERGED_OVERLAY_DIR}")

if __name__ == "__main__":
    main()
