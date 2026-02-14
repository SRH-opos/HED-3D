# -*- coding: utf-8 -*-
"""
多学生模型评估结果对比脚本

输入：若干个 eval CSV（每个来自你的评估脚本）
    必须包含列：
        - GT_Z_rel_cam
        - Pred_Z_rel_cam
        - AbsError
    若包含列：
        - Sigma     （来自不确定性回归版本）
    则会额外分析 误差-σ 的关系

输出：
    1) summary_metrics.csv            —— 各模型整体 MAE / RMSE / MaxErr 汇总表
    2) mae_rmse_vs_zbin_compare.png   —— 多模型 Z 分段 MAE / RMSE 对比曲线
    3) （可选，每个带 Sigma 的模型）
       error_vs_sigma_<model>.png     —— AbsError vs Sigma 散点 + 拟合线
"""

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- 用户配置：模型名称 -> 对应 eval CSV 路径 ----------------
MODEL_CSVS = {
    "A_L1":        r"F:\YOLO\DAV2\student\eval_results\spout_depth_student_eval.csv",
    "B_RangeL1":   r"F:\YOLO\DAV2\students\eval_results\spout_depth_student_eval.csv",
    "C_Uncertain": r"F:\YOLO\DAV2\students_2\eval_results_uncertainty\spout_depth_student_uncertainty_eval.csv",
    # ✅ 新增：NIG 版本
    "D_NIG":       r"F:\YOLO\DAV2\students_3\eval_results_uncertainty\spout_depth_student_uncertainty_eval.csv",
    "E_NEW":       r"F:\YOLO\DAV2\students_4\eval_results_uncertainty\spout_depth_student_uncertainty_eval.csv",
}

# 输出目录
OUT_DIR = r"F:\YOLO\DAV2\students_compare"
os.makedirs(OUT_DIR, exist_ok=True)

# Z 分段设置（和你评估脚本一致）
BINS = np.linspace(0.0, 1.0, 11)   # [0,0.1)...[0.9,1.0]


# ---------------- 基础指标计算 ----------------
def compute_global_metrics(df):
    """整体 MAE / RMSE / MaxErr"""
    mae = df["AbsError"].mean()
    rmse = math.sqrt(((df["Pred_Z_rel_cam"] - df["GT_Z_rel_cam"]) ** 2).mean())
    max_err = df["AbsError"].max()
    return mae, rmse, max_err


def compute_binned_metrics(df, bins):
    """按 GT_Z_rel_cam 分段统计 MAE / RMSE 和样本数"""
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    mae_list, rmse_list, count_list = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df["GT_Z_rel_cam"] >= lo) & (df["GT_Z_rel_cam"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            mae_list.append(np.nan)
            rmse_list.append(np.nan)
            count_list.append(0)
            continue
        mae = sub["AbsError"].mean()
        rmse = math.sqrt(((sub["Pred_Z_rel_cam"] - sub["GT_Z_rel_cam"]) ** 2).mean())
        mae_list.append(mae)
        rmse_list.append(rmse)
        count_list.append(len(sub))

    return bin_centers, np.array(mae_list), np.array(rmse_list), np.array(count_list)


# ---------------- 主流程 ----------------
all_summary_rows = []
binned_results = {}   # model -> dict(mae, rmse, bin_centers)

for model_name, csv_path in MODEL_CSVS.items():
    if not os.path.exists(csv_path):
        print(f"[WARN] 模型 {model_name} 的 CSV 不存在：{csv_path}，跳过")
        continue

    print(f"\n===== 处理模型：{model_name} =====")
    df = pd.read_csv(csv_path)

    # 基础列检查
    required_cols = {"GT_Z_rel_cam", "Pred_Z_rel_cam", "AbsError"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列 {required_cols}，实际列为：{df.columns.tolist()}")

    # 1) 整体指标
    mae, rmse, max_err = compute_global_metrics(df)
    print(f"  全局 MAE={mae:.6f}, RMSE={rmse:.6f}, MaxErr={max_err:.6f}")
    all_summary_rows.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MaxErr": max_err,
        "NumSamples": len(df)
    })

    # 2) 分段指标
    bin_centers, mae_bins, rmse_bins, counts = compute_binned_metrics(df, BINS)
    binned_results[model_name] = {
        "bin_centers": bin_centers,
        "mae_bins": mae_bins,
        "rmse_bins": rmse_bins,
        "counts": counts
    }

    # 3) 若有 Sigma 列，分析误差-σ 关系（只对不确定性模型有用）
    if "Sigma" in df.columns:
        print("  检测到 Sigma 列，绘制误差-σ 散点图...")
        sigma = df["Sigma"].values
        err = df["AbsError"].values

        # 计算皮尔逊相关系数
        if len(sigma) > 1:
            corr = np.corrcoef(sigma, err)[0, 1]
        else:
            corr = np.nan
        print(f"  corr(AbsError, Sigma) = {corr:.4f}")

        # 画散点 + 一条简单的线性拟合
        plt.figure(figsize=(5, 4))
        plt.scatter(sigma, err, s=8, alpha=0.5, label="samples")
        if len(sigma) > 1:
            k, b = np.polyfit(sigma, err, 1)
            xs = np.linspace(sigma.min(), sigma.max(), 100)
            ys = k * xs + b
            plt.plot(xs, ys, "r-", label=f"fit: y={k:.3f}x+{b:.3f}")
        plt.xlabel("Sigma (predicted uncertainty)")
        plt.ylabel("AbsError")
        plt.title(f"Error vs Sigma ({model_name}), corr={corr:.3f}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        out_scatter = os.path.join(OUT_DIR, f"error_vs_sigma_{model_name}.png")
        plt.tight_layout()
        plt.savefig(out_scatter, dpi=200)
        plt.close()
        print(f"  已保存误差-σ 散点图：{out_scatter}")

# ---------------- 保存整体汇总表 ----------------
if len(all_summary_rows) == 0:
    print("没有可用模型，检查 MODEL_CSVS 配置。")
else:
    df_summary = pd.DataFrame(all_summary_rows)
    summary_path = os.path.join(OUT_DIR, "summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n总评估表已保存：{summary_path}")
    print(df_summary)

# ---------------- 画 MAE / RMSE 分段对比曲线 ----------------
if len(binned_results) > 0:
    # MAE 对比
    plt.figure(figsize=(6.5, 4))
    for model_name, res in binned_results.items():
        plt.plot(res["bin_centers"], res["mae_bins"],
                 marker="o", label=f"{model_name}")
    plt.xlabel("GT Z_rel_cam")
    plt.ylabel("MAE")
    plt.title("MAE vs Z bins (multi-model)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_mae = os.path.join(OUT_DIR, "mae_vs_zbin_compare.png")
    plt.savefig(out_mae, dpi=200)
    plt.close()
    print(f"分模型 MAE 曲线已保存：{out_mae}")

    # RMSE 对比
    plt.figure(figsize=(6.5, 4))
    for model_name, res in binned_results.items():
        plt.plot(res["bin_centers"], res["rmse_bins"],
                 marker="s", label=f"{model_name}")
    plt.xlabel("GT Z_rel_cam")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Z bins (multi-model)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_rmse = os.path.join(OUT_DIR, "rmse_vs_zbin_compare.png")
    plt.savefig(out_rmse, dpi=200)
    plt.close()
    print(f"分模型 RMSE 曲线已保存：{out_rmse}")

print("\n对比完成。")
