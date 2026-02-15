import os
import pandas as pd
import matplotlib.pyplot as plt

# 日志路径（改成你自己的）
LOG_CSV = r"F:\YOLO\DAV2\students_2\pout_depth_student_uncertainty_rank_logs\train_spout_depth_student_uncertainty_rank.csv"

# 读日志
df = pd.read_csv(LOG_CSV)

# 如果没有全局 step，就自己造一个：按行累加
df["step"] = range(1, len(df) + 1)

print(df.head())

# 1) loss vs step（迭代级别）
plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["loss"], marker="", linewidth=1)
plt.xlabel("Step")
plt.ylabel("Loss (L1 on z)")
plt.title("Student Training Loss vs Step")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(LOG_CSV), "loss_vs_step.png"), dpi=200)
plt.show()

# 2) 每个 epoch 的平均 loss
epoch_mean = df.groupby("epoch")["loss"].mean().reset_index()

plt.figure(figsize=(6, 4))
plt.plot(epoch_mean["epoch"], epoch_mean["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Mean Loss per Epoch")
plt.title("Student Training Loss ")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(epoch_mean["epoch"])
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(LOG_CSV), "loss_vs_epoch.png"), dpi=200)
plt.show()

print("保存的曲线：")
print(" -", os.path.join(os.path.dirname(LOG_CSV), "loss_vs_step.png"))
print(" -", os.path.join(os.path.dirname(LOG_CSV), "loss_vs_epoch.png"))
