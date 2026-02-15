import numpy as np
import matplotlib.pyplot as plt

# 椭圆参数
a = 3.0   # 半长轴
b = 2.0   # 半短轴
center = (0.0, 0.0)

# ========= 工具函数 =========
def ellipse_xy(theta):
    """给定参数 t，返回椭圆上的 (x,y)"""
    cx, cy = center
    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)
    return x, y

def ellipse_tangent(theta):
    """椭圆在参数 theta 处的单位切向量"""
    # x = a cos t, y = b sin t
    dx = -a * np.sin(theta)
    dy =  b * np.cos(theta)
    v = np.array([dx, dy], dtype=float)
    n = np.linalg.norm(v) + 1e-8
    return v / n

def draw_common(ax):
    """画黑色椭圆并设置坐标系"""
    tt = np.linspace(0, 2*np.pi, 600)
    x, y = ellipse_xy(tt)
    ax.plot(x, y, color="black", linewidth=2)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    # 给底部留出文字空间
    ax.set_xlim(center[0] - a*1.4, center[0] + a*1.4)
    ax.set_ylim(center[1] - b*1.8, center[1] + b*1.2)

def draw_segment(ax, t_start, t_end, color="green", style="-", lw=4):
    """在椭圆上画 t_start → t_end 的弧段"""
    tt = np.linspace(0, 2*np.pi, 800)

    if t_start <= t_end:
        mask = (tt >= t_start) & (tt <= t_end)
    else:
        # 跨越 2π 的情况
        mask = (tt >= t_start) | (tt <= t_end)

    x, y = ellipse_xy(tt[mask])

    # ✅ 只在需要虚线时设置 dashes，避免传 None 触发 len(None) 错误
    if style == "--":
        ax.plot(x, y, linestyle="--", color=color, linewidth=lw, dashes=(8, 6))
    else:
        ax.plot(x, y, linestyle=style, color=color, linewidth=lw)

def draw_point_and_tangent(ax, t, label_p, label_t, rotate_tangent_deg=0.0):
    """画端点、切线箭头和标签"""
    # 点坐标
    x, y = ellipse_xy(t)
    ax.scatter([x], [y], s=40, color="red", zorder=5)

    # 椭圆切线
    t_vec = ellipse_tangent(t)

    # 如需人为制造“角度不合格”，对切线做一个旋转
    if rotate_tangent_deg != 0.0:
        ang = np.deg2rad(rotate_tangent_deg)
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang),  np.cos(ang)]])
        t_vec = R @ t_vec

    # 箭头长度
    L = 0.7
    end = np.array([x, y]) + L * t_vec

    # 画切向箭头
    ax.annotate(
        "", xy=end, xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color="blue", linewidth=2)
    )

    # 让文字稍微离开点/箭头（向外偏移一点法向量）
    # 椭圆法线近似：从中心指向该点
    center_to_p = np.array([x - center[0], y - center[1]], dtype=float)
    n_vec = center_to_p / (np.linalg.norm(center_to_p) + 1e-8)

    # p_i / p_j 放在点外侧
    offset_p = 0.18
    px = x + offset_p * n_vec[0]
    py = y + offset_p * n_vec[1]

    ax.text(px, py, label_p, fontsize=20,
            ha="center", va="center", color="black")

    # t_i / t_j 放在箭头末端稍远一点
    offset_t = 0.18
    tx = end[0] + offset_t * n_vec[0]
    ty = end[1] + offset_t * n_vec[1]

    ax.text(tx, ty, label_t, fontsize=20,
            ha="center", va="center", color="blue")


# ========= 情形 1：合格 =========
def figure_ok(fname="case_ok.png"):
    # 两个端点取得比较近，弧长适中
    t_i = 0.35 * np.pi
    t_j = 0.65 * np.pi

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_common(ax)

    # 绿色实线弧段（可拼接）
    draw_segment(ax, t_i, t_j, color="green", style="-", lw=5)

    # 端点和真实切线
    draw_point_and_tangent(ax, t_i, "p_i", "t_i")
    draw_point_and_tangent(ax, t_j, "p_j", "t_j")

    ax.text(
        0, center[1] - b*1.45,
        "合格：距离和方向都在阈值内，可以拼接",
        fontsize=16, ha="center", va="center", color="black"
    )

    plt.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ========= 情形 2：距离不合格（d > d_max） =========
def figure_dist_fail(fname="case_dist_fail.png"):
    # 取端点相距较远，仍然用真实切线
    t_i = 0.15 * np.pi
    t_j = 0.85 * np.pi

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_common(ax)

    # 红色虚线弧段：理论连接路径，但因距离太大“不允许”
    draw_segment(ax, t_i, t_j, color="red", style="--", lw=4)

    draw_point_and_tangent(ax, t_i, "p_i", "t_i")
    draw_point_and_tangent(ax, t_j, "p_j", "t_j")

    ax.text(
        0, center[1] - b*1.45,
        "距离约束不满足：d > d_max，禁止拼接",
        fontsize=16, ha="center", va="center", color="red"
    )

    plt.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ========= 情形 3：角度不合格（|phi| > phi_max） =========
def figure_angle_fail(fname="case_angle_fail.png"):
    # 距离不算太大，与“合格情况”类似，但故意让一端切线方向偏转很多
    t_i = 0.35 * np.pi
    t_j = 0.65 * np.pi

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_common(ax)

    # 同样是这段椭圆弧，但用红色虚线表示“候选但被拒绝”
    draw_segment(ax, t_i, t_j, color="red", style="--", lw=4)

    # p_i 用真实切线；p_j 的切线人为旋转 75°，模拟方向不一致
    draw_point_and_tangent(ax, t_i, "p_i", "t_i")
    draw_point_and_tangent(ax, t_j, "p_j", "t_j", rotate_tangent_deg=75.0)

    ax.text(
        0, center[1] - b*1.45,
        "方向差过大：|phi| > phi_max，禁止拼接",
        fontsize=16, ha="center", va="center", color="red"
    )

    plt.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_ok()
    figure_dist_fail()
    figure_angle_fail()
    print("已生成 case_ok.png, case_dist_fail.png, case_angle_fail.png")
