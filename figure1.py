"""
figure1.py
生成 Fig. 1 的 2 张子图：
  figure1a.png  — 转子链几何示意图
  figure1b.png  — 零模振幅分布

格式与其他图统一：
  - Times New Roman 字体
  - 单栏尺寸 3.4 x 2.8 英寸，300 dpi
  - 色盲友好调色板（Okabe-Ito）
  - 与 Fig. 2-5 一致的网格和标注样式

用法：
  python figure1.py
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from scipy.linalg import svd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7

# 与 Fig. 2-5 统一的调色板
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC',
          '#CA9161', '#949494', '#56B4E9', '#F0E442']


# ============================================================
# RotorChain
# ============================================================
class RotorChain:
    def __init__(self, n, r=1.0, a=1.0, theta_bar=np.pi/2 - 0.3):
        self.n = n; self.r = r; self.a = a
        self.theta_bar = theta_bar
        self.rho = a / r
        self.l_bar = np.sqrt(
            a**2 + 4 * r**2 * np.cos(theta_bar)**2)

    def spring_length(self, theta, j):
        ti = theta[j]; tj = theta[j + 1]
        return np.hypot(
            self.a + self.r * (np.sin(tj) - np.sin(ti)),
            self.r * (np.cos(ti) + np.cos(tj)))

    def constraint(self, theta):
        return np.array([
            self.spring_length(theta, j) - self.l_bar
            for j in range(self.n - 1)])

    def constraint_jacobian(self, theta):
        n = self.n
        C = np.zeros((n - 1, n))
        for i in range(n - 1):
            l = self.spring_length(theta, i)
            ti = theta[i]; tj = theta[i + 1]
            C[i, i] = (
                -self.a * self.r * np.cos(ti)
                - self.r**2 * np.sin(ti + tj)) / l
            C[i, i + 1] = (
                +self.a * self.r * np.cos(tj)
                - self.r**2 * np.sin(ti + tj)) / l
        return C


# ============================================================
# Fig. 1(a): 几何示意图
# ============================================================
def draw_rotor_chain(theta, r=1.0, a=1.0, ax=None):
    """绘制 Kane-Lubensky 交替约定下的转子链。"""
    theta = np.asarray(theta, dtype=float)
    n = len(theta)

    x_pivots = np.arange(n) * a
    y_pivots = np.zeros(n)
    signs = (-1.0) ** np.arange(n)
    x_tips = x_pivots + r * np.sin(theta)
    y_tips = r * np.cos(theta) * signs

    ax.set_xlim(x_pivots[0] - 0.6 * a, x_pivots[-1] + 1.6 * a)
    ax.set_ylim(-1.35 * r, 1.35 * r)

    # 基线
    ax.plot([x_pivots[0] - 0.6 * a, x_pivots[-1] + 1.6 * a],
            [0, 0], 'k--', lw=0.4, alpha=0.4, zorder=1)
    # 转轴
    ax.scatter(x_pivots, y_pivots, s=25, c='k', zorder=5)
    # 转子
    for i in range(n):
        ax.plot([x_pivots[i], x_tips[i]],
                [y_pivots[i], y_tips[i]],
                'o-', color=COLORS[0], lw=1.4,
                markersize=3.5, zorder=4)
    # 弹簧
    for i in range(n - 1):
        x1, y1 = x_tips[i], y_tips[i]
        x2, y2 = x_tips[i + 1], y_tips[i + 1]
        t = np.linspace(0, 1, 80)
        x_line = x1 + (x2 - x1) * t
        y_line = y1 + (y2 - y1) * t
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L > 1e-9:
            nx, ny = -dy / L, dx / L
            amp = min(0.07 * L, 0.06 * r)
            x_line += nx * amp * np.sin(6 * np.pi * t)
            y_line += ny * amp * np.sin(6 * np.pi * t)
        ax.plot(x_line, y_line, '-', color=COLORS[1],
                lw=0.9, zorder=3)

    # 间距 a 标注
    ax.annotate('', xy=(x_pivots[0], -0.85 * r),
                xytext=(x_pivots[1], -0.85 * r),
                arrowprops=dict(arrowstyle='<->',
                                color='gray', lw=0.5))
    ax.text((x_pivots[0] + x_pivots[1]) / 2, -1.05 * r, r'$a$',
            ha='center', va='top', color='gray', fontsize=9)

    # 转子长度 r 标注
    vx = x_tips[0] - x_pivots[0]
    vy = y_tips[0] - y_pivots[0]
    vlen = np.hypot(vx, vy)
    px, py = -vy / vlen, vx / vlen
    mid_x = (x_pivots[0] + x_tips[0]) / 2
    mid_y = (y_pivots[0] + y_tips[0]) / 2
    ax.text(mid_x + 0.25 * px * r, mid_y + 0.25 * py * r, r'$r$',
            color=COLORS[0], fontsize=9,
            ha='center', va='center')

    # θ_1 弧线
    ax.plot([x_pivots[0], x_pivots[0]], [0, r * 0.95],
            'k:', lw=0.4, alpha=0.5, zorder=2)
    angle_deg = np.degrees(theta[0])
    arc_r = 0.55 * r
    arc = Arc((x_pivots[0], 0), 2 * arc_r, 2 * arc_r,
              angle=0, theta1=90 - angle_deg, theta2=90,
              color=COLORS[0], lw=0.6, zorder=6)
    ax.add_patch(arc)
    la = np.radians(90 - angle_deg / 2)
    ax.text(x_pivots[0] + arc_r * 1.4 * np.cos(la),
            arc_r * 1.4 * np.sin(la),
            r'$\theta_1$', color=COLORS[0], fontsize=9,
            ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/a$', fontsize=10)
    ax.set_ylabel(r'$y/a$', fontsize=10)
    ax.grid(alpha=0.3, lw=0.3)


# ============================================================
# 主程序
# ============================================================
def main():
    n = 16
    theta_bar = np.pi / 2 - 0.3
    rho = 1.0

    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

    # ---------- Fig. 1(a): 几何示意图（前 5 个转子） ----------
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    theta_display = np.full(5, theta_bar)
    draw_rotor_chain(theta_display, r=1.0, a=1.0, ax=ax)
    plt.tight_layout()
    plt.savefig('figure1a.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure1a.png")

    # ---------- Fig. 1(b): 零模振幅 ----------
    theta0 = np.full(n, theta_bar)
    C0 = chain.constraint_jacobian(theta0)
    U, S, Vt = svd(C0, full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0:
        v0 = -v0

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    idx = np.arange(n)
    ax.semilogy(idx, np.abs(v0) / np.max(np.abs(v0)),
                'o-', color=COLORS[0], lw=1.2, markersize=5,
                markerfacecolor=COLORS[0],
                markeredgecolor='white', markeredgewidth=0.3)
    ax.set_xlabel(r'rotor index $i$', fontsize=10)
    ax.set_ylabel(r'$|v_i|/\max|v|$', fontsize=10)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figure1b.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure1b.png")

    print("\nFig. 1 complete: figure1a.png, figure1b.png")


if __name__ == '__main__':
    main()
