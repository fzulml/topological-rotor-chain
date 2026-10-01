"""
figure2_4_long_trace.py
生成 Fig. 2-4 的 6 张子图，每张单独保存。

输出：
  figure2a.png  — D_j 沿路径
  figure2b.png  — σ_min(C) 和 σ_min(C^R) 全路径
  figure3a.png  — D_13 穿零细节
  figure3b.png  — σ_min 在零点附近（y 轴缩放）
  figure4a.png  — D_j 在驱动折叠附近
  figure4b.png  — σ_min(C^R) → 0 折叠细节（y 轴扩展）

改进：
  (1) Fig. 2(a) 图例移至 lower right
  (2) Fig. 3(b) y 轴范围收紧至 [0.516, 0.517]
  (3) Fig. 4(b) y 轴下界扩展到 1e-5

用法：
  python figure2_4_long_trace.py
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, null_space
import csv
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7

COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC',
          '#CA9161', '#949494', '#56B4E9', '#F0E442']


# ============================================================
# RotorChain 与延拓
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

    def local_denominator(self, theta, j):
        ti = theta[j]; tj = theta[j + 1]
        return self.rho * np.cos(tj) - np.sin(ti + tj)


def tangent_direction(chain, theta):
    C = chain.constraint_jacobian(theta)
    V = null_space(C)
    if V.shape[1] == 0:
        return None
    v = V[:, 0]
    return v / np.linalg.norm(v)


def newton_on_constraint(chain, theta_pred, tangent,
                         tol=1e-11, max_iter=80):
    theta = theta_pred.copy()
    for _ in range(max_iter):
        F = chain.constraint(theta)
        C = chain.constraint_jacobian(theta)
        J = np.vstack([C, tangent.reshape(1, -1)])
        r = -np.concatenate([F, [tangent @ (theta - theta_pred)]])
        try:
            delta = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, r, rcond=None)
        theta = theta + delta
        if np.linalg.norm(delta) < tol:
            return theta, True
    return theta, False


def run_long_trace(n=16, rho=1.0, theta_bar=np.pi/2 - 0.3,
                   ds=0.005, s_max=20.0):
    """运行长距离伪弧长延拓，返回 history 记录。"""
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

    # 初始化：沿零模方向
    theta = np.full(n, theta_bar)
    C0 = chain.constraint_jacobian(theta)
    U, S, Vt = svd(C0, full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0:
        v0 = -v0
    theta = theta + 1e-6 * v0
    for _ in range(50):
        F = chain.constraint(theta)
        if np.linalg.norm(F) < 1e-12:
            break
        C = chain.constraint_jacobian(theta)
        delta, *_ = np.linalg.lstsq(C, -F, rcond=None)
        theta += delta

    tangent = tangent_direction(chain, theta)
    if tangent[-1] < 0:
        tangent = -tangent

    history = []
    s = 0.0
    while s < s_max:
        C = chain.constraint_jacobian(theta)
        C_R = C[:, :-1]
        sv_C = svd(C, compute_uv=False)
        sv_CR = svd(C_R, compute_uv=False)
        D = np.array([chain.local_denominator(theta, j)
                      for j in range(n - 1)])
        row = {
            's': s,
            'q': theta[-1] - theta_bar,
            'sigma_min_C': float(sv_C[-1]),
            'sigma_min_CR': float(sv_CR[-1]),
        }
        for j in range(n - 1):
            row[f'D_{j}'] = float(D[j])
        history.append(row)

        if sv_C[-1] < 1e-8:
            break

        theta_pred = theta + ds * tangent
        theta_new, ok = newton_on_constraint(
            chain, theta_pred, tangent)
        if not ok:
            break
        tn = tangent_direction(chain, theta_new)
        if tn is None:
            break
        if np.dot(tn, tangent) < 0:
            tn = -tn
        tangent = tn
        theta = theta_new
        s += ds

    return history


def save_history_csv(history, path):
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def load_history_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data = {}
    for col in reader.fieldnames:
        values = [row[col] for row in rows]
        try:
            data[col] = np.array([float(v) for v in values])
        except ValueError:
            data[col] = np.array(values)
    return data


# ============================================================
# 辅助函数
# ============================================================
def find_crossing(s, D):
    """找 D 首次穿零的位置（线性插值）。"""
    for k in range(1, len(D)):
        if D[k-1] * D[k] < 0:
            frac = -D[k-1] / (D[k] - D[k-1])
            return s[k-1] + frac * (s[k] - s[k-1])
    return None


def find_fold(s, smin_CR, threshold=1e-3):
    """找 σ_min(C^R) 首次下穿阈值的位置。"""
    for k, val in enumerate(smin_CR):
        if val < threshold:
            return s[k]
    return None


# ============================================================
# 主程序
# ============================================================
def main():
    data_path = 'results_s20/forward/long_trace.csv'

    # ---------- 读取或生成数据 ----------
    if os.path.exists(data_path):
        print(f"Loading: {data_path}")
        data = load_history_csv(data_path)
    else:
        print(f"Data not found. Running long trace...")
        history = run_long_trace(n=16, rho=1.0,
                                 theta_bar=np.pi/2 - 0.3,
                                 ds=0.005, s_max=20.0)
        save_history_csv(history, data_path)
        print(f"Saved: {data_path}")
        data = load_history_csv(data_path)

    # ---------- 提取数据 ----------
    s = data['s']
    q = data['q']
    smin_C = data['sigma_min_C']
    smin_CR = data['sigma_min_CR']
    D_cols = sorted(
        [c for c in data.keys() if c.startswith('D_')],
        key=lambda x: int(x.split('_')[1]))

    j_vals = [int(c.split('_')[1]) for c in D_cols]

    print(f"n_constraints = {len(j_vals)}")
    print(f"s range: {s[0]:.3f} to {s[-1]:.3f}")
    print(f"Total steps: {len(s)}")

    # 用于 Fig. 2(a) 和 Fig. 4(a) 的 j 值范围
    plot_js = [j for j in j_vals if 6 <= j <= 13]

    # ============================================================
    # Fig. 2(a): D_j 沿路径
    # ============================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    for i, j in enumerate(plot_js):
        D = data[f'D_{j}']
        ax.plot(s, D, '-', color=COLORS[i % len(COLORS)],
                lw=0.9, label=f'$D_{{{j}}}$')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel(r'arclength $s$', fontsize=10)
    ax.set_ylabel(r'$D_j$', fontsize=10)
    # 改进 1：图例移至 lower right
    ax.legend(fontsize=5.5, ncol=2, loc='lower right',
              frameon=True, framealpha=0.9,
              handlelength=1.0, columnspacing=0.5,
              borderpad=0.3, labelspacing=0.2)
    ax.grid(alpha=0.3, lw=0.3)
    plt.tight_layout()
    plt.savefig('figure2a.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure2a.png")

    # ============================================================
    # Fig. 2(b): σ_min(C) 和 σ_min(C^R) 全路径
    # ============================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    ax.semilogy(s, smin_C, '-', color=COLORS[0], lw=1.2,
                label=r'$\sigma_{\min}(C)$')
    ax.semilogy(s, smin_CR, '-', color=COLORS[1], lw=1.2,
                label=r'$\sigma_{\min}(C^R)$')
    ax.set_xlabel(r'arclength $s$', fontsize=10)
    ax.set_ylabel(r'$\sigma_{\min}$', fontsize=10)
    ax.legend(fontsize=7, loc='lower left',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figure2b.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure2b.png")

    # ============================================================
    # Fig. 3(a): D_13 穿零细节
    # ============================================================
    D13 = data['D_13']
    s_cross = find_crossing(s, D13)
    if s_cross is None:
        print("WARNING: D_13 does not cross zero.")
        return
    print(f"D_13 crosses zero at s = {s_cross:.4f}")

    window = 0.5
    mask3 = (s >= s_cross - window) & (s <= s_cross + window)

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    ax.plot(s[mask3], D13[mask3], '-', color=COLORS[0], lw=1.5)
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(s_cross, color='#CC78BC', ls=':', lw=0.8)
    ax.set_xlabel(r'arclength $s$', fontsize=10)
    ax.set_ylabel(r'$D_{13}$', fontsize=10)
    ax.grid(alpha=0.3, lw=0.3)
    plt.tight_layout()
    plt.savefig('figure3a.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure3a.png")

    # ============================================================
    # Fig. 3(b): σ_min 在零点附近
    # 改进 2：y 轴范围收紧至 [0.516, 0.517]
    # ============================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    ax.plot(s[mask3], smin_C[mask3], '-', color=COLORS[0], lw=1.2,
            label=r'$\sigma_{\min}(C)$')
    ax.plot(s[mask3], smin_CR[mask3], '-', color=COLORS[1], lw=1.2,
            label=r'$\sigma_{\min}(C^R)$')
    ax.axvline(s_cross, color='#CC78BC', ls=':', lw=0.8)
    ax.set_xlabel(r'arclength $s$', fontsize=10)
    ax.set_ylabel(r'$\sigma_{\min}$', fontsize=10)
    # 改进 2：y 轴范围 [0.516, 0.517]
    ax.set_ylim(0.516, 0.517)
    ax.legend(fontsize=7, loc='lower left',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3)
    plt.tight_layout()
    plt.savefig('figure3b.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure3b.png")

    # ============================================================
    # Fig. 4(a): D_j 在驱动折叠附近
    # ============================================================
    s_fold = find_fold(s, smin_CR, threshold=1e-3)
    if s_fold is None:
        print("WARNING: no fold detected.")
        return
    print(f"First fold at s = {s_fold:.4f}")

    window = 1.0
    mask4 = (s >= s_fold - window) & (s <= s_fold + window)

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    for i, j in enumerate(plot_js):
        D = data[f'D_{j}']
        ax.plot(s[mask4], D[mask4], '-',
                color=COLORS[i % len(COLORS)], lw=0.9,
                label=f'$D_{{{j}}}$')
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(s_fold, color='#CC78BC', ls=':', lw=0.8)
    ax.set_xlabel(r'arclength $s$', fontsize=10)
    ax.set_ylabel(r'$D_j$', fontsize=10)
    ax.legend(fontsize=5.5, ncol=2, loc='upper right',
              frameon=True, framealpha=0.9,
              handlelength=1.0, columnspacing=0.5,
              borderpad=0.3, labelspacing=0.2)
    ax.grid(alpha=0.3, lw=0.3)
    plt.tight_layout()
    plt.savefig('figure4a.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure4a.png")

    # ============================================================
    # Fig. 4(b): σ_min(C^R) → 0 折叠细节
    # 改进 3：y 轴下界扩展到 1e-5
    # ============================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    ax.semilogy(s[mask4], smin_C[mask4], '-',
                color=COLORS[0], lw=1.2,
                label=r'$\sigma_{\min}(C)$')
    ax.semilogy(s[mask4], smin_CR[mask4], '-',
                color=COLORS[1], lw=1.2,
                label=r'$\sigma_{\min}(C^R)$')
    ax.axvline(s_fold, color='#CC78BC', ls=':', lw=0.8)
    ax.set_xlabel(r'arclength $s$', fontsize=10)
    ax.set_ylabel(r'$\sigma_{\min}$', fontsize=10)
    # 改进 3：y 轴范围 [1e-5, 1e0]
    ax.set_ylim(1e-5, 1e0)
    ax.legend(fontsize=7, loc='lower left',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figure4b.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure4b.png")

    print("\nAll 6 panels saved:")
    print("  figure2a.png, figure2b.png")
    print("  figure3a.png, figure3b.png")
    print("  figure4a.png, figure4b.png")


if __name__ == '__main__':
    main()
