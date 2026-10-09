"""
figure_configurations.py
生成构型快照图：
  figure_config_a.png — D_13=0 处的链构型
  figure_config_b.png — Driver fold 处的链构型
  figure_config_c.png — 构型演化序列 (参考态 → D_13=0 → fold)
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, null_space

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC',
          '#CA9161', '#949494', '#56B4E9', '#F0E442']


class RotorChain:
    def __init__(self, n, r=1.0, a=1.0, theta_bar=None):
        self.n = n; self.r = r; self.a = a
        self.rho = a / r
        if theta_bar is not None:
            self.theta_bar = theta_bar
            self.l_bar = np.sqrt(a**2 + 4*r**2*np.cos(theta_bar)**2)

    def spring_length(self, theta, j):
        ti = theta[j]; tj = theta[j + 1]
        return np.hypot(
            self.a + self.r*(np.sin(tj) - np.sin(ti)),
            self.r*(np.cos(ti) + np.cos(tj)))

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


def newton_correct(chain, theta_pred, tangent, tol=1e-12, max_iter=100):
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
        theta += delta
        if np.linalg.norm(delta) < tol:
            return theta, True
    return theta, False


def run_continuation(chain, theta0, ds, s_max):
    theta = theta0.copy()
    C = chain.constraint_jacobian(theta)
    ns = null_space(C)
    if ns.shape[1] == 0:
        return []
    tangent = ns[:, 0]
    if tangent[-1] < 0: tangent = -tangent

    results = [(0.0, theta.copy(), tangent.copy())]
    s = 0.0

    while s < s_max:
        theta_pred = theta + ds * tangent
        theta_new, ok = newton_correct(chain, theta_pred, tangent)
        if not ok:
            break
        C_new = chain.constraint_jacobian(theta_new)
        ns_new = null_space(C_new)
        if ns_new.shape[1] == 0:
            break
        tn = ns_new[:, 0]
        if np.dot(tn, tangent) < 0: tn = -tn
        theta = theta_new
        tangent = tn
        s += ds
        results.append((s, theta.copy(), tangent.copy()))

    return results


def draw_rotor_chain(theta, r=1.0, a=1.0, ax=None, title='',
                      highlight_j=None, color_override=None):
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    x_pivots = np.arange(n) * a
    y_pivots = np.zeros(n)
    signs = (-1.0) ** np.arange(n)
    x_tips = x_pivots + r * np.sin(theta)
    y_tips = r * np.cos(theta) * signs

    ax.set_xlim(x_pivots[0] - 0.6 * a, x_pivots[-1] + 1.6 * a)
    ax.set_ylim(-1.5 * r, 1.5 * r)

    ax.plot([x_pivots[0] - 0.6 * a, x_pivots[-1] + 1.6 * a],
            [0, 0], 'k--', lw=0.4, alpha=0.3, zorder=1)
    ax.scatter(x_pivots, y_pivots, s=20, c='k', zorder=5)

    rotor_color = COLORS[0]
    if color_override:
        rotor_color = color_override

    for i in range(n):
        ax.plot([x_pivots[i], x_tips[i]],
                [y_pivots[i], y_tips[i]],
                'o-', color=rotor_color, lw=1.2,
                markersize=3, zorder=4)

    for i in range(n - 1):
        x1, y1 = x_tips[i], y_tips[i]
        x2, y2 = x_tips[i + 1], y_tips[i + 1]
        t = np.linspace(0, 1, 60)
        x_line = x1 + (x2 - x1) * t
        y_line = y1 + (y2 - y1) * t
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L > 1e-9:
            nx, ny = -dy / L, dx / L
            amp = min(0.06 * L, 0.05 * r)
            x_line += nx * amp * np.sin(5 * np.pi * t)
            y_line += ny * amp * np.sin(5 * np.pi * t)

        spring_color = COLORS[1]
        lw = 0.8
        if highlight_j is not None and i == highlight_j:
            spring_color = '#CC78BC'
            lw = 1.5
        ax.plot(x_line, y_line, '-', color=spring_color,
                lw=lw, zorder=3)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/a$', fontsize=10)
    ax.set_ylabel(r'$y/a$', fontsize=10)
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    ax.grid(alpha=0.2, lw=0.3)


def main():
    n = 16; rho = 1.0
    theta_bar = np.pi/2 - 0.3
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

    # Init
    theta0 = np.full(n, theta_bar)
    C0 = chain.constraint_jacobian(theta0)
    U, S, Vt = svd(C0, full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0: v0 = -v0
    theta0 += 1e-6 * v0

    for _ in range(100):
        F = chain.constraint(theta0)
        if np.linalg.norm(F) < 1e-12: break
        C = chain.constraint_jacobian(theta0)
        delta, *_ = np.linalg.lstsq(C, -F, rcond=None)
        theta0 += delta

    print("Running continuation...")
    path = run_continuation(chain, theta0, ds=0.005, s_max=20.0)
    print(f"  Path: {len(path)} points, s=[{path[0][0]:.3f}, {path[-1][0]:.3f}]")

    # Find D_13 crossing
    D13_vals = [chain.local_denominator(theta, 13) for _, theta, _ in path]
    s_vals = [s for s, _, _ in path]

    cross_idx = None
    for k in range(1, len(D13_vals)):
        if D13_vals[k-1] * D13_vals[k] < 0:
            cross_idx = k
            break

    if cross_idx:
        s1, theta1, t1 = path[cross_idx - 1]
        s2, theta2, t2 = path[cross_idx]
        D1 = D13_vals[cross_idx - 1]
        D2 = D13_vals[cross_idx]
        frac = -D1 / (D2 - D1)
        theta_cross = theta1 + frac * (theta2 - theta1)

        # Newton refine
        tangent = t1
        for _ in range(50):
            D_val = chain.local_denominator(theta_cross, 13)
            if abs(D_val) < 1e-14: break
            eps = 1e-7
            grad = np.zeros(n)
            for k in range(n):
                tp = theta_cross.copy(); tp[k] += eps
                tm = theta_cross.copy(); tm[k] -= eps
                grad[k] = (chain.local_denominator(tp, 13) -
                          chain.local_denominator(tm, 13)) / (2 * eps)
            tdg = np.dot(tangent, grad)
            if abs(tdg) < 1e-15: break
            theta_cross += -(D_val / tdg) * tangent

        s_cross = s1 + frac * (s2 - s1)
        print(f"  D_13=0 at s≈{s_cross:.4f}")

    # Find fold
    smin_CR_vals = []
    for _, theta, _ in path:
        C = chain.constraint_jacobian(theta)
        C_R = C[:, :-1]
        sv_CR = svd(C_R, compute_uv=False)
        smin_CR_vals.append(sv_CR[-1])

    fold_idx = None
    for k in range(1, len(smin_CR_vals)):
        if smin_CR_vals[k] < 1e-3 and smin_CR_vals[k] < smin_CR_vals[k-1]:
            if fold_idx is None or smin_CR_vals[k] < smin_CR_vals[fold_idx]:
                fold_idx = k
            if fold_idx is not None and k > fold_idx + 5:
                break

    if fold_idx is None:
        fold_idx = np.argmin(smin_CR_vals)

    s_fold = path[fold_idx][0]
    theta_fold = path[fold_idx][1]
    print(f"  Fold at s≈{s_fold:.4f}, σ_min(C^R)={smin_CR_vals[fold_idx]:.2e}")

    # ============================================================
    # Fig (a): Reference, D_13=0, Fold 三列对比
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), dpi=300)

    # (i) Reference state
    ax = axes[0]
    theta_ref = np.full(n, theta_bar)
    draw_rotor_chain(theta_ref, r=1.0, a=1.0, ax=ax,
                     title=fr'Reference ($s=0$)')

    # (ii) D_13 = 0
    ax = axes[1]
    draw_rotor_chain(theta_cross, r=1.0, a=1.0, ax=ax,
                     title=fr'$D_{{13}}=0$ ($s\approx{s_cross:.1f}$)',
                     highlight_j=13)

    # (iii) Driver fold
    ax = axes[2]
    draw_rotor_chain(theta_fold, r=1.0, a=1.0, ax=ax,
                     title=fr'Driver fold ($s\approx{s_fold:.1f}$)')

    plt.tight_layout()
    plt.savefig('figure_config.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure_config.png")

    # ============================================================
    # Fig (a2): 单独的参考态构型
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=300)
    draw_rotor_chain(theta_ref, r=1.0, a=1.0, ax=ax,
                     title=fr'Chain configuration at reference state ($s=0$)')

    plt.tight_layout()
    plt.savefig('figure6a.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure6a.png")

    # ============================================================
    # Fig (b): 单独的 D_13=0 构型，更大尺寸
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=300)
    draw_rotor_chain(theta_cross, r=1.0, a=1.0, ax=ax,
                     title=fr'Chain configuration at $D_{{13}}=0$ ($s\approx{s_cross:.2f}$)',
                     highlight_j=13)

    # 标注 θ_14
    x_pivots = np.arange(n) * 1.0
    signs = (-1.0) ** np.arange(n)
    x_tip_14 = x_pivots[13] + np.sin(theta_cross[13])
    y_tip_14 = np.cos(theta_cross[13]) * signs[13]
    ax.annotate(r'$\theta_{14}$ at extremum',
                xy=(x_tip_14, y_tip_14),
                xytext=(x_tip_14 + 1.5, y_tip_14 + 0.8),
                arrowprops=dict(arrowstyle='->', color='#CC78BC', lw=0.8),
                fontsize=8, color='#CC78BC')

    plt.tight_layout()
    plt.savefig('figure6b.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure6b.png")

    # ============================================================
    # Fig (c): 单独的 fold 构型
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=300)
    draw_rotor_chain(theta_fold, r=1.0, a=1.0, ax=ax,
                     title=fr'Chain configuration at driver fold ($s\approx{s_fold:.2f}$)',
                     color_override=COLORS[2])

    # 标注 θ_n
    x_tip_n = x_pivots[n-1] + np.sin(theta_fold[n-1])
    y_tip_n = np.cos(theta_fold[n-1]) * signs[n-1]
    ax.annotate(r'$\theta_n$ at fold',
                xy=(x_tip_n, y_tip_n),
                xytext=(x_tip_n + 0.5, y_tip_n - 1.0),
                arrowprops=dict(arrowstyle='->', color=COLORS[2], lw=0.8),
                fontsize=8, color=COLORS[2])

    plt.tight_layout()
    plt.savefig('figure6c.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Saved: figure6c.png")

    print("\nAll configuration figures saved.")


if __name__ == '__main__':
    main()
