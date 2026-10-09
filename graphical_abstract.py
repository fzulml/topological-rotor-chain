"""
graphical_abstract.py

生成投稿用图形摘要（Graphical Abstract），单张高分辨率图片。

输出：
  graphical_abstract.png   — 300 dpi, RGB
  graphical_abstract.tif   — 300 dpi, LZW 压缩（Springer 首选格式）

规格依据 Springer Nature 图形摘要要求：
  - 单张图，左→右、上→下阅读
  - >= 300 dpi，RGB
  - 字体 Arial
  - 标注简洁，避免冗余文字与箭头

内容四格（左→右、上→下）：
  (a) 等静定转子链几何 + 局域零模（|g| = 3.196, n = 16）
  (b) 两类事件沿路径的交替"阶梯"（传递 D_j = 0 / 驱动折叠 F_j = 0）
  (c) 力放大 prod|F_j/D_j| 与运动权限 |dtheta_n/ds|
      —— 传递事件处内力发散，折叠事件处末端驱动失效
  (d) sigma_min(C) 与 sigma_min(C^R) —— C 始终满秩，运动不终止

数据来自伪弧长延拓（与正文 Section 4.1 同一实现），
首次运行会计算并缓存到 ga_trace_cache.csv。

用法：
  python graphical_abstract.py
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.linalg import svd, null_space
import csv
import os

# ============================================================
# 字体：注册并锁定 Arial（Springer 图形摘要要求）
# ============================================================
_ARIAL = '/System/Library/Fonts/Supplemental/Arial.ttf'
_ARIAL_B = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
_ARIAL_I = '/System/Library/Fonts/Supplemental/Arial Italic.ttf'
_ARIAL_BI = '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf'
for _p in (_ARIAL, _ARIAL_B, _ARIAL_I, _ARIAL_BI):
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['legend.fontsize'] = 8

BLUE = '#0173B2'
ORANGE = '#DE8F05'
MAGENTA = '#CC78BC'
GREEN = '#029E73'
GREY = '#4D4D4D'
LIGHT = '#C8C8C8'

CACHE = 'ga_trace_cache.csv'


# ============================================================
# 模型与伪弧长延拓（与 figure2_4_long_trace.py 一致）
# ============================================================
class RotorChain:
    def __init__(self, n, r=1.0, a=1.0, theta_bar=np.pi / 2 - 0.3):
        self.n = n
        self.r = r
        self.a = a
        self.theta_bar = theta_bar
        self.rho = a / r
        self.l_bar = np.sqrt(a**2 + 4 * r**2 * np.cos(theta_bar)**2)

    def spring_length(self, theta, j):
        ti, tj = theta[j], theta[j + 1]
        return np.hypot(self.a + self.r * (np.sin(tj) - np.sin(ti)),
                        self.r * (np.cos(ti) + np.cos(tj)))

    def constraint(self, theta):
        return np.array([self.spring_length(theta, j) - self.l_bar
                         for j in range(self.n - 1)])

    def constraint_jacobian(self, theta):
        n = self.n
        C = np.zeros((n - 1, n))
        for i in range(n - 1):
            l = self.spring_length(theta, i)
            ti, tj = theta[i], theta[i + 1]
            C[i, i] = (-self.a * self.r * np.cos(ti)
                       - self.r**2 * np.sin(ti + tj)) / l
            C[i, i + 1] = (+self.a * self.r * np.cos(tj)
                           - self.r**2 * np.sin(ti + tj)) / l
        return C

    def denoms(self, theta):
        """返回 (F, D) 两组分母，长度 n-1。"""
        ti, tj = theta[:-1], theta[1:]
        sn = np.sin(ti + tj)
        return self.rho * np.cos(ti) + sn, self.rho * np.cos(tj) - sn


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


def run_trace(n=16, rho=1.0, theta_bar=np.pi / 2 - 0.3,
              ds=0.005, s_max=20.0):
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

    theta = np.full(n, theta_bar)
    _, _, Vt = svd(chain.constraint_jacobian(theta), full_matrices=True)
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
        sv_C = svd(C, compute_uv=False)
        sv_CR = svd(C[:, :-1], compute_uv=False)
        Fd, Dd = chain.denoms(theta)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(Fd / Dd)
        row = {'s': s,
               'sigma_min_C': float(sv_C[-1]),
               'sigma_min_CR': float(sv_CR[-1]),
               'dtheta_n_ds': float(abs(tangent[-1])),
               'force_amp': float(np.prod(ratio))}
        for j in range(n - 1):
            row[f'F_{j}'] = float(Fd[j])
            row[f'D_{j}'] = float(Dd[j])
        history.append(row)

        if sv_C[-1] < 1e-8:
            break
        theta_pred = theta + ds * tangent
        theta_new, ok = newton_on_constraint(chain, theta_pred, tangent)
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


def save_cache(history, path):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)


def load_cache(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    data = {}
    for col in rows[0].keys():
        try:
            data[col] = np.array([float(r[col]) for r in rows])
        except ValueError:
            data[col] = np.array([r[col] for r in rows])
    return data


def get_data():
    if os.path.exists(CACHE):
        print(f'Loading cache: {CACHE}')
        return load_cache(CACHE)
    print('Running arclength continuation (first run only)...')
    hist = run_trace()
    save_cache(hist, CACHE)
    print(f'Cached -> {CACHE}  ({len(hist)} steps)')
    return load_cache(CACHE)


def crossings(s, y):
    out = []
    for k in range(1, len(y)):
        if y[k - 1] * y[k] < 0:
            frac = -y[k - 1] / (y[k] - y[k - 1])
            out.append(s[k - 1] + frac * (s[k] - s[k - 1]))
    return np.array(out)


# ============================================================
# 转子链示意（沿用正文 Fig.1 画法）
# 限制范围按坐标框的真实宽高比自动适配，保证图元撑满面板
# ============================================================
def draw_chain(ax, n_rot=7, r=1.0, a=1.0):
    theta = np.full(n_rot, np.pi / 2 - 0.3)
    xp = np.arange(n_rot) * a
    signs = (-1.0) ** np.arange(n_rot)
    xt = xp + r * np.sin(theta)
    yt = r * np.cos(theta) * signs

    fig = ax.figure
    bb = ax.get_position()
    box_ratio = ((bb.width * fig.get_figwidth())
                 / (bb.height * fig.get_figheight()))
    y_span = 2.42
    x_span = box_ratio * y_span
    xc = xp.mean() + 0.05 * a
    ax.set_xlim(xc - x_span / 2, xc + x_span / 2)
    ax.set_ylim(-y_span / 2, y_span / 2)

    ax.plot([xc - x_span / 2, xc + x_span / 2], [0, 0],
            ls='--', color=LIGHT, lw=0.5, zorder=1)
    ax.scatter(xp, np.zeros(n_rot), s=11, c='k', zorder=5,
               linewidths=0)
    for i in range(n_rot):
        ax.plot([xp[i], xt[i]], [0, yt[i]], '-', color=GREY,
                lw=1.5, zorder=4, solid_capstyle='round')
    for i in range(n_rot - 1):
        x1, y1, x2, y2 = xt[i], yt[i], xt[i + 1], yt[i + 1]
        t = np.linspace(0, 1, 80)
        xl = x1 + (x2 - x1) * t
        yl = y1 + (y2 - y1) * t
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L > 1e-9:
            nx, ny = -dy / L, dx / L
            amp = min(0.075 * L, 0.068 * r)
            xl = xl + nx * amp * np.sin(6 * np.pi * t)
            yl = yl + ny * amp * np.sin(6 * np.pi * t)
        ax.plot(xl, yl, '-', color=ORANGE, lw=0.9, zorder=3)

    ax.set_aspect('equal')
    ax.axis('off')


# ============================================================
# 主图
# ============================================================
def build_figure():
    data = get_data()
    s = data['s']
    smin_C = data['sigma_min_C']
    smin_CR = data['sigma_min_CR']
    tn = data['dtheta_n_ds']
    famp = data['force_amp']

    F_cols = sorted([c for c in data if c.startswith('F_')],
                    key=lambda x: int(x.split('_')[1]))
    D_cols = sorted([c for c in data if c.startswith('D_')],
                    key=lambda x: int(x.split('_')[1]))

    D_ev, F_ev = {}, {}
    for c in D_cols:
        z = crossings(s, data[c])
        if len(z):
            D_ev[int(c.split('_')[1])] = z[0]
    for c in F_cols:
        z = crossings(s, data[c])
        if len(z):
            F_ev[int(c.split('_')[1])] = z[0]

    print('transfer events at', np.round(sorted(D_ev.values()), 3))
    print('fold     events at', np.round(sorted(F_ev.values()), 3))

    # ---- 零模（n = 16，局域在受驱动一端；按"离驱动端的距离"排序）----
    chain = RotorChain(n=16)
    _, _, Vt = svd(chain.constraint_jacobian(np.full(16, chain.theta_bar)),
                   full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0:
        v0 = -v0
    mode16 = np.abs(v0) / np.max(np.abs(v0))
    mode_plot = mode16[::-1]        # 使曲线自左向右衰减
    idx_plot = np.arange(1, 17)

    # ============================================================
    fig = plt.figure(figsize=(7.09, 5.15), dpi=300)
    gs = GridSpec(2, 2, figure=fig,
                  left=0.116, right=0.962, top=0.815, bottom=0.088,
                  wspace=0.55, hspace=0.70)

    ga = gs[0, 0].subgridspec(2, 1, height_ratios=[1.45, 1.0],
                              hspace=0.60)
    ax_a1 = fig.add_subplot(ga[0])
    ax_a2 = fig.add_subplot(ga[1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_cr = ax_c.twinx()
    ax_d = fig.add_subplot(gs[1, 1])

    BOX = dict(facecolor='white', edgecolor='none', alpha=0.78,
               pad=1.4)

    # ------------------------------------------------------------
    # (a) 转子链 + 局域零模
    # ------------------------------------------------------------
    draw_chain(ax_a1, n_rot=7)
    ax_a2.semilogy(idx_plot, mode_plot, 'o-', color=BLUE,
                   lw=1.0, markersize=2.7,
                   markerfacecolor=BLUE, markeredgecolor='white',
                   markeredgewidth=0.4)
    ax_a2.set_xlabel(r'rotor index from the driven end',
                     labelpad=1.5)
    ax_a2.set_ylabel(r'$|v_i|/\max|v|$', labelpad=1.5)
    ax_a2.set_xlim(0.4, 16.6)
    ax_a2.set_ylim(3e-17, 4)
    ax_a2.set_yticks([1e-15, 1e-10, 1e-5, 1e0])
    ax_a2.grid(alpha=0.28, lw=0.35, which='major')
    ax_a2.text(0.05, 0.42, r'localized zero mode', fontsize=8.4,
               color=BLUE, transform=ax_a2.transAxes)
    ax_a2.text(0.05, 0.16, r'$|g| = 3.196$', fontsize=8.4,
               color=BLUE, transform=ax_a2.transAxes)

    # ------------------------------------------------------------
    # (b) 事件阶梯
    # ------------------------------------------------------------
    jD = np.array([j for j in sorted(D_ev, reverse=True)])
    sD = np.array([D_ev[j] for j in jD])
    jF = np.array([j for j in sorted(F_ev, reverse=True)])
    sF = np.array([F_ev[j] for j in jF])

    ax_b.plot(sD, jD, 'o', color=BLUE, ms=4.6,
              markeredgecolor='white', markeredgewidth=0.5,
              label=r'transfer  $D_j = 0$', zorder=4)
    ax_b.plot(sF, jF, 's', color=ORANGE, ms=4.2,
              markeredgecolor='white', markeredgewidth=0.5,
              label=r'fold  $F_j = 0$', zorder=4)
    for jj in sorted(D_ev):
        if (jj + 1) in F_ev and F_ev[jj + 1] > D_ev[jj]:
            ax_b.annotate(
                '', xy=(F_ev[jj + 1], jj + 1), xytext=(D_ev[jj], jj),
                arrowprops=dict(arrowstyle='-|>', lw=0.7,
                                color=LIGHT, shrinkA=3.4, shrinkB=3.4))

    ax_b.set_xlabel(r'arclength $s$', labelpad=1.5)
    ax_b.set_ylabel(r'constraint index $j$', labelpad=1.5)
    ax_b.set_xlim(-0.4, 20.4)
    ax_b.set_ylim(6.2, 14.3)
    ax_b.set_yticks([7, 8, 9, 10, 11, 12, 13])
    ax_b.grid(alpha=0.28, lw=0.35)
    ax_b.legend(fontsize=8, loc='lower left', frameon=True,
                framealpha=0.94, borderpad=0.4,
                handletextpad=0.4, labelspacing=0.4)
    ax_b.text(0.975, 0.965, 'never coincide',
              transform=ax_b.transAxes, fontsize=8.4,
              ha='right', va='top', color=GREY)

    # ------------------------------------------------------------
    # (c) 力放大 vs 运动权限
    # ------------------------------------------------------------
    for t in sorted(D_ev.values()):
        ax_c.axvline(t, color=MAGENTA, ls=':', lw=0.5, alpha=0.30,
                     zorder=1)
    for t in sorted(F_ev.values()):
        ax_cr.axvline(t, color=GREEN, ls=':', lw=0.5, alpha=0.30,
                      zorder=1)

    l1, = ax_c.semilogy(s, famp, '-', color=MAGENTA, lw=1.15, zorder=3)
    l2, = ax_cr.semilogy(s, tn, '-', color=GREEN, lw=1.15, zorder=3)

    ax_c.set_xlabel(r'arclength $s$', labelpad=1.5)
    ax_c.set_ylabel(r'$\prod_j |F_j/D_j|$', color=MAGENTA, labelpad=1.5)
    ax_cr.set_ylabel(r'$|d\theta_n/ds|$', color=GREEN, labelpad=1.0)
    ax_c.tick_params(axis='y', colors=MAGENTA)
    ax_cr.tick_params(axis='y', colors=GREEN)
    ax_c.set_xlim(-0.4, 20.4)
    ax_c.set_ylim(1e-5, 1e17)
    ax_cr.set_ylim(1e-11, 1e1)
    ax_c.set_yticks([1e-5, 1e0, 1e5, 1e10, 1e15])
    ax_cr.set_yticks([1e-10, 1e-7, 1e-4, 1e-1])
    ax_c.grid(alpha=0.28, lw=0.35, which='major', zorder=0)
    ax_c.text(0.97, 0.945, 'transfer: force diverges',
              transform=ax_c.transAxes, fontsize=8.2, color=MAGENTA,
              va='center', ha='right', bbox=BOX, zorder=5)
    ax_cr.text(0.03, 0.055, 'fold: actuator stalls',
               transform=ax_cr.transAxes, fontsize=8.2, color=GREEN,
               va='center', ha='left', bbox=BOX, zorder=5)

    # ------------------------------------------------------------
    # (d) sigma_min(C) vs sigma_min(C^R)
    # ------------------------------------------------------------
    ax_d.semilogy(s, smin_C, '-', color=BLUE, lw=1.25,
                  label=r'$\sigma_{\min}(C)$')
    ax_d.semilogy(s, smin_CR, '-', color=ORANGE, lw=1.0,
                  label=r'$\sigma_{\min}(C^R)$')
    ax_d.axhline(0.42, color=BLUE, ls='--', lw=0.7, alpha=0.85)
    ax_d.fill_between([-0.4, 20.4], 0.42, 1e4, color=BLUE,
                      alpha=0.07, lw=0)
    ax_d.set_xlabel(r'arclength $s$', labelpad=1.5)
    ax_d.set_ylabel(r'$\sigma_{\min}$', labelpad=1.5)
    ax_d.set_xlim(-0.4, 20.4)
    ax_d.set_ylim(1e-17, 1e4)
    ax_d.grid(alpha=0.28, lw=0.35, which='major')
    ax_d.legend(fontsize=8, loc='lower left', frameon=True,
                framealpha=0.94, borderpad=0.4,
                handletextpad=0.4, labelspacing=0.4)
    ax_d.text(0.50, 0.978,
              r'$\sigma_{\min}(C) > 0.42$' + '\n' + 'full rank, no stop',
              transform=ax_d.transAxes, fontsize=7.5, color=BLUE,
              ha='center', va='top', linespacing=1.4)

    # ------------------------------------------------------------
    # 标题与面板标注
    # ------------------------------------------------------------
    fig.text(0.5, 0.955,
             'Two alternating singularity families, '
             'and the motion that never terminates',
             ha='center', va='center', fontsize=12.5,
             fontweight='bold', color='#1A1A1A')
    fig.text(0.5, 0.893,
             'Isostatic rotor chains: transfer and driver-fold events '
             'split the kinetostatic labour; the constraint manifold '
             'stays smooth',
             ha='center', va='center', fontsize=8.6, color=GREY)

    def tag(ax, letter):
        p = ax.get_position()
        fig.text(p.x0 - 0.035, p.y1 + 0.028, letter,
                 fontsize=11, fontweight='bold', color='#1A1A1A')

    tag(ax_a1, '(a)')
    tag(ax_b, '(b)')
    tag(ax_c, '(c)')
    tag(ax_d, '(d)')

    return fig


def to_rgb(src, dst, dpi=300, **kw):
    from PIL import Image
    im = Image.open(src)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        if im.mode == 'RGBA':
            bg.paste(im, mask=im.split()[3])
        else:
            bg.paste(im)
        im = bg
    im.save(dst, dpi=(dpi, dpi), **kw)


def main():
    fig = build_figure()
    tmp_png = 'graphical_abstract.png'
    fig.savefig(tmp_png, dpi=300, facecolor='white')
    plt.close(fig)

    to_rgb(tmp_png, tmp_png)
    to_rgb(tmp_png, 'graphical_abstract.tif',
           compression='tiff_lzw')

    from PIL import Image
    for f in ['graphical_abstract.png', 'graphical_abstract.tif']:
        im = Image.open(f)
        print(f'{f}: {im.size[0]} x {im.size[1]} px, '
              f'{im.mode}, {os.path.getsize(f)/1e6:.2f} MB')
    print('\nDone.')


if __name__ == '__main__':
    main()
