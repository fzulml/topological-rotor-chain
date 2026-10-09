"""
figure_events.py
生成事件结构四面板图（fig:events，随 §5.4 The event staircase 排版）：
 (a) 事件阶梯图：D_j=0（传递事件）与 F_j=0（折叠事件）沿弧长的排列
 (b) 折叠正规形：sigma_min(C^R) 对数-对数标度（斜率 1）
 (c) theta_n 的二次极值（折叠正规形）
 (d) 运动权限 |dtheta_n/ds| 与内约束力放大 prod|F_j/D_j|
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import brentq
import io, contextlib

with contextlib.redirect_stdout(io.StringIO()):
    from theory_verification import RotorChain, run

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7

COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']

N = 16; RHO = 1.0; THBAR = np.pi/2 - 0.3
chain = RotorChain(n=N, r=1.0, a=RHO, theta_bar=THBAR)
path = run(chain, ds=0.0025, s_max=20.0)
S = np.array([p[0] for p in path])
TH = np.array([p[1] for p in path])
TT = np.array([p[2] for p in path])


def th_of(s):
    return np.array([np.interp(s, S, TH[:, k]) for k in range(N)])


def Ff(s, j):
    return chain.A_j(th_of(s), j)


def Df(s, j):
    return chain.B_j(th_of(s), j)


# ---------- 事件定位 ----------
folds, transfers = {}, {}
for j in range(N-1):
    for lo, hi in zip(S[:-1], S[1:]):
        if Df(lo, j)*Df(hi, j) < 0:
            transfers[j] = brentq(lambda x: Df(x, j), lo, hi, xtol=1e-13)
            break
    for lo, hi in zip(S[:-1], S[1:]):
        if Ff(lo, j)*Ff(hi, j) < 0:
            folds[j] = brentq(lambda x: Ff(x, j), lo, hi, xtol=1e-13)
            break

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), dpi=300)

# ---------- (a) 事件阶梯 ----------
ax = axes[0, 0]
ax.set_title('(a)', loc='left', fontsize=11, fontweight='bold', pad=4)
for j, s in transfers.items():
    ax.plot(s, j, 'o', ms=4.5, mfc=COLORS[0], mec='white', mew=0.4)
for j, s in folds.items():
    ax.plot(s, j, 's', ms=4.0, mfc=COLORS[1], mec='white', mew=0.4)
for j, s in transfers.items():
    if j in folds:
        ax.annotate('', xy=(folds[j], j), xytext=(s, j),
                    arrowprops=dict(arrowstyle='-', lw=0.6,
                                    color='gray', alpha=0.8))
ax.plot([], [], 'o', ms=4.5, mfc=COLORS[0], mec='white', label=r'$D_j=0$ (transfer)')
ax.plot([], [], 's', ms=4.0, mfc=COLORS[1], mec='white', label=r'$F_j=0$ (fold)')
ax.set_xlabel(r'arclength $s$')
ax.set_ylabel(r'constraint index $j$')
ax.set_ylim(6, 14)
ax.legend(loc='upper left', handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
ax.grid(alpha=0.25, lw=0.3)

# ---------- (b) 折叠正规形: sigma_min(C^R) ~ |s-s*| ----------
ax = axes[0, 1]
ax.set_title('(b)', loc='left', fontsize=11, fontweight='bold', pad=4)
for k, (j, s_star) in enumerate(list(folds.items())[:3]):
    win = np.abs(S - s_star) < 0.05
    ss = S[win] - s_star
    smin = np.array([svd(chain.C(t)[:, :-1], compute_uv=False)[-1]
                     for t in TH[win]])
    ok = np.abs(ss) > 1e-5
    ax.loglog(np.abs(ss[ok]), smin[ok], 'o', ms=1.6,
              color=COLORS[k], label=fr'$F_{{{j}}}=0$')
    p = np.polyfit(np.log(np.abs(ss[ok])), np.log(smin[ok]), 1)
    xg = np.linspace(np.abs(ss[ok]).min(), np.abs(ss[ok]).max(), 20)
    ax.loglog(xg, np.exp(p[1])*xg**p[0], '-', lw=0.8,
              color=COLORS[k], alpha=0.9)
ax.loglog([], [], 'k-', lw=0.8, label='linear fit (slope $\\approx 1$)')
ax.set_xlabel(r'$|s-s^{*}|$')
ax.set_ylabel(r'$\sigma_{\min}(C^{R})$')
ax.legend(loc='upper left', handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
ax.grid(alpha=0.25, lw=0.3, which='both')

# ---------- (c) theta_n 的二次极值 ----------
ax = axes[1, 0]
ax.set_title('(c)', loc='left', fontsize=11, fontweight='bold', pad=4)
s_star = folds[13]
th_star = th_of(s_star)[-1]
win = np.abs(S - s_star) < 0.06
ss = S[win] - s_star
qn = TH[win][:, -1] - th_star
ax.plot(ss**2, qn, 'o', ms=1.8, color=COLORS[0], label=r'$\theta_n(s)$')
A = np.vstack([np.ones(win.sum()), ss**2]).T
coef, *_ = np.linalg.lstsq(A, qn, rcond=None)
xg = np.linspace(0, ss.max()**2, 50)
ax.plot(xg, coef[0] + coef[1]*xg, '-', lw=1.0, color=COLORS[1],
        label=fr'fit $b={coef[1]:.4f}$ rad')
ax.set_xlabel(r'$(s-s^{*})^{2}$')
ax.set_ylabel(r'$\theta_n-\theta_n^{*}$ (rad)')
ax.legend(loc='upper right', handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
ax.grid(alpha=0.25, lw=0.3)

# ---------- (d) 运动权限 vs 力放大 ----------
ax = axes[1, 1]
ax.set_title('(d)', loc='left', fontsize=11, fontweight='bold', pad=4)
app = np.array([abs(t[-1]) for t in TT])
Fv = np.array([[chain.A_j(t, j) for j in range(N-1)] for t in TH])
Dv = np.array([[chain.B_j(t, j) for j in range(N-1)] for t in TH])
ampl = np.prod(np.abs(Fv/Dv), axis=1)
ax.semilogy(S, np.clip(app, 1e-7, None), '-', lw=1.0, color=COLORS[0],
            label=r'$|d\theta_n/ds|$ (motion)')
ax.semilogy(S, np.clip(ampl, None, 1e12), '-', lw=1.0, color=COLORS[3],
            label=r'$\prod_j|F_j/D_j|$ (force)')
for j, s in folds.items():
    if s < 20:
        ax.axvline(s, color=COLORS[1], lw=0.5, ls=':', alpha=0.8)
for j, s in transfers.items():
    ax.axvline(s, color='gray', lw=0.5, ls='-.', alpha=0.6)
ax.set_xlabel(r'arclength $s$')
ax.set_ylabel(r'transmission coefficient')
ax.legend(loc='lower left', handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
ax.grid(alpha=0.25, lw=0.3, which='both')

plt.tight_layout()
plt.savefig('figure4_events.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)
print("Saved: figure4_events.png")
print("folds:", {k: round(v, 5) for k, v in folds.items()})
print("transfers:", {k: round(v, 5) for k, v in transfers.items()})
