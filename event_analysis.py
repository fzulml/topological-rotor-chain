"""
event_analysis.py
精确事件定位 + 正规形验证 + 力/运动传导，为文稿提供可引用的数值。
输出 event_summary.txt 供写作引用。
"""
import numpy as np
from scipy.linalg import svd, null_space
from scipy.optimize import brentq
import io, contextlib

with contextlib.redirect_stdout(io.StringIO()):
    from theory_verification import RotorChain, run, newton

N = 16; RHO = 1.0; R = 1.0
THBAR = np.pi/2 - 0.3
chain = RotorChain(n=N, r=R, a=RHO, theta_bar=THBAR)
path = run(chain, ds=0.0025, s_max=20.0)
S = np.array([p[0] for p in path])
TH = np.array([p[1] for p in path])
TT = np.array([p[2] for p in path])


def interp(sq, arr, s):
    return np.array([np.interp(s, S, arr[:, k]) for k in range(arr.shape[1])])


def F_of(s, j):                      # 折叠分母 F_j (0-based j)
    th = interp(S, TH, s)
    return chain.A_j(th, j)


def D_of(s, j):                      # 传递分母 D_j (0-based j)
    th = interp(S, TH, s)
    return chain.B_j(th, j)


lines = []
def P(*a):
    t = " ".join(str(x) for x in a)
    lines.append(t)
    print(t)


P("=" * 74)
P("A. 事件位置（Brent 精确求根）")
P("=" * 74)
P(f"{'事件':<16}{'s*':>12}{'sigma_min(C)':>16}{'sigma_min(C^R)':>17}")
P("-" * 74)
folds, transfers = {}, {}
for j in (13, 12, 11, 10, 9, 8, 7):
    # 传递事件 D_j = 0：取最靠近首个符号变化的根
    for lo, hi in zip(S[:-1], S[1:]):
        if D_of(lo, j)*D_of(hi, j) < 0:
            sT = brentq(lambda s: D_of(s, j), lo, hi, xtol=1e-13)
            break
    transfers[j] = sT
    for lo, hi in zip(S[:-1], S[1:]):
        if F_of(lo, j)*F_of(hi, j) < 0:
            sF = brentq(lambda s: F_of(s, j), lo, hi, xtol=1e-13)
            break
    folds[j] = sF
    th_T = interp(S, TH, sT); C_T = chain.C(th_T)
    th_F = interp(S, TH, sF); C_F = chain.C(th_F)
    sc = svd(C_T, compute_uv=False)[-1]
    scr = svd(C_T[:, :-1], compute_uv=False)[-1]
    fc = svd(C_F, compute_uv=False)[-1]
    fcr = svd(C_F[:, :-1], compute_uv=False)[-1]
    P(f"transfer D_{j:<2}=0 {sT:12.5f}{sc:16.6f}{scr:17.6f}")
    P(f"fold     F_{j:<2}=0 {sF:12.5f}{fc:16.6f}{fcr:17.3e}")

P("")
P("=" * 74)
P("B. 事件顺序（右端向内）与相邻间隔")
P("=" * 74)
ev = []
for j in (13, 12, 11, 10, 9, 8, 7):
    ev.append((transfers[j], f"D_{j}"))
    ev.append((folds[j], f"F_{j}"))
ev.sort()
prev = None
for s, name in ev:
    gap = f"{s-prev:8.3f}" if prev else "       -"
    P(f"  {name:>5}  s = {s:8.5f}   间隔 = {gap}")
    prev = s

P("")
P("=" * 74)
P("C. 正规形验证（首次折叠 F_13 = 0, s* = %.5f）" % folds[13])
P("=" * 74)
s_star = folds[13]
th_star = interp(S, TH, s_star)
qn_star = th_star[-1]
win = np.abs(S - s_star) < 0.06
ss = S[win] - s_star
th_win = TH[win]
smin_CR = np.array([svd(chain.C(t)[:, :-1], compute_uv=False)[-1] for t in th_win])
tn_win = np.array([t[-1] for t in TT[win]])
qn_win = th_win[:, -1] - qn_star
ok = np.abs(ss) > 1e-4
sl1 = np.polyfit(np.log(np.abs(ss[ok])), np.log(smin_CR[ok]), 1)
sl2 = np.polyfit(np.log(np.abs(ss[ok])), np.log(np.abs(tn_win[ok])), 1)
P(f"  sigma_min(C^R) ~ |s-s*|^{sl1[0]:.4f}   (R^2 = "
  f"{np.corrcoef(np.log(np.abs(ss[ok])), np.log(smin_CR[ok]))[0,1]**2:.6f})")
P(f"  dtheta_n/ds   ~ |s-s*|^{sl2[0]:.4f}   (R^2 = "
  f"{np.corrcoef(np.log(np.abs(ss[ok])), np.log(np.abs(tn_win[ok])))[0,1]**2:.6f})")
Aq = np.vstack([np.ones(ok.sum()), ss[ok]**2]).T
coef, res, *_ = np.linalg.lstsq(Aq, qn_win[ok], rcond=None)
pred = Aq @ coef
r2 = 1 - np.sum((qn_win[ok]-pred)**2)/np.sum((qn_win[ok]-qn_win[ok].mean())**2)
P(f"  theta_n - theta_n* = a + b(s-s*)^2 :  b = {coef[1]:+.4f} rad, "
  f"a = {coef[0]:+.2e} rad,  R^2 = {r2:.8f}")
Aq3 = np.vstack([np.ones(ok.sum()), ss[ok], ss[ok]**2, ss[ok]**3]).T
c3, *_ = np.linalg.lstsq(Aq3, qn_win[ok], rcond=None)
P(f"  含三次项拟合: c2 = {c3[2]:+.4f}, c3 = {c3[3]:+.4f} (三次项可忽略)")
P(f"  折叠处 |t_n| = {abs(np.interp(s_star, S, TT[:, -1])):.3e}")

P("")
P("=" * 74)
P("D. 运动/力传导（末端驱动）")
P("=" * 74)
P("  (i) 运动权限 |dtheta_n/ds|: 折叠处线性趋于零 => 末端驱动失效")
P("  (ii) 自平衡边界载荷下约束力: lambda_{n-2} = tau_n l_{n-2}/(r D_{n-2})")
P(f"{'s':>9}{'|dtheta_n/ds|':>18}{'l/|D_14|':>14}{'kappa(C^R)':>14}")
for s in (4.5, 4.8, 4.912, 5.0, 5.074, 5.2, 7.0, 7.348, 7.51):
    th = interp(S, TH, s)
    C = chain.C(th)
    CR = C[:, :-1]
    svc = svd(CR, compute_uv=False)
    tn = np.interp(s, S, TT[:, -1])
    l14 = chain.spring_length(th, 14)
    P(f"{s:9.3f}{abs(tn):18.3e}{l14/abs(chain.B_j(th, 14)):14.3e}"
      f"{svc[0]/svc[-1]:14.3e}")

# 自平衡边界载荷： tau = e_n - (t_n/t_1) e_1  （满足 tau . t = 0）
P("")
P("  自平衡载荷 tau = e_n - (t_n/t_1) e_1 下的末端约束力 lambda_14：")
P(f"{'s':>9}{'||C^T lam - tau||':>20}{'lambda_14':>14}{'解析 l_14/(r D_14)':>20}")
for s in (4.7, 4.90, 4.912, 4.95, 5.10, 7.30, 7.348, 7.40):
    th = interp(S, TH, s)
    C = chain.C(th)
    t = interp(S, TT, s); t = t/np.linalg.norm(t)
    tau = np.zeros(N); tau[-1] = 1.0; tau[0] = -t[-1]/t[0]
    lam, *_ = np.linalg.lstsq(C.T, tau, rcond=None)
    l14 = chain.spring_length(th, 14)
    P(f"{s:9.3f}{np.linalg.norm(C.T@lam-tau):20.2e}{lam[14]:14.3e}"
      f"{l14/(R*chain.B_j(th, 14)):20.3e}")

P("")
P("=" * 74)
P("E. 传递事件处 theta_{j+1} 取极值（j=13 -> theta_14）")
P("=" * 74)
sT = transfers[13]
P(f"  D_13 = 0 于 s = {sT:.5f}; dtheta_14/ds 左右符号：")
for ds_ in (-0.02, -0.005, 0.005, 0.02):
    P(f"    s = {sT+ds_:8.4f}:  dtheta_14/ds = "
      f"{np.interp(sT+ds_, S, TT[:, 13]):+12.4e}   "
      f"dtheta_15/ds = {np.interp(sT+ds_, S, TT[:, 14]):+12.4e}")

P("")
P("=" * 74)
P("F. 秩判据：整条路径上 rank(C) = n-1")
P("=" * 74)
bad = 0
for i in range(0, len(S), 20):
    C = chain.C(TH[i])
    Fv = np.array([chain.A_j(TH[i], j) for j in range(N-1)])
    Dv = np.array([chain.B_j(TH[i], j) for j in range(N-1)])
    iF = np.where(np.abs(Fv) < 1e-9)[0]
    iD = np.where(np.abs(Dv) < 1e-9)[0]
    pred = (len(iF) == 0) or (len(iD) == 0) or (iF.min() > iD.max())
    if pred != (np.linalg.matrix_rank(C, tol=1e-9) == N-1):
        bad += 1
P(f"  判据与数值秩不一致次数 = {bad} / {len(range(0, len(S), 20))}")
P(f"  路径上 sigma_min(C) 最小值 = "
  f"{min(svd(chain.C(t), compute_uv=False)[-1] for t in TH):.4f}")

with open('event_summary.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print("\n[saved] event_summary.txt")
