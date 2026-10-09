"""
theory_verification.py
对拟写入文稿的结构性结论做逐条数值核验：
 (V1) Jacobian 为二对角结构；两条分母族 A_j（对角）与 B_j（超对角）的定义
 (V2) 零模递推 A_j v_j = B_j v_{j+1} 与显式零模 v_j = prod(A_i/B_i)
 (V3) 互补余子式行列式因子化: det(C 删去第 m 列) ∝ (prod_{i<m} A_i)(prod_{i>=m} B_i)
 (V4) det(C^R) ∝ prod_i A_i  ——驱动折叠 = A_j 的零点
 (V5) 秩判据: rank(C)=n-1 <=> min{i:A_i=0} > max{i:B_j=0}
 (V6) 传递奇异处零模节点位置
 (V7) 折叠附近的标度: sigma_min(C^R) 线性、dtheta_n/ds 线性、theta_n-theta_n* 二次
 (V8) 力传导: 末端单位力矩下约束力范数 ∝ 1/prod B_j
"""
import numpy as np
from scipy.linalg import svd, null_space
import csv, os

plt = None

class RotorChain:
    def __init__(self, n, r=1.0, a=1.0, theta_bar=np.pi/2 - 0.3):
        self.n = n; self.r = r; self.a = a
        self.theta_bar = theta_bar
        self.rho = a / r
        self.l_bar = np.sqrt(a**2 + 4*r**2*np.cos(theta_bar)**2)

    def spring_length(self, theta, j):
        ti = theta[j]; tj = theta[j+1]
        return np.hypot(self.a + self.r*(np.sin(tj) - np.sin(ti)),
                        self.r*(np.cos(ti) + np.cos(tj)))

    def constraint(self, theta):
        return np.array([self.spring_length(theta, j) - self.l_bar
                         for j in range(self.n - 1)])

    def C(self, theta):
        n = self.n; C = np.zeros((n - 1, n))
        for i in range(n - 1):
            l = self.spring_length(theta, i)
            ti = theta[i]; tj = theta[i+1]
            C[i, i]   = (-self.a*self.r*np.cos(ti)
                         - self.r**2*np.sin(ti+tj)) / l
            C[i, i+1] = (+self.a*self.r*np.cos(tj)
                         - self.r**2*np.sin(ti+tj)) / l
        return C

    # ---- 两条分母族（0-based j，对应 theta[j], theta[j+1]）----
    def A_j(self, theta, j):
        ti = theta[j]; tj = theta[j+1]
        return self.rho*np.cos(ti) + np.sin(ti+tj)

    def B_j(self, theta, j):
        ti = theta[j]; tj = theta[j+1]
        return self.rho*np.cos(tj) - np.sin(ti+tj)


def newton(chain, theta_pred, tangent, tol=1e-12, itmax=100):
    theta = theta_pred.copy()
    for _ in range(itmax):
        F = chain.constraint(theta)
        C = chain.C(theta)
        J = np.vstack([C, tangent.reshape(1, -1)])
        rhs = -np.concatenate([F, [tangent @ (theta - theta_pred)]])
        delta, *_ = np.linalg.lstsq(J, rhs, rcond=None)
        theta = theta + delta
        if np.linalg.norm(delta) < tol:
            return theta, True
    return theta, False


def tangent_dir(chain, theta):
    ns = null_space(chain.C(theta))
    if ns.shape[1] == 0:
        return None
    t = ns[:, 0]
    return t


def run(chain, ds=0.0025, s_max=8.0):
    """返回 (s, theta) 序列"""
    n = chain.n
    theta = np.full(n, chain.theta_bar)
    C0 = chain.C(theta)
    U, S, Vt = svd(C0, full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0: v0 = -v0
    theta = theta + 1e-6*v0
    for _ in range(60):
        F = chain.constraint(theta)
        if np.linalg.norm(F) < 1e-13: break
        d, *_ = np.linalg.lstsq(chain.C(theta), -F, rcond=None)
        theta = theta + d
    t = tangent_dir(chain, theta)
    if t[-1] < 0: t = -t
    out = [(0.0, theta.copy(), t.copy())]
    s = 0.0
    while s < s_max:
        pred = theta + ds*t
        th, ok = newton(chain, pred, t)
        if not ok: break
        tn = tangent_dir(chain, th)
        if tn is None: break
        if tn @ t < 0: tn = -tn
        theta, t = th, tn
        s += ds
        out.append((s, theta.copy(), t.copy()))
    return out


def main():
    n = 16; rho = 1.0; r = 1.0
    chain = RotorChain(n=n, r=r, a=rho, theta_bar=np.pi/2 - 0.3)
    path = run(chain, ds=0.0025, s_max=8.0)
    print(f"path points: {len(path)}, s_max = {path[-1][0]:.4f}")

    # ---------- V1: 二对角结构与 A_j, B_j 定义 ----------
    th = path[500][1]
    C = chain.C(th)
    err_A = err_B = 0.0
    for i in range(n-1):
        l = chain.spring_length(th, i)
        err_A = max(err_A, abs(C[i, i]   + r*chain.A_j(th, i)/l))
        err_B = max(err_B, abs(C[i, i+1] - r*chain.B_j(th, i)/l))
    nnz_per_row = [(C[i] != 0).sum() for i in range(n-1)]
    print(f"V1  [C[j,j]= -r A_j/l] max err = {err_A:.2e} ; "
          f"[C[j,j+1]= r B_j/l] max err = {err_B:.2e}")
    print(f"V1  每行非零个数: {set(nnz_per_row)}  (1 表示二对角)")

    # ---------- V2: 零模递推与显式公式 ----------
    err_rec = 0.0
    for (s, theta, t) in path[::400]:
        for j in range(n-1):
            err_rec = max(err_rec, abs(chain.A_j(theta, j)*t[j]
                                       - chain.B_j(theta, j)*t[j+1]))
    s0, th0, t0 = path[0]
    prod = np.ones(n)
    for j in range(n-1):
        prod[j+1] = prod[j]*chain.A_j(th0, j)/chain.B_j(th0, j)
    v_expl = prod/np.linalg.norm(prod)
    v_num = t0/np.linalg.norm(t0)
    if v_expl @ v_num < 0: v_expl = -v_expl
    print(f"V2  row equation A_j t_j = B_j t_{j+1}: max err = {err_rec:.2e}")
    print(f"V2  显式公式 vs 数值零模: max|dv| = {np.max(np.abs(v_expl-v_num)):.2e}")

    # ---------- V3/V4: 余子式行列式因子化 & det C^R ----------
    worst3 = worst4 = 0.0
    for (s, theta, t) in path[::300]:
        C = chain.C(theta)
        l = np.array([chain.spring_length(theta, i) for i in range(n-1)])
        A = np.array([chain.A_j(theta, j) for j in range(n-1)])
        B = np.array([chain.B_j(theta, j) for j in range(n-1)])
        for m in range(n):
            sub = np.delete(C, m, axis=1)
            d_num = np.linalg.det(sub)
            d_fac = ((-1.0)**m) * (r**(n-1)) * np.prod(A[:m]) * np.prod(B[m:]) / np.prod(l)
            # 归一化比较（比例因子只允许 ±1 差）
            if abs(d_fac) > 1e-10:
                rel = abs(d_num - d_fac)/abs(d_fac)
                worst3 = max(worst3, rel)
            else:
                worst3 = max(worst3, abs(d_num))
        CR = C[:, :-1]
        d_CR = np.linalg.det(CR)
        f_CR = ((-1.0)**n) * (r**(n-1)) * np.prod(A) / np.prod(l)
        worst4 = max(worst4, abs(d_CR - f_CR)/max(abs(f_CR), 1e-30))
    print(f"V3  删列余子式因子化公式: 最大相对偏差 = {worst3:.2e}")
    print(f"V4  det(C^R) = (-1)^n r^(n-1) prod A / prod l: 最大相对偏差 = {worst4:.2e}")

    # ---------- V5: 秩判据 ----------
    bad = 0
    for (s, theta, t) in path[::200]:
        C = chain.C(theta)
        A = np.array([chain.A_j(theta, j) for j in range(n-1)])
        B = np.array([chain.B_j(theta, j) for j in range(n-1)])
        za = np.where(np.abs(A) < 1e-10)[0]
        zb = np.where(np.abs(B) < 1e-10)[0]
        pred_full = (len(za) == 0) or (len(zb) == 0) or (za.min() > zb.max())
        rank_ok = (np.linalg.matrix_rank(C, tol=1e-9) == n-1)
        if pred_full != rank_ok:
            bad += 1
    print(f"V5  秩判据与数值秩不一致的次数 = {bad}")

    # ---------- V6: 传递奇异处零模节点 ----------
    Bvals = np.array([[chain.B_j(th, j) for j in range(n-1)]
                      for (s, th, t) in path])
    s_vals = np.array([s for (s, th, t) in path])
    k = 13
    idx = np.argmin(np.abs(Bvals[:, k]))
    s_x, th_x, t_x = path[idx]
    print(f"V6  B_{k} 穿零于 s = {s_x:.4f}; 零模 |t| 剖面（末尾 6 个分量）:")
    print("    ", np.array2string(np.abs(t_x[-6:])/np.abs(t_x).max(),
                                  precision=3))
    node = int(np.argmin(np.abs(t_x)/np.abs(t_x).max()))
    print(f"V6  归一化振幅最小分量 index = {node} (0-based), "
          f"即第 {node+1} 号转子(1-based); 相对幅值 = "
          f"{np.abs(t_x[node])/np.abs(t_x).max():.2e}")

    # ---------- V7: 折叠附近标度 ----------
    Avals = np.array([[chain.A_j(th, j) for j in range(n-1)]
                      for (s, th, t) in path])
    kk = int(np.argmin(np.abs(Avals).min(axis=0)))
    col = Avals[:, kk]
    sign_ch = np.where(np.diff(np.sign(col)))[0]
    print(f"V7  首个 A_j 穿零的 j = {kk}, 穿零序号 = "
          f"{sign_ch[:3] if len(sign_ch) else 'none'}")
    if len(sign_ch):
        i0 = sign_ch[0]
        s1, s2 = s_vals[i0], s_vals[i0+1]
        a1, a2 = col[i0], col[i0+1]
        s_star = s1 - a1*(s2-s1)/(a2-a1)
        print(f"V7  A_{kk} = 0 于 s* = {s_star:.5f}")
        # sigma_min(C^R) 与 dtheta_n/ds, theta_n 的局部标度
        win = np.abs(s_vals - s_star) < 0.05
        ss = s_vals[win] - s_star
        smin_CR = np.array([svd(chain.C(th)[:, :-1], compute_uv=False)[-1]
                            for (s, th, t) in path])[win]
        dpsidt = np.array([t[-1] for (s, th, t) in path])[win]
        qn = np.array([th[-1] for (s, th, t) in path])[win]
        ok = np.abs(ss) > 1e-6
        sl_smin = np.polyfit(np.log(np.abs(ss[ok])), np.log(smin_CR[ok]), 1)[0]
        sl_dpsid = np.polyfit(np.log(np.abs(ss[ok])), np.log(np.abs(dpsidt[ok])), 1)[0]
        sl_qn = np.polyfit(np.log(np.abs(ss[ok])), np.log(np.abs(qn[ok]-qn[np.argmin(np.abs(ss))])+1e-18), 1)[0]
        print(f"V7  log-log 斜率: sigma_min(C^R) ~ |s-s*|^{sl_smin:.3f}, "
              f"d theta_n/ds ~ |s-s*|^{sl_dpsid:.3f}, theta_n-theta_n* ~ |s-s*|^{sl_qn:.3f}")

    # ---------- V8: 力传导（末端单位力矩下约束力范数）----------
    print("V8  末端单位力矩下约束力范数随 s 的变化（Transfer 前后）:")
    for (s, theta, t) in path[::200]:
        C = chain.C(theta)
        b = np.zeros(n); b[-1] = 1.0
        lam, *_ = np.linalg.lstsq(C.T, b, rcond=None)
        B = np.array([chain.B_j(theta, j) for j in range(n-1)])
        print(f"    s={s:5.2f}  ||C^T lam - b||={np.linalg.norm(C.T@lam-b):.1e}"
              f"  ||lam||={np.linalg.norm(lam):9.3e}"
              f"  1/prod|B|={1.0/np.prod(np.abs(B)):9.3e}")


if __name__ == '__main__':
    main()
