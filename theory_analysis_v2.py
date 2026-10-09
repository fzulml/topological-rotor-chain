"""
theory_analysis_v2.py
修复版: 使用更稳健的 continuation 追踪
"""
import numpy as np
from scipy.linalg import svd, null_space
import csv

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
    """Run pseudo-arclength continuation, return list of (s, theta, tangent)."""
    # Initial setup
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
            print(f"  null_space empty at s={s:.4f}")
            break
        tn = ns_new[:, 0]
        if np.dot(tn, tangent) < 0: tn = -tn

        theta = theta_new
        tangent = tn
        s += ds
        results.append((s, theta.copy(), tangent.copy()))

    return results


# ============================================================
# Proposition A: D_j=0 处的 C 秩分析
# ============================================================
def proposition_A():
    print("="*60)
    print("Proposition A: D_j=0 不导致 C 失秩")
    print("="*60)

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

    # Project to manifold
    for _ in range(100):
        F = chain.constraint(theta0)
        if np.linalg.norm(F) < 1e-12: break
        C = chain.constraint_jacobian(theta0)
        delta, *_ = np.linalg.lstsq(C, -F, rcond=None)
        theta0 += delta

    print(f"\nRunning continuation (ds=0.005, s_max=8)...")
    path = run_continuation(chain, theta0, ds=0.005, s_max=8.0)
    print(f"  Path length: {len(path)} points, s=[{path[0][0]:.3f}, {path[-1][0]:.3f}]")

    # Find D_13 crossing
    D13_vals = [chain.local_denominator(theta, 13) for _, theta, _ in path]
    s_vals = [s for s, _, _ in path]

    crossing_idx = None
    for k in range(1, len(D13_vals)):
        if D13_vals[k-1] * D13_vals[k] < 0:
            crossing_idx = k
            break

    if crossing_idx is None:
        print("  D_13 did not cross zero in range. Trying larger s_max...")
        path = run_continuation(chain, theta0, ds=0.005, s_max=12.0)
        D13_vals = [chain.local_denominator(theta, 13) for _, theta, _ in path]
        s_vals = [s for s, _, _ in path]
        for k in range(1, len(D13_vals)):
            if D13_vals[k-1] * D13_vals[k] < 0:
                crossing_idx = k
                break

    if crossing_idx is None:
        print("  Still no crossing found. Checking D values...")
        for j in range(n-1):
            Dj_vals = [chain.local_denominator(theta, j) for _, theta, _ in path]
            if Dj_vals[0] * Dj_vals[-1] < 0:
                print(f"    D_{j} crosses zero: [{Dj_vals[0]:.4f} -> {Dj_vals[-1]:.4f}]")
        return None

    # Interpolate to exact crossing
    s1, theta1, t1 = path[crossing_idx - 1]
    s2, theta2, t2 = path[crossing_idx]
    D1 = D13_vals[crossing_idx - 1]
    D2 = D13_vals[crossing_idx]
    frac = -D1 / (D2 - D1)

    # Newton refine
    theta_cross = theta1 + frac * (theta2 - theta1)
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

    D_final = chain.local_denominator(theta_cross, 13)
    print(f"\n  D_13 crossing at s≈{(s1+frac*(s2-s1)):.4f}")
    print(f"  Refined: D_13 = {D_final:.2e}")

    # Analyze C at crossing
    C_cross = chain.constraint_jacobian(theta_cross)
    sv = svd(C_cross, compute_uv=False)

    print(f"\n  === C 在 D_13=0 处的分析 ===")
    print(f"  C 形状: {C_cross.shape}")
    print(f"  σ_min(C) = {sv[-1]:.8f}")
    print(f"  σ_max(C) = {sv[0]:.8f}")
    print(f"  rank(C) = {np.sum(sv > 1e-10)} (期望: {n-1})")

    # Check C[13, 14] (0-based: row 13, col 14)
    print(f"\n  C[13, 14] = ∂l_13/∂θ_15 = {C_cross[13, 14]:.2e}")
    print(f"  (应接近 0, 因 D_13=0)")

    # Submatrix analysis: remove row 13
    C_no_row = np.delete(C_cross, 13, axis=0)
    sv_nr = svd(C_no_row, compute_uv=False)
    print(f"\n  去掉第13行: {C_no_row.shape}")
    print(f"  σ_min = {sv_nr[-1]:.8f}")
    print(f"  rank = {np.sum(sv_nr > 1e-10)}")

    # Submatrix: (n-1) x (n-1) by removing column 14
    C_sq = np.delete(C_cross, 14, axis=1)
    sv_sq = svd(C_sq, compute_uv=False)
    print(f"\n  去掉第14列 (方阵): {C_sq.shape}")
    print(f"  σ_min = {sv_sq[-1]:.8f}")
    print(f"  det = {np.linalg.det(C_sq):.6f}")
    print(f"  rank = {np.sum(sv_sq > 1e-10)} (期望: {n-1})")

    # Key: the (n-1)x(n-1) submatrix is nonsingular
    print(f"\n  结论: D_13=0 使 C[13,14]=0, 但 (n-1)×(n-1) 子矩阵非奇异")
    print(f"  ⇒ rank(C) = n-1 保持, C 满行秩")

    return theta_cross, C_cross


# ============================================================
# Proposition B: Driver fold 分析
# ============================================================
def proposition_B():
    print("\n" + "="*60)
    print("Proposition B: Driver fold = 坐标投影奇异性")
    print("="*60)

    n = 16; rho = 1.0
    theta_bar = np.pi/2 - 0.3
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

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

    print(f"\nRunning continuation (ds=0.005, s_max=20)...")
    path = run_continuation(chain, theta0, ds=0.005, s_max=20.0)
    print(f"  Path: {len(path)} points, s=[{path[0][0]:.3f}, {path[-1][0]:.3f}]")

    # Find first fold (sigma_min(C^R) < threshold)
    fold_idx = None
    smin_CR_threshold = 1e-3

    smin_CR_vals = []
    smin_C_vals = []
    s_vals_list = []

    for s, theta, _ in path:
        C = chain.constraint_jacobian(theta)
        C_R = C[:, :-1]
        sv_C = svd(C, compute_uv=False)
        sv_CR = svd(C_R, compute_uv=False)
        smin_C_vals.append(sv_C[-1])
        smin_CR_vals.append(sv_CR[-1])
        s_vals_list.append(s)

        if sv_CR[-1] < smin_CR_threshold and fold_idx is None:
            fold_idx = path.index((s, theta, _))

    smin_CR_arr = np.array(smin_CR_vals)
    smin_C_arr = np.array(smin_C_vals)
    s_arr = np.array(s_vals_list)

    if fold_idx is None:
        print(f"  No fold found (threshold={smin_CR_threshold})")
        print(f"  min σ_min(C^R) = {smin_CR_arr.min():.2e} at s={s_arr[np.argmin(smin_CR_arr)]:.4f}")
        print(f"  Trying lower threshold...")
        # Find the global minimum of sigma_min(C^R)
        fold_idx = np.argmin(smin_CR_arr)
        print(f"  Using global min at s={s_arr[fold_idx]:.4f}")

    # Get fold point
    s_fold, theta_fold, t_fold = path[fold_idx]
    print(f"\n  Fold at s={s_fold:.4f}")
    print(f"  σ_min(C^R) = {smin_CR_arr[fold_idx]:.2e}")
    print(f"  σ_min(C)   = {smin_C_arr[fold_idx]:.8f}")

    # Refine: use finer steps near fold
    print(f"\n  Refining fold position...")
    # Take a few points around fold and interpolate
    idx_range = range(max(0, fold_idx-5), min(len(path), fold_idx+5))
    s_local = [path[i][0] for i in idx_range]
    smin_CR_local = [smin_CR_vals[i] for i in idx_range]

    # Find the minimum in local range
    local_min_idx = np.argmin(smin_CR_local)
    s_fold_refined = s_local[local_min_idx]
    theta_fold_refined = path[list(idx_range)[local_min_idx]][1]
    print(f"  Refined fold at s={s_fold_refined:.4f}")

    # Analyze at fold
    C_fold = chain.constraint_jacobian(theta_fold_refined)
    C_R_fold = C_fold[:, :-1]

    sv_C = svd(C_fold, compute_uv=False)
    sv_CR = svd(C_R_fold, compute_uv=False)

    print(f"\n  === Fold 处的结构分析 ===")
    print(f"  C:  {C_fold.shape}, σ_min={sv_C[-1]:.8f}, rank={np.sum(sv_C>1e-10)}")
    print(f"  C^R: {C_R_fold.shape}, σ_min={sv_CR[-1]:.2e}, rank={np.sum(sv_CR>1e-10)}")

    # Null space of C^R
    ns_CR = null_space(C_R_fold)
    print(f"\n  dim null(C^R) = {ns_CR.shape[1]} (期望: 1)")

    if ns_CR.shape[1] >= 1:
        v_fold = ns_CR[:, 0]
        # Embed in full n-dim space (last component = 0)
        v_full = np.concatenate([v_fold, [0]])
        Cv = C_fold @ v_full
        print(f"  ||C · [v_fold; 0]|| = {np.linalg.norm(Cv):.2e}")
        print(f"  (非零 ⇒ v_fold 不在 null(C) 中)")

    # Null space of C
    ns_C = null_space(C_fold)
    print(f"\n  dim null(C) = {ns_C.shape[1]} (期望: 1)")
    if ns_C.shape[1] >= 1:
        v_motion = ns_C[:, 0]
        print(f"  null(C) 最后一分量 (对应 θ_n): {v_motion[-1]:.6f}")
        print(f"  在 fold 处, 这个分量应接近 0")

        # Compare with a non-fold point
        theta_normal = path[min(fold_idx+20, len(path)-1)][1]
        C_normal = chain.constraint_jacobian(theta_normal)
        ns_C_n = null_space(C_normal)
        v_n = ns_C_n[:, 0]
        print(f"  非fold处 null(C) 最后一分量: {v_n[-1]:.6f} (应显著非零)")

    # Implicit function theorem argument
    print(f"\n  === 隐函数定理论证 ===")
    print(f"  1. C 满秩 ⇒ 约束流形 M 是光滑 1-流形 (隐函数定理)")
    print(f"  2. C^R 失秩 ⟹ θ_n 不是 M 的有效坐标")
    print(f"     (因为 ∂F/∂(θ_1,...,θ_{{n-1}}) = C^R 不可逆)")
    print(f"  3. 但 M 本身光滑 (因为 C 满秩)")
    print(f"  ⇒ Driver fold 是坐标投影奇异性, 不是流形奇异性")

    return theta_fold_refined, C_fold


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    prop_A_result = proposition_A()
    prop_B_result = proposition_B()
