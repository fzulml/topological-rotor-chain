"""
theory_analysis.py
理论推导的数值验证：
1. Proposition A: D_j=0 时 C 的子矩阵行列式
2. Proposition B: Driver fold 处的隐函数定理验证
3. D_j 与 ∂l_j/∂θ_{j+2} 精确关系
4. σ_min(C) 下界分析
"""
import numpy as np
from scipy.linalg import svd, null_space
import csv

# ============================================================
# RotorChain (from figure2_4_long_trace.py)
# ============================================================
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


# ============================================================
# 验证 D_j 与 ∂l_j/∂θ_{j+2} 的精确关系
# ============================================================
def verify_Dj_relation():
    print("="*60)
    print("验证: D_j 与 ∂l_j/∂θ_{j+2} 的精确关系")
    print("="*60)

    n = 16; rho = 1.0
    theta_bar = np.pi/2 - 0.3
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)
    theta = np.full(n, theta_bar + 0.1 * np.random.randn(n))

    # 重新投影到约束流形
    for _ in range(200):
        F = chain.constraint(theta)
        C = chain.constraint_jacobian(theta)
        delta, *_ = np.linalg.lstsq(C, -F, rcond=None)
        theta += delta
        if np.linalg.norm(delta) < 1e-14:
            break

    print(f"\n约束残差: {np.linalg.norm(chain.constraint(theta)):.2e}")
    print(f"\n精确关系: ∂l_j/∂θ_(j+2) = (r/l_j) * D_j")
    print(f"即: D_j = (l_j/r) * ∂l_j/∂θ_(j+2)")
    print(f"\n逐项验证 (r=1):")

    C = chain.constraint_jacobian(theta)
    max_err = 0
    for j in range(n - 1):
        l_j = chain.spring_length(theta, j)
        Dj = chain.local_denominator(theta, j)
        partial = C[j, j+1]  # ∂l_j/∂θ_(j+2)
        Dj_computed = l_j * partial / chain.r
        err = abs(Dj - Dj_computed)
        max_err = max(max_err, err)
        if j < 5 or j >= n-3:
            print(f"  j={j:2d}: D_j={Dj:+.8f}, l_j*∂l/∂θ={Dj_computed:+.8f}, err={err:.2e}")

    print(f"\n最大误差: {max_err:.2e}")
    print("结论: D_j = (l_j/r) * ∂l_j/∂θ_(j+2)  ✓")


# ============================================================
# Proposition A: D_j=0 时 C 的秩分析
# ============================================================
def proposition_A_analysis():
    print("\n" + "="*60)
    print("Proposition A: D_j=0 不导致 C 失秩")
    print("="*60)

    n = 16; rho = 1.0
    theta_bar = np.pi/2 - 0.3
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

    # 运行 continuation 到 D_13=0
    theta = np.full(n, theta_bar)
    C0 = chain.constraint_jacobian(theta)
    U, S, Vt = svd(C0, full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0: v0 = -v0
    theta = theta + 1e-6 * v0

    # 投影到流形
    for _ in range(50):
        F = chain.constraint(theta)
        if np.linalg.norm(F) < 1e-12: break
        C = chain.constraint_jacobian(theta)
        delta, *_ = np.linalg.lstsq(C, -F, rcond=None)
        theta += delta

    # Pseudo-arclength continuation
    ds = 0.005; s_max = 6.0
    tangent = null_space(chain.constraint_jacobian(theta))[:, 0]
    if tangent[-1] < 0: tangent = -tangent
    s = 0.0

    print(f"\n追踪路径到 D_13 = 0...")
    crossing_found = False
    D13_prev = chain.local_denominator(theta, 13)

    while s < s_max:
        theta_pred = theta + ds * tangent
        for _ in range(80):
            F = chain.constraint(theta_pred)
            C = chain.constraint_jacobian(theta_pred)
            J = np.vstack([C, tangent.reshape(1, -1)])
            r = -np.concatenate([F, [tangent @ (theta_pred - theta)]])
            try:
                delta = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J, r, rcond=None)
            theta_pred += delta
            if np.linalg.norm(delta) < 1e-11: break

        tn = null_space(chain.constraint_jacobian(theta_pred))[:, 0]
        if tn is None: break
        if np.dot(tn, tangent) < 0: tn = -tn
        tangent = tn
        theta = theta_pred
        s += ds

        D13_now = chain.local_denominator(theta, 13)
        if D13_prev * D13_now < 0 and not crossing_found:
            # 线性插值找到 D_13=0 的精确位置
            frac = -D13_prev / (D13_now - D13_prev)
            theta_cross = theta  # 近似
            crossing_found = True
            print(f"  D_13 在 s={s:.4f} 附近穿零")

            # 精确 Newton 到 D_13=0
            for _ in range(50):
                D13_val = chain.local_denominator(theta, 13)
                C = chain.constraint_jacobian(theta)
                # D_13 对 theta 的梯度
                eps = 1e-8
                grad_D13 = np.zeros(n)
                for k in range(n):
                    tp = theta.copy(); tp[k] += eps
                    tm = theta.copy(); tm[k] -= eps
                    grad_D13[k] = (chain.local_denominator(tp, 13) -
                                  chain.local_denominator(tm, 13)) / (2*eps)

                # 沿 tangent 方向调整使 D_13=0
                t_dot_g = np.dot(tangent, grad_D13)
                if abs(t_dot_g) < 1e-15: break
                theta += -(D13_val / t_dot_g) * tangent
                if abs(D13_val) < 1e-12: break

            D13_final = chain.local_denominator(theta, 13)
            print(f"  精确位置: D_13 = {D13_final:.2e}")

            # 在 D_13=0 处分析 C
            C_at_cross = chain.constraint_jacobian(theta)
            sv_C = svd(C_at_cross, compute_uv=False)

            print(f"\n  在 D_13=0 处:")
            print(f"    σ_min(C) = {sv_C[-1]:.8f}")
            print(f"    σ_max(C) = {sv_C[0]:.8f}")
            print(f"    条件数 = {sv_C[0]/sv_C[-1]:.4f}")
            print(f"    rank(C) = {np.sum(sv_C > 1e-10)} (应为 {n-1})")

            # 验证: C 的第 14 列 (j+2=15, one-based) 的第 13 行
            print(f"\n  C[13, 14] (= ∂l_13/∂θ_15) = {C_at_cross[13, 14]:.2e}")
            print(f"  这应该接近 0 (因为 D_13=0)")

            # 验证: 去掉第 13 行后，剩余子矩阵的秩
            C_sub = np.delete(C_at_cross, 13, axis=0)  # 去掉第13行
            sv_sub = svd(C_sub, compute_uv=False)
            print(f"\n  去掉第13行后的子矩阵:")
            print(f"    形状: {C_sub.shape}")
            print(f"    σ_min = {sv_sub[-1]:.8f}")
            print(f"    rank = {np.sum(sv_sub > 1e-10)}")

            # 验证: 去掉第14列后，剩余子矩阵的秩 (即 C^R 不含第13行的部分)
            C_no_col = np.delete(C_at_cross, 14, axis=1)
            sv_nc = svd(C_no_col, compute_uv=False)
            print(f"\n  去掉第14列后的子矩阵 (n-1 × n-1):")
            print(f"    形状: {C_no_col.shape}")
            print(f"    σ_min = {sv_nc[-1]:.8f}")
            print(f"    rank = {np.sum(sv_nc > 1e-10)} (应为 {n-1})")

            break
        D13_prev = D13_now


# ============================================================
# Proposition B: Driver fold 处的隐函数定理分析
# ============================================================
def proposition_B_analysis():
    print("\n" + "="*60)
    print("Proposition B: Driver fold = 坐标投影奇异性")
    print("="*60)

    n = 16; rho = 1.0
    theta_bar = np.pi/2 - 0.3
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)

    # 运行 continuation 找到第一个 fold
    theta = np.full(n, theta_bar)
    C0 = chain.constraint_jacobian(theta)
    U, S, Vt = svd(C0, full_matrices=True)
    v0 = Vt[-1]
    if v0[-1] < 0: v0 = -v0
    theta = theta + 1e-6 * v0

    for _ in range(50):
        F = chain.constraint(theta)
        if np.linalg.norm(F) < 1e-12: break
        C = chain.constraint_jacobian(theta)
        delta, *_ = np.linalg.lstsq(C, -F, rcond=None)
        theta += delta

    ds = 0.005; s_max = 20.0
    tangent = null_space(chain.constraint_jacobian(theta))[:, 0]
    if tangent[-1] < 0: tangent = -tangent
    s = 0.0

    fold_found = False
    prev_smin_CR = None

    print(f"\n追踪路径到第一个 driver fold...")

    while s < s_max:
        theta_pred = theta + ds * tangent
        for _ in range(80):
            F = chain.constraint(theta_pred)
            C = chain.constraint_jacobian(theta_pred)
            J = np.vstack([C, tangent.reshape(1, -1)])
            r = -np.concatenate([F, [tangent @ (theta_pred - theta)]])
            try:
                delta = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J, r, rcond=None)
            theta_pred += delta
            if np.linalg.norm(delta) < 1e-11: break

        tn = null_space(chain.constraint_jacobian(theta_pred))[:, 0]
        if tn is None: break
        if np.dot(tn, tangent) < 0: tn = -tn
        tangent = tn
        theta = theta_pred
        s += ds

        C = chain.constraint_jacobian(theta)
        C_R = C[:, :-1]
        sv_C = svd(C, compute_uv=False)
        sv_CR = svd(C_R, compute_uv=False)

        if sv_CR[-1] < 1e-4 and not fold_found:
            fold_found = True
            print(f"  Fold 在 s={s:.4f} 附近, σ_min(C^R)={sv_CR[-1]:.2e}")

            # 精确定位 fold: 用更小步长
            theta_back = theta - ds * tangent  # 回退一步
            ds_fine = ds / 100

            for _ in range(200):
                theta_pred = theta_back + ds_fine * tangent
                for _ in range(80):
                    F = chain.constraint(theta_pred)
                    C = chain.constraint_jacobian(theta_pred)
                    J = np.vstack([C, tangent.reshape(1, -1)])
                    r = -np.concatenate([F, [tangent @ (theta_pred - theta_back)]])
                    try:
                        delta = np.linalg.solve(J, r)
                    except:
                        delta, *_ = np.linalg.lstsq(J, r, rcond=None)
                    theta_pred += delta
                    if np.linalg.norm(delta) < 1e-12: break

                tn = null_space(chain.constraint_jacobian(theta_pred))[:, 0]
                if tn is None: break
                if np.dot(tn, tangent) < 0: tn = -tn
                tangent_fine = tn

                C = chain.constraint_jacobian(theta_pred)
                C_R = C[:, :-1]
                sv_CR_new = svd(C_R, compute_uv=False)[-1]
                sv_C_new = svd(C, compute_uv=False)[-1]

                theta_back = theta_pred

                if sv_CR_new < 1e-6:
                    print(f"  精确 fold: s≈{s:.4f}, σ_min(C^R)={sv_CR_new:.2e}")
                    print(f"  σ_min(C) at fold = {sv_C_new:.8f}")

                    # 分析 fold 处的结构
                    print(f"\n  在 fold 处:")
                    print(f"    C 满秩? rank(C) = {np.sum(svd(C, compute_uv=False) > 1e-10)} (n-1={n-1})")
                    print(f"    C^R 失秩? rank(C^R) = {np.sum(svd(C_R, compute_uv=False) > 1e-10)} (应为 {n-2})")

                    # C^R 的零空间方向
                    ns_CR = null_space(C_R)
                    print(f"    dim null(C^R) = {ns_CR.shape[1]} (应为 1)")

                    # 验证: C^R 的零向量不在 C 的零空间中
                    if ns_CR.shape[1] > 0:
                        v_fold = ns_CR[:, 0]
                        # 补上最后一个分量 0
                        v_full = np.concatenate([v_fold, [0]])
                        Cv = C @ v_full
                        print(f"    ||C * [v_fold; 0]|| = {np.linalg.norm(Cv):.2e}")
                        print(f"    (如果 C 满秩, 这个应非零, 说明 v_fold 不在 null(C) 中)")

                        # C 的零空间
                        ns_C = null_space(C)
                        print(f"    dim null(C) = {ns_C.shape[1]} (应为 1)")
                        if ns_C.shape[1] > 0:
                            v_motion = ns_C[:, 0]
                            # 验证 v_motion 的最后一个分量
                            print(f"    null(C) 的最后一个分量: {v_motion[-1]:.6f}")
                            print(f"    (在 fold 处, 这个分量应接近 0, 表示 θ_n 不再是有效参数)")

                    break

            break


# ============================================================
# σ_min(C) 下界分析
# ============================================================
def sigma_min_C_bounds():
    print("\n" + "="*60)
    print("σ_min(C) 下界分析")
    print("="*60)

    # 读取参数扫描数据
    import os
    csv_path = 'cascade_data/full_scan.csv'
    if not os.path.exists(csv_path):
        print(f"数据文件不存在: {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    smin_values = []
    rho_values = []
    tb_values = []
    n_values = []

    for row in rows:
        try:
            smin = float(row['smin_C_min'])
            rho = float(row['rho'])
            tb = float(row['theta_bar_deg'])
            n_val = int(row['n'])
            if smin > 1e-6 and tb != 90:
                smin_values.append(smin)
                rho_values.append(rho)
                tb_values.append(tb)
                n_values.append(n_val)
        except (ValueError, TypeError):
            continue

    smin_arr = np.array(smin_values)
    rho_arr = np.array(rho_values)
    tb_arr = np.array(tb_values)
    n_arr = np.array(n_values)

    print(f"\n有效数据点: {len(smin_arr)}")
    print(f"σ_min(C) 统计:")
    print(f"  最小值: {smin_arr.min():.6f}")
    print(f"  最大值: {smin_arr.max():.6f}")
    print(f"  均值: {smin_arr.mean():.6f}")
    print(f"  中位数: {np.median(smin_arr):.6f}")

    # 找最小值对应的参数
    idx_min = np.argmin(smin_arr)
    print(f"\n最小 σ_min(C) 对应参数:")
    print(f"  n={n_arr[idx_min]}, ρ={rho_arr[idx_min]}, θ̄={tb_arr[idx_min]}°")
    print(f"  σ_min(C) = {smin_arr[idx_min]:.6f}")

    # 按参数分组统计
    print(f"\n按 ρ 分组的 σ_min(C) 最小值:")
    for rho_val in sorted(set(rho_arr.tolist())):
        mask = rho_arr == rho_val
        print(f"  ρ={rho_val:.1f}: min={smin_arr[mask].min():.6f}, "
              f"max={smin_arr[mask].max():.6f}")

    print(f"\n按 θ̄ 分组的 σ_min(C) 最小值:")
    for tb_val in sorted(set(tb_arr.tolist())):
        mask = tb_arr == tb_val
        print(f"  θ̄={tb_val:.1f}°: min={smin_arr[mask].min():.6f}, "
              f"max={smin_arr[mask].max():.6f}")

    print(f"\n按 n 分组的 σ_min(C) 最小值:")
    for n_val in sorted(set(n_arr.tolist())):
        mask = n_arr == n_val
        print(f"  n={n_val}: min={smin_arr[mask].min():.6f}, "
              f"max={smin_arr[mask].max():.6f}")

    # 下界估计
    print(f"\n下界估计:")
    print(f"  全局下界: σ_min(C) > {smin_arr.min():.4f}")
    print(f"  论文中报告的 0.26 对应 θ̄=60°, ρ=0.8")
    print(f"  验证: ", end="")
    mask = (tb_arr == 60) & (rho_arr == 0.8)
    if mask.any():
        print(f"σ_min(C) at θ̄=60°, ρ=0.8: {smin_arr[mask].min():.6f}")
    else:
        print("未找到对应数据")


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)
    verify_Dj_relation()
    proposition_A_analysis()
    proposition_B_analysis()
    sigma_min_C_bounds()
