"""
cascade_data_full.py
完整收集拓扑转子链的级联事件数据。

不给出任何解析公式。只报告数值观察。

用法：
  # 先跑小范围测试（约 5 分钟）
  python cascade_data_full.py --test

  # 完整扫描（约 2-4 小时）
  python cascade_data_full.py --full
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.linalg import svd, null_space
import matplotlib.pyplot as plt
import csv
import os
import time
import argparse

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


# ============================================================
# 核心类：RotorChain
# ============================================================
class RotorChain:
    """拓扑转子链。"""

    def __init__(self, n, r=1.0, a=1.0, theta_bar=np.pi/2 - 0.3):
        self.n = n
        self.r = r
        self.a = a
        self.theta_bar = theta_bar
        self.rho = a / r
        self.l_bar = np.sqrt(
            a**2 + 4 * r**2 * np.cos(theta_bar)**2
        )

    def spring_length(self, theta, j):
        """约束 j 的长度（零基约束索引 j=0,...,n-2）。"""
        ti = theta[j]
        tj = theta[j + 1]
        return np.hypot(
            self.a + self.r * (np.sin(tj) - np.sin(ti)),
            self.r * (np.cos(ti) + np.cos(tj))
        )

    def constraint(self, theta):
        """所有约束的残差：F_j = l_j - l_bar。"""
        return np.array([
            self.spring_length(theta, j) - self.l_bar
            for j in range(self.n - 1)
        ])

    def constraint_jacobian(self, theta):
        """约束雅可比 C[i,j] = ∂F_i/∂θ_j，形状 (n-1, n)。"""
        n = self.n
        C = np.zeros((n - 1, n))
        for i in range(n - 1):
            l = self.spring_length(theta, i)
            ti = theta[i]
            tj = theta[i + 1]
            C[i, i] = (
                -self.a * self.r * np.cos(ti)
                - self.r**2 * np.sin(ti + tj)
            ) / l
            C[i, i + 1] = (
                +self.a * self.r * np.cos(tj)
                - self.r**2 * np.sin(ti + tj)
            ) / l
        return C

    def local_denominator(self, theta, j):
        """D_j = ρ cos θ_{j+1} - sin(θ_j + θ_{j+1})。"""
        ti = theta[j]
        tj = theta[j + 1]
        return self.rho * np.cos(tj) - np.sin(ti + tj)

    def gain(self):
        """参考态传递增益。"""
        tb = self.theta_bar
        num = self.rho * np.cos(tb) + np.sin(2 * tb)
        den = self.rho * np.cos(tb) - np.sin(2 * tb)
        if abs(den) < 1e-14:
            return np.inf
        return num / den


# ============================================================
# 辅助函数
# ============================================================
def tangent_direction(chain, theta):
    """约束流形在 theta 处的单位切向量（C 的零空间）。"""
    C = chain.constraint_jacobian(theta)
    V = null_space(C)
    if V.shape[1] == 0:
        return None
    v = V[:, 0]
    return v / np.linalg.norm(v)


def newton_on_constraint(chain, theta_pred, tangent,
                         tol=1e-11, max_iter=80):
    """在约束流形上做 Newton 校正。"""
    theta = theta_pred.copy()
    for k in range(max_iter):
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


def initialize_on_manifold(chain):
    """从参考态出发，沿零模方向小扰动，投影回约束流形。"""
    n = chain.n
    theta = np.full(n, chain.theta_bar)
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
    return theta


# ============================================================
# 核心：长距离伪弧长延拓
# ============================================================
def long_trace(chain, ds=0.005, s_max=40.0, direction=+1):
    """
    伪弧长延拓。

    返回：
      history : list of dict，每步的状态
      events  : list of dict，D_j = 0 事件
    """
    n = chain.n
    theta = initialize_on_manifold(chain)

    tangent = tangent_direction(chain, theta)
    if tangent is None:
        return [], []
    if tangent[-1] * direction < 0:
        tangent = -tangent

    history = []
    events = []
    D_prev = None
    s = 0.0

    while s < s_max:
        # 记录当前状态
        C = chain.constraint_jacobian(theta)
        C_R = C[:, :-1]
        sv_C = svd(C, compute_uv=False)
        sv_CR = svd(C_R, compute_uv=False)
        D = np.array([
            chain.local_denominator(theta, j)
            for j in range(n - 1)
        ])

        history.append({
            's': s,
            'q': theta[-1] - chain.theta_bar,
            'sigma_min_C': float(sv_C[-1]),
            'sigma_min_CR': float(sv_CR[-1]),
            'D': D.copy(),
            'theta': theta.copy(),
        })

        # 检测 D_j 穿零
        if D_prev is not None:
            for j in range(n - 1):
                if D_prev[j] * D[j] < 0:
                    already = any(e['j'] == j for e in events)
                    if not already:
                        s_prev = history[-2]['s']
                        s_curr = history[-1]['s']
                        q_prev = history[-2]['q']
                        q_curr = history[-1]['q']
                        D_p = D_prev[j]
                        D_c = D[j]
                        frac = -D_p / (D_c - D_p)
                        events.append({
                            'j': j,
                            's': s_prev + frac * (s_curr - s_prev),
                            'q': q_prev + frac * (q_curr - q_prev),
                            'order': len(events),
                        })
        D_prev = D

        # 流形奇异检测
        if sv_C[-1] < 1e-8:
            break

        # 前进一步
        theta_pred = theta + ds * tangent
        theta_new, ok = newton_on_constraint(
            chain, theta_pred, tangent
        )
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

    return history, events


# ============================================================
# 单次运行
# ============================================================
def run_one(n, rho, theta_bar_deg, s_max=40.0, ds=0.005):
    """运行单个参数组合，返回摘要。"""
    theta_bar = theta_bar_deg * np.pi / 180
    try:
        chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=theta_bar)
        history, events = long_trace(
            chain, ds=ds, s_max=s_max, direction=+1
        )

        # 计算事件间隔
        if len(events) >= 2:
            intervals = [
                events[k]['s'] - events[k-1]['s']
                for k in range(1, len(events))
            ]
            ds_mean = float(np.mean(intervals))
            ds_std = float(np.std(intervals))
        else:
            ds_mean = None
            ds_std = None

        # q 饱和值
        if history:
            tail = history[int(0.8 * len(history)):]
            q_sat = float(np.mean([h['q'] for h in tail]))
            q_std = float(np.std([h['q'] for h in tail]))
        else:
            q_sat = None
            q_std = None

        # σ_min(C) 的下界
        smin_C_min = (
            float(min(h['sigma_min_C'] for h in history))
            if history else None
        )

        return {
            'n': n,
            'rho': rho,
            'theta_bar_deg': theta_bar_deg,
            'n_events': len(events),
            'ds_mean': ds_mean,
            'ds_std': ds_std,
            'q_sat': q_sat,
            'q_std': q_std,
            'smin_C_min': smin_C_min,
            'gain': chain.gain(),
            'abs_gain': abs(chain.gain()),
            'events': events,
            'status': 'ok',
        }
    except Exception as e:
        return {
            'n': n,
            'rho': rho,
            'theta_bar_deg': theta_bar_deg,
            'status': f'failed: {e}',
        }


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='小范围测试（约 5 分钟）')
    parser.add_argument('--full', action='store_true',
                        help='完整扫描（约 2-4 小时）')
    args = parser.parse_args()

    if args.test:
        n_values = [16]
        rho_values = [1.0]
        tb_values = [72.8]
        s_max = 20.0
    elif args.full:
        n_values = [8, 12, 16, 24, 32]
        rho_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        tb_values = [40, 60, 72.8, 90, 110, 130]
        s_max = 40.0
    else:
        print("Usage: python cascade_data_full.py --test | --full")
        return

    os.makedirs('cascade_data', exist_ok=True)

    print("=" * 70)
    print("CASCADE DATA COLLECTION")
    print("=" * 70)
    print(f"n_values: {n_values}")
    print(f"rho_values: {rho_values}")
    print(f"theta_bar (deg): {tb_values}")
    print(f"s_max = {s_max}")
    print()

    all_data = []
    t_start = time.time()

    total = len(n_values) * len(rho_values) * len(tb_values)
    count = 0

    for n in n_values:
        for rho in rho_values:
            for tb_deg in tb_values:
                count += 1
                t0 = time.time()
                result = run_one(n, rho, tb_deg, s_max=s_max)
                dt = time.time() - t0

                if result['status'] == 'ok':
                    print(f"[{count:3d}/{total}] n={n:2d}, "
                          f"ρ={rho:.2f}, θ̄={tb_deg:5.1f}° → "
                          f"N_events={result['n_events']:2d}, "
                          f"Δs={result['ds_mean']}, "
                          f"q_sat={result['q_sat']}, "
                          f"({dt:.1f}s)")
                else:
                    print(f"[{count:3d}/{total}] n={n:2d}, "
                          f"ρ={rho:.2f}, θ̄={tb_deg:5.1f}° → "
                          f"{result['status']}")

                all_data.append(result)

    # 保存到 CSV
    csv_path = 'cascade_data/full_scan.csv'
    fieldnames = ['n', 'rho', 'theta_bar_deg', 'n_events',
                  'ds_mean', 'ds_std', 'q_sat', 'q_std',
                  'smin_C_min', 'abs_gain', 'status']
    with open(csv_path, 'w', encoding='utf-8-sig',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_data:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"\nSaved: {csv_path}")

    # ---------- 绘图 ----------
    valid = [d for d in all_data
             if d.get('status') == 'ok'
             and d.get('ds_mean') is not None]

    if len(valid) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

        gains = [d['abs_gain'] for d in valid]
        ds_vals = [d['ds_mean'] for d in valid]
        ds_errs = [d['ds_std'] for d in valid]
        ns = [d['n'] for d in valid]

        # (a) Δs vs |g|
        ax = axes[0, 0]
        ax.errorbar(gains, ds_vals, yerr=ds_errs, fmt='o',
                    color='#0173B2', markersize=6, capsize=3)
        ax.set_xlabel(r'$|g|$', fontsize=11)
        ax.set_ylabel(r'$\Delta s$', fontsize=11)
        ax.grid(alpha=0.3, lw=0.3)
        ax.text(0.02, 0.92, '(a)', transform=ax.transAxes,
                fontsize=12, va='top')

        # (b) q_sat vs |g|
        ax = axes[0, 1]
        qsat_vals = [d['q_sat'] for d in valid]
        ax.plot(gains, qsat_vals, 's', color='#029E73',
                markersize=6)
        ax.set_xlabel(r'$|g|$', fontsize=11)
        ax.set_ylabel(r'$q_{\rm sat}$', fontsize=11)
        ax.grid(alpha=0.3, lw=0.3)
        ax.text(0.02, 0.92, '(b)', transform=ax.transAxes,
                fontsize=12, va='top')

        # (c) N_events vs n
        ax = axes[1, 0]
        unique_n = sorted(set(ns))
        for tb_deg in sorted(set(d['theta_bar_deg'] for d in valid)):
            data_tb = [d for d in valid
                       if d['theta_bar_deg'] == tb_deg]
            ns_tb = [d['n'] for d in data_tb]
            ne_tb = [d['n_events'] for d in data_tb]
            ax.plot(ns_tb, ne_tb, 'o-', lw=1.2, markersize=5,
                    label=f'θ̄={tb_deg}°')
        ax.set_xlabel('chain length $n$', fontsize=11)
        ax.set_ylabel('number of events', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, lw=0.3)
        ax.text(0.02, 0.92, '(c)', transform=ax.transAxes,
                fontsize=12, va='top')

        # (d) Δs vs θ̄
        ax = axes[1, 1]
        for rho_val in sorted(set(d['rho'] for d in valid)):
            data_rho = [d for d in valid if d['rho'] == rho_val]
            tb_arr = [d['theta_bar_deg'] for d in data_rho]
            ds_arr = [d['ds_mean'] for d in data_rho]
            ax.plot(tb_arr, ds_arr, 'o-', lw=1.2, markersize=5,
                    label=f'ρ={rho_val}')
        ax.set_xlabel(r'$\bar\theta$ (deg)', fontsize=11)
        ax.set_ylabel(r'$\Delta s$', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, lw=0.3)
        ax.text(0.02, 0.92, '(d)', transform=ax.transAxes,
                fontsize=12, va='top')

        plt.tight_layout()
        plt.savefig('cascade_data/full_scan.png', dpi=300,
                    bbox_inches='tight', pad_inches=0.03)
        plt.close()
        print("Saved: cascade_data/full_scan.png")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Total combinations: {total}")
    print(f"Successful: {sum(1 for d in all_data if d.get('status') == 'ok')}")


if __name__ == '__main__':
    main()
