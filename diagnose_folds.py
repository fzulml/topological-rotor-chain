"""
diagnose_folds.py
精确诊断：σ_min(C^R) 的深谷、|t_n|、两条分母族之间的关系。
"""
import numpy as np
from scipy.linalg import svd, null_space
import sys
sys.path.insert(0, '.')
from theory_verification import RotorChain, run


def main():
    n = 16; rho = 1.0
    chain = RotorChain(n=n, r=1.0, a=rho, theta_bar=np.pi/2 - 0.3)
    path = run(chain, ds=0.0025, s_max=20.0)
    s = np.array([p[0] for p in path])
    TH = np.array([p[1] for p in path])
    T = np.array([p[2] for p in path])

    A = np.array([[chain.A_j(th, j) for j in range(n-1)] for th in TH])
    B = np.array([[chain.B_j(th, j) for j in range(n-1)] for th in TH])
    L = np.array([[chain.spring_length(th, j) for j in range(n-1)]
                  for th in TH])

    # 零模显式乘积
    ratio = A[:, :] / B[:, :]
    logprod = np.cumsum(np.log(np.abs(ratio)), axis=1)      # log|v_{j+1}/v_1|
    cum = np.hstack([np.zeros((len(s), 1)), logprod])        # log|v_j/v_1|
    tn = T[:, -1]

    smin_CR = np.array([svd(chain.C(th)[:, :-1], compute_uv=False)[-1]
                        for th in TH])
    det_CR = np.array([np.linalg.det(chain.C(th)[:, :-1]) for th in TH])
    prodA = np.prod(A[:, :n-1], axis=1) / np.prod(L[:, :n-1], axis=1)

    print("=" * 78)
    print("1) sigma_min(C^R) 的局部极小（深谷）")
    print("=" * 78)
    idx = []
    for i in range(1, len(s)-1):
        if smin_CR[i] <= smin_CR[i-1] and smin_CR[i] < smin_CR[i+1] \
           and smin_CR[i] < 1e-3:
            idx.append(i)
    for i in idx[:12]:
        jmin = int(np.argmin(np.abs(A[i, :n-1])))
        print(f" s={s[i]:6.3f}  sigma_min(C^R)={smin_CR[i]:9.2e}"
              f"  |t_n|={abs(tn[i]):9.2e}"
              f"  |det C^R|={abs(det_CR[i]):9.2e}"
              f"  prod A/l={abs(prodA[i]):9.2e}"
              f"  min|A_j|={abs(A[i, jmin]):.3e}(j={jmin})"
              f"  min|B_j|={np.min(np.abs(B[i, :n-1])):.3e}"
              f"(j={int(np.argmin(np.abs(B[i, :n-1])))})")

    print()
    print("=" * 78)
    print("2) A_j 与 B_j 的零点（符号变化）位置")
    print("=" * 78)
    for j in range(n-1):
        zs = []
        col = A[:, j]
        for i in range(len(s)-1):
            if col[i]*col[i+1] < 0:
                zs.append(s[i] - col[i]*(s[i+1]-s[i])/(col[i+1]-col[i]))
        zb = []
        colb = B[:, j]
        for i in range(len(s)-1):
            if colb[i]*colb[i+1] < 0:
                zb.append(s[i] - colb[i]*(s[i+1]-s[i])/(colb[i+1]-colb[i]))
        if zs or zb:
            print(f" j={j:2d}  A_j=0 at {['%.3f'%z for z in zs[:4]]}"
                  f"   B_j=0 at {['%.3f'%z for z in zb[:4]]}")

    print()
    print("=" * 78)
    print("3) 关键位置细看（穿零事件前后）")
    print("=" * 78)
    for s_target in (2.04, 4.60, 5.06, 7.51):
        i = int(np.argmin(np.abs(s - s_target)))
        print(f"\n -- s = {s[i]:.4f} --")
        print(f"    sigma_min(C)   = {svd(chain.C(TH[i]), compute_uv=False)[-1]:.4f}")
        print(f"    sigma_min(C^R) = {smin_CR[i]:.3e}")
        print(f"    |t_n|          = {abs(tn[i]):.3e}")
        print(f"    det C^R        = {det_CR[i]:.3e}   prod A/l = {prodA[i]:.3e}")
        jm = np.argsort(np.abs(A[i, :n-1]))[:3]
        print(f"    最小的 |A_j|: " + ", ".join(
            f"A_{j}={A[i,j]:+.3e}" for j in jm))
        print(f"    log10|v_j/v_1| (每 2 个一段): "
              + np.array2string(cum[i, ::2]/np.log(10), precision=2))

    print()
    print("=" * 78)
    print("4) 累积传递增益与 sigma_min(C^R) 的相关性")
    print("=" * 78)
    gain = logprod[:, -1]/np.log(10)     # log10 prod|A_i/B_i|
    print(f"   corr(log10|t_n|, log10 prod|A_i/B_i|) = "
          f"{np.corrcoef(np.log10(np.abs(tn)), gain)[0,1]:.4f}")
    print(f"   corr(log10 sigma_min(C^R), log10 prod|A_i/B_i|) = "
          f"{np.corrcoef(np.log10(smin_CR), gain)[0,1]:.4f}")
    print(f"   corr(log10 sigma_min(C^R), log10|det C^R|) = "
          f"{np.corrcoef(np.log10(smin_CR), np.log10(np.abs(det_CR)))[0,1]:.4f}")
    ii = np.argsort(gain)[-5:]
    print("   增益最大的 5 个位置：")
    for i in sorted(ii):
        print(f"     s={s[i]:6.3f}  log10 prod|A/B|={gain[i]:7.2f}"
              f"  sigma_min(C^R)={smin_CR[i]:.2e}  |t_n|={abs(tn[i]):.2e}")


if __name__ == '__main__':
    main()
