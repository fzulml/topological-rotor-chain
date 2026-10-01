"""
figure5_param_scan.py (v4, final)
修复：图例整数化、x 轴刻度精简、θ̄ 值精度。
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
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


def read_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    columns = reader.fieldnames
    data = {}
    for col in columns:
        values = [row[col] for row in rows]
        try:
            data[col] = np.array([float(v) if v != '' else np.nan
                                   for v in values])
        except (ValueError, TypeError):
            data[col] = np.array(values)
    return data


def setup_log_axis(ax):
    """精简 log 坐标轴刻度。"""
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())


def main():
    csv_path = 'cascade_data/full_scan.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    data = read_csv(csv_path)
    n_arr = data['n'].astype(int)
    rho_arr = data['rho']
    tb_arr = data['theta_bar_deg']
    n_events = data['n_events']
    ds_mean = data['ds_mean']
    q_sat = data['q_sat']
    abs_gain = data['abs_gain']
    status = data['status']

    valid = np.array([
        (s == 'ok') and np.isfinite(ds) and (tb != 90)
        for s, ds, tb in zip(status, ds_mean, tb_arr)
    ])

    n_v = n_arr[valid]
    rho_v = rho_arr[valid]
    tb_v = tb_arr[valid]
    ds_v = ds_mean[valid]
    qs_v = q_sat[valid]
    g_v = abs_gain[valid]
    ne_v = n_events[valid]

    n_unique = sorted(set(n_v.tolist()))
    tb_unique = sorted(set(tb_v.tolist()))
    rho_unique = sorted(set(rho_v.tolist()))

    # ==========================================================
    # (a) Δs vs |g|
    # ==========================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    for i, n_val in enumerate(n_unique):
        mask = (n_v == n_val)
        ax.scatter(g_v[mask], ds_v[mask],
                   s=18, alpha=0.75,
                   color=COLORS[i % len(COLORS)],
                   edgecolors='white', linewidths=0.3,
                   label=f'$n={n_val}$')
    ax.set_xlabel(r'$|g|$', fontsize=10)
    ax.set_ylabel(r'$\Delta s$', fontsize=10)
    setup_log_axis(ax)
    ax.legend(fontsize=6, ncol=2, loc='upper right',
              frameon=True, framealpha=0.9,
              handletextpad=0.2, columnspacing=0.5)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figure5a.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close()

    # ==========================================================
    # (b) q_sat vs |g|
    # ==========================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    for i, n_val in enumerate(n_unique):
        mask = (n_v == n_val)
        ax.scatter(g_v[mask], qs_v[mask],
                   s=18, alpha=0.75,
                   color=COLORS[i % len(COLORS)],
                   edgecolors='white', linewidths=0.3,
                   label=f'$n={n_val}$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$|g|$', fontsize=10)
    ax.set_ylabel(r'$q_{\rm sat}$ (rad)', fontsize=10)
    setup_log_axis(ax)
    ax.legend(fontsize=6, ncol=2, loc='upper right',
              frameon=True, framealpha=0.9,
              handletextpad=0.2, columnspacing=0.5)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figure5b.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close()

    # ==========================================================
    # (c) N_events vs n
    # ==========================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    for i, tb_val in enumerate(tb_unique):
        mask = (tb_v == tb_val)
        ns = n_v[mask]
        ne = ne_v[mask]
        order = np.argsort(ns)
        ax.plot(ns[order], ne[order], 'o-',
                lw=1.2, markersize=4,
                color=COLORS[i % len(COLORS)],
                label=fr'$\bar\theta={tb_val:.1f}°$')
    ax.set_xlabel(r'$n$', fontsize=10)
    ax.set_ylabel(r'$N_{\rm events}$', fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc='upper left',
              frameon=True, framealpha=0.9,
              handletextpad=0.2, columnspacing=0.5)
    ax.grid(alpha=0.3, lw=0.3)
    plt.tight_layout()
    plt.savefig('figure5c.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close()

    # ==========================================================
    # (d) Δs vs θ̄
    # ==========================================================
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    for i, rho_val in enumerate(rho_unique):
        mask = (rho_v == rho_val)
        tbs = tb_v[mask]
        dss = ds_v[mask]
        order = np.argsort(tbs)
        ax.plot(tbs[order], dss[order], 'o-',
                lw=1.2, markersize=4,
                color=COLORS[i % len(COLORS)],
                label=fr'$\rho={rho_val:.1f}$')
    ax.set_xlabel(r'$\bar\theta$ (deg)', fontsize=10)
    ax.set_ylabel(r'$\Delta s$', fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc='upper left',
              frameon=True, framealpha=0.9,
              handletextpad=0.2, columnspacing=0.5)
    ax.grid(alpha=0.3, lw=0.3)
    plt.tight_layout()
    plt.savefig('figure5d.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close()

    # ==========================================================
    # 合并图
    # ==========================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    ax = axes[0, 0]
    for i, n_val in enumerate(n_unique):
        mask = (n_v == n_val)
        ax.scatter(g_v[mask], ds_v[mask],
                   s=20, alpha=0.75,
                   color=COLORS[i % len(COLORS)],
                   edgecolors='white', linewidths=0.3,
                   label=f'$n={n_val}$')
    ax.set_xlabel(r'$|g|$', fontsize=10)
    ax.set_ylabel(r'$\Delta s$', fontsize=10)
    setup_log_axis(ax)
    ax.legend(fontsize=7, ncol=2, loc='upper right',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    ax.text(0.02, 0.96, '(a)', transform=ax.transAxes,
            fontsize=12, va='top', fontweight='bold')

    ax = axes[0, 1]
    for i, n_val in enumerate(n_unique):
        mask = (n_v == n_val)
        ax.scatter(g_v[mask], qs_v[mask],
                   s=20, alpha=0.75,
                   color=COLORS[i % len(COLORS)],
                   edgecolors='white', linewidths=0.3,
                   label=f'$n={n_val}$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$|g|$', fontsize=10)
    ax.set_ylabel(r'$q_{\rm sat}$ (rad)', fontsize=10)
    setup_log_axis(ax)
    ax.legend(fontsize=7, ncol=2, loc='upper right',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3, which='both')
    ax.text(0.02, 0.96, '(b)', transform=ax.transAxes,
            fontsize=12, va='top', fontweight='bold')

    ax = axes[1, 0]
    for i, tb_val in enumerate(tb_unique):
        mask = (tb_v == tb_val)
        ns = n_v[mask]
        ne = ne_v[mask]
        order = np.argsort(ns)
        ax.plot(ns[order], ne[order], 'o-',
                lw=1.2, markersize=5,
                color=COLORS[i % len(COLORS)],
                label=fr'$\bar\theta={tb_val:.1f}°$')
    ax.set_xlabel(r'$n$', fontsize=10)
    ax.set_ylabel(r'$N_{\rm events}$', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='upper left',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3)
    ax.text(0.02, 0.96, '(c)', transform=ax.transAxes,
            fontsize=12, va='top', fontweight='bold')

    ax = axes[1, 1]
    for i, rho_val in enumerate(rho_unique):
        mask = (rho_v == rho_val)
        tbs = tb_v[mask]
        dss = ds_v[mask]
        order = np.argsort(tbs)
        ax.plot(tbs[order], dss[order], 'o-',
                lw=1.2, markersize=5,
                color=COLORS[i % len(COLORS)],
                label=fr'$\rho={rho_val:.1f}$')
    ax.set_xlabel(r'$\bar\theta$ (deg)', fontsize=10)
    ax.set_ylabel(r'$\Delta s$', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='upper left',
              frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.3)
    ax.text(0.02, 0.96, '(d)', transform=ax.transAxes,
            fontsize=12, va='top', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figure5_combined.png', dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    plt.close()

    print("Saved: figure5a.png through figure5d.png")


if __name__ == '__main__':
    main()
