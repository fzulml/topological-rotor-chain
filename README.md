# Code and Data for
# "Singular Events in Isostatic Rotor Chains: Transfer Events, Driver Folds, and the Non-Termination of Motion"

Minglin Li (corresponding author) and Shaoyan Lin  
School of Mechanical Engineering and Automation  
Institute of Embodied Intelligent Robotics  
Fuzhou University, Fuzhou 350108, China  
Contact: liminglin@fzu.edu.cn

---

## Overview

This repository contains the Python scripts and numerical data
that support the findings in our manuscript on the geometric
structure of finite-motion paths in topological rotor chains. The
code generates all nineteen main-text figures and reproduces the
parameter scan and arclength continuation data reported in the
paper.

The central result is the distinction between two classes of
geometric events along finite-motion paths:

1. **Local transfer singularities** — the vanishing of a single
   local denominator $D_j = 0$. These are coordinate folds that
   the path continues through.

2. **Driver folds** — the singularity of the right-end controlled
   Jacobian $C^R$. These are projection singularities of the
   $\theta_n$ coordinate, not manifold singularities.

In all tested configurations, the full constraint Jacobian $C$
remains full rank, indicating that the constraint manifold is
smooth throughout the motion.

---

## Repository Structure

```
topological-rotor-chain/
├── README.md                  (this file)
├── LICENSE                    (MIT License)
├── requirements.txt           (Python dependencies)
├── CITATION.cff               (machine-readable citation)
│
├── figures/                   12 main-text figures (300 dpi PNG)
│   ├── figure1a.png
│   ├── figure1b.png
│   ├── figure2a.png
│   ├── figure2b.png
│   ├── figure3a.png
│   ├── figure3b.png
│   ├── figure4a.png
│   ├── figure4b.png
│   ├── figure4c.png
│   ├── figure4d.png
│   ├── figure5a.png
│   ├── figure5b.png
│   ├── figure6a.png
│   ├── figure6b.png
│   ├── figure6c.png
│   ├── figure7a.png
│   ├── figure7b.png
│   ├── figure7c.png
│   └── figure7d.png
│
├── scripts/                   4 Python scripts
│   ├── figure1.py             Generates Fig. 1(a)(b)
│   ├── figure2_4_long_trace.py Generates Fig. 2-4 (8 panels)
│   ├── figure5_param_scan.py  Generates Fig. 5 (2 panels)
│   ├── figure5_param_scan.py  Generates Fig. 5 (2 panels)
│   └── cascade_data_full.py   Produces data/full_scan.csv
│
└── data/                      2 numerical data files
    ├── full_scan.csv          Parameter scan over (n, rho, theta_bar)
    └── long_trace.csv         Arclength continuation for the reference case
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Regenerate all figures

```bash
cd scripts
python figure1.py
python figure2_4_long_trace.py
python figure5_param_scan.py
```

The scripts will produce PNG files in the current directory. Note
that `figure2_4_long_trace.py` generates `data/long_trace.csv`
automatically if the file does not already exist.

### Reproduce the parameter scan data

```bash
cd scripts
python cascade_data_full.py --full
```

This runs 180 combinations of chain length, geometric parameter,
and reference angle, and saves the results to
`data/full_scan.csv`. The full scan takes approximately 5-10
minutes on a standard laptop.

For a quick test (1 configuration, ~2 seconds):

```bash
python cascade_data_full.py --test
```
---
## Requirements

- Python >= 3.9
- NumPy >= 1.22
- SciPy >= 1.9
- Matplotlib >= 3.6

All dependencies are listed in `requirements.txt`.

---

## Data Description

### `data/full_scan.csv`

Parameter scan over 180 configurations with columns:

| Column | Description |
|---|---|
| `n` | Chain length |
| `rho` | Geometric parameter a/r |
| `theta_bar_deg` | Reference angle (degrees) |
| `n_events` | Number of local transfer singularities |
| `ds_mean`, `ds_std` | Mean and standard deviation of event spacing |
| `q_sat`, `q_std` | Saturation position q_sat and its std |
| `smin_C_min` | Minimum sigma_min(C) along the path |
| `abs_gain` | Reference-state gain |g| |
| `status` | "ok" if the run succeeded |

### `data/long_trace.csv`

Arclength continuation data for the reference case
(n = 16, rho = 1.0, theta_bar = pi/2 - 0.3):

| Column | Description |
|---|---|
| `s` | Arclength along the constraint manifold |
| `q` | Right-end displacement theta_n - theta_bar |
| `sigma_min_C` | Smallest singular value of the full Jacobian C |
| `sigma_min_CR` | Smallest singular value of C^R |
| `D_0` ... `D_14` | Local denominators along the path |

The file uses UTF-8 with BOM encoding, so it opens correctly in
Excel and LibreOffice Calc.

---

## Figure Map

| Figure | Files | Content | Script |
|---|---|---|---|
| **Fig. 1** | `figure1a.png`, `figure1b.png` | Rotor chain geometry; zero mode amplitude | `figure1.py` |
| **Fig. 2** | `figure2a.png`, `figure2b.png` | D_j along path; sigma_min(C) and sigma_min(C^R) | `figure2_4_long_trace.py` |
| **Fig. 3** | `figure3a.png`, `figure3b.png` | Behavior near the first local transfer singularity | `figure2_4_long_trace.py` |
| **Fig. 4** | `figure4a.png`, `figure4b.png` | Behavior near the first driver fold | `figure2_4_long_trace.py` |
| **Fig. 5** | `figure5a.png` to `figure5d.png` | Parameter scan over (n, rho, theta_bar) | `figure5_param_scan.py` |

All figures are 300 dpi PNG, single-column width (3.4 x 2.8
inches for most panels), and use Times New Roman font with a
colorblind-friendly palette.

---

## Key Numerical Results

For the reference case (n = 16, rho = 1.0,
theta_bar = pi/2 - 0.3):

- First local transfer singularity: D_13 = 0 at s ≈ 5.05
- First driver fold: sigma_min(C^R) ≈ 3e-5
- Full Jacobian condition: sigma_min(C) > 0.42 throughout
- Zero mode decay rate: |g| = 3.196178

Across all 180 tested configurations:

- sigma_min(C) > 0.26 everywhere
- No manifold singularity observed
- No universal scaling of event spacing or fold position
- Degenerate case: theta_bar = 90 degrees yields no events

---

## Methods Summary

The numerical method is **pseudo-arclength continuation** on the
constraint manifold M = {theta : l_i(theta) = l_bar for all i}.
At each step, the augmented system

    F(theta) = 0
    theta_dot_prev . (theta - theta_pred) = 0

is solved by Newton correction. The tangent is the null space of
the constraint Jacobian. The arclength step is Delta_s = 0.005,
with Newton tolerance 1e-11 and constraint residual kept below
1e-12.

Event positions are refined by linear interpolation between
adjacent steps and verified with step-halving tests (agreement
to 1e-13 rad).

---

## AI Use Disclosure

DeepSeek (V3, 2025) and ChatGPT (GPT-5, 2025) were used to
assist with deriving equations, generating initial Python code,
and drafting the manuscript. All AI-generated content was
independently verified by the authors through analytical
cross-checks, step-halving convergence tests, and comparison
with the existing literature.

The AI tools had no role in the conceptual design of the study,
the interpretation of physical mechanisms, or the final
scientific conclusions.

---

## License

This repository is released under the MIT License. See
`LICENSE` for details.

---

## Contact

For questions about the code or data, please contact:

**Minglin Li**  
Email: liminglin@fzu.edu.cn  
School of Mechanical Engineering and Automation  
Fuzhou University, Fuzhou 350108, China
```
---
