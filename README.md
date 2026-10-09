# Code and Data for

# "Singular Events in Isostatic Rotor Chains: Transfer Events, Driver Folds, and the Non-Termination of Motion"

Minglin Li (corresponding author) and Shaoyan Lin
School of Mechanical Engineering and Automation
Institute of Embodied Intelligent Robotics
Fuzhou University, Fuzhou 350108, China
Contact: liminglin@fzu.edu.cn

Manuscript under review at *Nonlinear Dynamics* (Springer Nature).

---

## Overview

This repository contains the Python scripts, figures, and numerical
data that support the findings of our manuscript on the geometric
structure of finite-motion paths in topological (Kane--Lubensky)
rotor chains. The code generates all seven main-text figures (19
panels) plus the graphical abstract, and reproduces the parameter
scan and the pseudo-arclength continuation data reported in the
paper.

The central result is the distinction between two classes of
geometric events along finite-motion paths:

1. **Transfer events** -- the vanishing of a single transfer
   denominator $D_j = 0$. Internal force amplification diverges
   while the path continues smoothly through the configuration.

2. **Driver folds** -- the vanishing of a single fold denominator
   $F_j = 0$, equivalently the singularity of the right-end
   controlled Jacobian $C^R$. The end actuator loses its velocity
   authority, and $\theta_n$ has a quadratic extremum. These are
   projection singularities of the $\theta_n$ coordinate, not
   manifold singularities.

Because the constraint Jacobian is bidiagonal, every complementary
minor factorizes into products of the two denominator families, so
the rank criterion is explicit: **rank drops only when a transfer
denominator and a fold denominator vanish simultaneously.** A single
zero denominator never destroys the rank. Over 150 non-degenerate
parameter combinations the smallest singular value of $C$ stays
above $0.2636$, so the constraint manifold remains smooth and
motion termination is excluded by a scalar coincidence condition
rather than by any single singularity.

---

## Repository Structure

The repository is **flat**: all files sit at the top level, so every
script and command below is run from the repository root.

```
topological-rotor-chain/
├── README.md                    (this file)
├── LICENSE                      (MIT License)
├── requirements.txt             (Python dependencies)
├── CITATION.cff                 (machine-readable citation)
│
├── figure1a.png  ┐
├── figure1b.png  ┤
├── figure2a.png  │
├── figure2b.png  │
├── figure3a.png  │  16 figure files
├── figure3b.png  │  (7 main-text figures, 19 panels)
├── figure4_events.png  ┤
├── figure5a.png  │
├── figure5b.png  │
├── figure6a.png  │
├── figure6b.png  │
├── figure6c.png  │
├── figure7a.png  │
├── figure7b.png  │
├── figure7c.png  │
├── figure7d.png  ┘
│
├── figure1.py                   Geometry and zero mode      -> Fig. 1
├── figure2_3_5long_trace.py     Arclength continuation      -> Fig. 2, 3, 5
├── figure4_events.py            Event structure/transmission-> Fig. 4
├── figure6_configurations.py    Configuration snapshots     -> Fig. 6
├── figure7_param_scan.py        Parameter scan              -> Fig. 7
├── graphical_abstract.py        Graphical abstract          -> GA image
│
├── cascade_data_full.py         Parameter scan driver       -> full_scan.csv
├── event_analysis.py            Event position / normal-form analysis
├── diagnose_folds.py            Fold diagnosis helper
├── theory_verification.py       Shared RotorChain model + continuation
├── theory_analysis.py           Analytical cross-checks
├── theory_analysis_v2.py        Analytical cross-checks (v2)
│
├── full_scan.csv                Parameter scan, 180 configurations
└── long_trace.csv               Arclength continuation, reference case
```

**Naming caveat.** `figure2_3_5long_trace.py` is named after the
figures it produces (Fig. 2, 3, and 5). Its internal docstring still
refers to the earlier name `figure2_4_long_trace.py`; the file on
disk is `figure2_3_5long_trace.py`, which is the name to use.

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Regenerate the figures

```bash
python figure1.py                    # Fig. 1  -> figure1a.png, figure1b.png
python figure2_3_5long_trace.py      # Fig. 2,3,5-> 6 panels
python figure4_events.py             # Fig. 4  -> figure4_events.png
python figure6_configurations.py      # Fig. 6  -> figure6a/b/c.png
python figure7_param_scan.py         # Fig. 7  -> figure7a-d.png
python graphical_abstract.py         #         -> graphical_abstract.png/.tif
```

All scripts are self-contained apart from
`figure2_3_5long_trace.py`, `figure4_events.py`, and
`graphical_abstract.py`, which import the shared model from
`theory_verification.py` and run their own continuation.

### Input data expected by the scripts

| Script | Reads | Writes |
|---|---|---|
| `figure2_3_5long_trace.py` | `results_s20/forward/long_trace.csv` (regenerated if absent) | 6 PNG panels |
| `figure7_param_scan.py` | `cascade_data/full_scan.csv` | 4 PNG panels + `figure7_combined.png` |
| `graphical_abstract.py` | `ga_trace_cache.csv` (computed on first run) | PNG + TIFF |
| `cascade_data_full.py` | -- | `cascade_data/full_scan.csv`, `cascade_data/full_scan.png` |

The two CSVs are committed at the repository root as `full_scan.csv`
and `long_trace.csv`, whereas the scripts read them from
`cascade_data/` and `results_s20/forward/`. Either copy the files
into those paths or adjust the path constants at the top of the
respective script.

### Reproduce the parameter scan

```bash
python cascade_data_full.py --full
```

This runs all 180 combinations of chain length $n$, geometric
parameter $\rho$, and reference angle $\bar\theta$, saving
`cascade_data/full_scan.csv` and `cascade_data/full_scan.png`.
**The full scan takes roughly 2--4 hours**, since each configuration
is traced to $s = 40$ with a step of $\Delta s = 0.005$.

For a quick single-configuration test (1 configuration, $s = 20$,
about 5 minutes):

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

### `full_scan.csv`

Parameter scan over 180 configurations (5 chain lengths x 6 values of
$\rho$ x 6 reference angles):

| Column | Description |
|---|---|
| `n` | Chain length, in {8, 12, 16, 24, 32} |
| `rho` | Geometric parameter $a/r$, in {0.8, 1.0, 1.2, 1.4, 1.6, 1.8} |
| `theta_bar_deg` | Reference angle in degrees, in {40, 60, 72.8, 90, 110, 130} |
| `n_events` | Number of transfer events ($D_j = 0$) along the path |
| `ds_mean`, `ds_std` | Mean and standard deviation of the event spacing |
| `q_sat`, `q_std` | Saturation value of $q = \theta_n - \bar\theta$ and its std, averaged over the last 20% of the path |
| `smin_C_min` | Minimum of $\sigma_{\min}(C)$ along the traced path |
| `abs_gain` | Absolute reference-state gain $\lvert g\rvert$ |
| `status` | `ok` if the continuation ran to $s = 40$ without failure |

Two caveats when reading this file:

- **Angle convention.** The reference case analysed in the paper
  uses $\bar\theta = \pi/2 - 0.3$ rad $= 72.812$ deg; the grid point
  `72.8` is its rounded nearest neighbour. Reference-case quantities
  quoted in the text are therefore computed at 72.812 deg and differ
  from the corresponding CSV row in the fourth decimal place
  (e.g. $\lvert g\rvert = 3.196178$ in closed form versus
  `abs_gain = 3.196458` in the CSV row for $n=16$, $\rho=1.0$).
- **Event spacing.** `ds_mean` averages over the whole path to
  $s = 40$, transient included. For the reference row it is 2.466,
  whereas the steady staircase spacing after the transient is 2.271,
  the value quoted in the paper. Both describe the same path.

### `long_trace.csv`

Pseudo-arclength continuation for the reference case
($n = 16$, $\rho = 1.0$, $\bar\theta = \pi/2 - 0.3$):

| Column | Description |
|---|---|
| `s` | Arclength along the constraint manifold |
| `q` | Right-end displacement, $\theta_n - \bar\theta$ |
| `sigma_min_C` | Smallest singular value of the full Jacobian $C$ |
| `sigma_min_CR` | Smallest singular value of $C^R$ |
| `D_0` ... `D_14` | Transfer denominators along the path (15 columns) |

The file uses UTF-8 with BOM encoding, so it opens correctly in
Excel and LibreOffice Calc.

---

## Figure Map

| Figure | Files | Content | Script |
|---|---|---|---|
| **Fig. 1** | `figure1a.png`, `figure1b.png` | Rotor chain geometry; zero-mode amplitude | `figure1.py` |
| **Fig. 2** | `figure2a.png`, `figure2b.png` | $D_j$ along the path; $\sigma_{\min}(C)$ and $\sigma_{\min}(C^R)$ | `figure2_3_5long_trace.py` |
| **Fig. 3** | `figure3a.png`, `figure3b.png` | Behaviour near the first transfer event ($D_{13} = 0$) | `figure2_3_5long_trace.py` |
| **Fig. 4** | `figure4_events.png` | Four panels: event staircase; fold normal form; quadratic extremum of $\theta_n$; motion authority vs force amplification | `figure4_events.py` |
| **Fig. 5** | `figure5a.png`, `figure5b.png` | Behaviour near the first driver fold ($F_{13} = 0$) | `figure2_3_5long_trace.py` |
| **Fig. 6** | `figure6a.png`, `figure6b.png`, `figure6c.png` | Configuration snapshots: reference state, transfer event, fold event | `figure6_configurations.py` |
| **Fig. 7** | `figure7a.png`--`figure7d.png` | Parameter scan over $(n, \rho, \bar\theta)$ | `figure7_param_scan.py` |
| **Graphical abstract** | `graphical_abstract.png`, `graphical_abstract.tif` | Four-panel summary for the journal | `graphical_abstract.py` |

All main-text panels are 300 dpi PNG, single-column width
(3.4 x 2.8 in), set in Times New Roman with the colourblind-friendly
Okabe--Ito palette. Fig. 4 and the graphical abstract are
double-column figures. `figure6_configurations.py` also writes an
auxiliary `figure_config.png`, and `figure7_param_scan.py` writes an
auxiliary `figure7_combined.png`; neither is used in the manuscript.

---

## Key Numerical Results

Reference case ($n = 16$, $\rho = 1.0$, $\bar\theta = \pi/2 - 0.3$ rad):

- First **transfer event**: $D_{13} = 0$ at $s = 2.04017$,
  with $\sigma_{\min}(C) = 0.5163$ and
  $\sigma_{\min}(C^R) = 0.5163$ -- the path passes through smoothly.
- First **driver fold**: $F_{13} = 0$ at $s = 5.07431$, with
  $\sigma_{\min}(C) = 0.5174$ bounded but
  $\sigma_{\min}(C^R) = 0$ to machine precision.
- **Full Jacobian along the path**: $\sigma_{\min}(C) \geq 0.4205$
  over all 401 sampled steps, and the numerical rank agrees with the
  rank criterion at every step (0 mismatches).
- **Zero-mode decay rate**: $\lvert g\rvert = 3.196178$.
- **Event staircase**: transfer events are separated by 2.271 in
  arclength (the first interval is 2.872), and each transfer event is
  followed by the fold event of the adjacent constraint after 0.162.
- **Fold normal form**: $\sigma_{\min}(C^R) \sim \lvert s - s^\ast\rvert^{1.0006}$
  ($R^2 = 0.999987$) and
  $\theta_n - \theta_n^\ast = b(s-s^\ast)^2$ with
  $b = -0.0423$ rad ($R^2 = 0.99997$).

Across the 150 **non-degenerate** configurations
($\bar\theta \neq 90$ deg):

- $\sigma_{\min}(C) > 0.2636$ everywhere (minimum at $n = 8$,
  $\rho = 0.8$, $\bar\theta = 60$ deg), so no manifold singularity is
  encountered.
- No universal scaling of event spacing or saturation position with
  $\lvert g\rvert$.

The remaining 30 configurations ($\bar\theta = 90$ deg) form a
**degenerate family** and are excluded from the statement above. At
that angle both denominator families vanish simultaneously at the
reference state ($F_j = D_j = 0$ for all $j$), every entry of $C$ is
zero, the recorded `smin_C_min` values lie between $1.2\times10^{-16}$
and $1.6\times10^{-9}$ (numerical zero), and no events are found
along the path. This is a property of the chosen reference state, not
a singularity encountered during finite motion.

---

## Methods Summary

The numerical method is **pseudo-arclength continuation** on the
constraint manifold $\mathcal{M} = \{\theta : l_i(\theta) = \bar{l}$
for all $i\}$. At each step the augmented system

    F(theta) = 0
    t_prev . (theta - theta_pred) = 0

is solved by Newton correction, where the tangent $t$ is the unit
null vector of the constraint Jacobian $C$. The arclength step is
$\Delta s = 0.005$, the Newton tolerance is $10^{-11}$, and the
constraint residual is kept below $10^{-12}$. The scan follows each
path to $s = 40$.

Both event families are scalar zeros along the path, so events are
localized by **Brent's method on the cubic-spline interpolant** of the
relevant denominator, bracketed by a sign change between adjacent
steps, with a tolerance of $10^{-13}$ in $s$. No threshold on
$\sigma_{\min}(C^R)$ is involved: at a refined fold position
$\det C^R = 0$ holds to machine precision by Corollary 3 of the paper.
Manifold singularities are monitored directly through
$\sigma_{\min}(C)$.

---

## AI Use Disclosure

DeepSeek (V3, 2025) and ChatGPT (GPT-5, 2025) were used to assist
with deriving equations, generating Python code, analysing numerical
output, and drafting the manuscript. All AI-generated content was
independently verified by the authors through analytical cross-checks,
step-halving convergence tests, and comparison with the existing
literature.

The AI tools had no role in the conceptual design of the study, the
interpretation of physical mechanisms, or the final scientific
conclusions.

---

## Citation

If you use this code or data, please cite the associated manuscript.
`CITATION.cff` contains a machine-readable record.

---

## License

This repository is released under the MIT License. See `LICENSE` for
details.

---

## Contact

For questions about the code or data, please contact:

**Minglin Li**
Email: liminglin@fzu.edu.cn
School of Mechanical Engineering and Automation
Fuzhou University, Fuzhou 350108, China
