# Impedance Control of a 2-DOF Robot Arm

ECE 717 course project. We model a two-link arm moving in a vertical plane,
analyze its linearization (stability, controllability, observability,
stabilizability, detectability), design a joint-space impedance controller,
prove it is globally stable, and test every prediction against simulations of
the full nonlinear model.

## The system

| | |
|---|---|
| Plant | two links, point masses `m1 = m2 = 1 kg` at the tips, `l1 = l2 = 1 m`, `g = 9.81 m/s^2`; angles are zero when the arm hangs straight down |
| Input | motor torques `u = [tau1, tau2]` (one motor per joint) |
| State | `x = [theta1, theta2, theta1_dot, theta2_dot]` |
| Output | joint angles `y = [theta1, theta2]` |
| Disturbance | external torque `tau_ext` from contact |
| Controller | `u = -tau_g(theta) + K (theta_d - theta) - D theta_dot` (gravity compensation plus a virtual spring and damper) |

## Main results

- The hanging arm is only marginally stable (eigenvalues `±j2.40`, `±j5.79` rad/s): without friction it swings forever.
- The linearization is controllable and observable at all four equilibria, and stays so with a single motor or a single sensor as long as gravity couples the joints. With `g = 0` it does not.
- Under the impedance law, a steady push deflects the arm by exactly `K^-1 tau_ext`. An energy function proves global asymptotic stability and passivity.
- The arm's inertia differs by up to 34x between directions, so a scalar damping gain cannot damp both modes. The damping matrix `D = 2 zeta sqrt(k) M(theta_d)^(1/2)` gives every mode damping ratio `zeta`. It settles a 1.22 rad step in 2.6 s (1.3% overshoot) instead of 7.0 s (13%), but deflects further under a sharp push.

## Files

| File | What it does |
|---|---|
| `arm.py` | The model: `M`, `C`, `tau_g`, `f(x, u)`, plus an independent energy-based derivation used for checking |
| `impedance.py` | Impedance control law, linear error model `A_cl`, simulation helpers |
| `stage1_verify.py` | Checks the model and the linearization, and every number quoted in the model section of the paper |
| `stage2_structural.py` | Controllability, observability, stabilizability and detectability at all equilibria, with actuators or sensors removed, and without gravity |
| `sim_stage3.py` | Step, push, static-stiffness and gain-sweep simulations with scalar gains |
| `sim_damping.py` | Uniform damping design, global-stability check from random starts, gravity-model error |
| `results/*.txt` | Saved output of the scripts above; the paper's numbers come from these |
| `figures/*.png` | Figures written by `sim_stage3.py` and `sim_damping.py` |
| `main.tex` | The report (IEEE conference format); `mybibfile.bib`, `ieeeconf.cls` and `IEEEtran.bst` are needed to build it |
| `ECE_717_Impedance_control.pdf` | Compiled report |

## Running

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python stage1_verify.py        # prints PASS/FAIL for each check
.venv/bin/python stage2_structural.py    # writes results/stage2_structural.txt
.venv/bin/python sim_stage3.py > results/stage3_results.txt   # also writes figures/stage3_*.png
.venv/bin/python sim_damping.py          # writes figures/damping_compare.png and results/damping_results.txt (several minutes)
```

## Building the report

```sh
latexmk -pdf main.tex
```

This needs a LaTeX installation with BibTeX. `ECE_717_Impedance_control.pdf` is a copy of the resulting `main.pdf`.
