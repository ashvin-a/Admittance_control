"""Stage 3: impedance controller in simulation. Run:  .venv/bin/python sim_stage3.py
Writes figures to figures/ and prints every number used in the report.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import arm
import impedance as imp

np.set_printoptions(precision=4, suppress=True, linewidth=120)
os.makedirs("figures", exist_ok=True)

# ---- plot style: validated categorical slots 1-3, neutral ink, recessive grid ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 150,
})

K0, D0 = 20 * np.eye(2), 5 * np.eye(2)   # the proposal's example gains
T = 15.0


def hdr(s):
    print(f"\n=== {s} ===")


# =====================================================================================
# 1. Eigenvalues verified in simulation (DMD on a small-amplitude nonlinear trajectory)
# =====================================================================================
hdr("1. A_cl eigenvalues: predicted vs identified from the NONLINEAR simulation")
d_small = np.array([0.01, -0.007])
t, X = imp.simulate_nonlinear(np.zeros(4), 3.0, K0, D0, th_d=lambda t: d_small)
Z = X - np.concatenate([d_small, np.zeros(2)])[:, None]          # error state
dt = t[1] - t[0]
Phi = Z[:, 1:] @ np.linalg.pinv(Z[:, :-1])
ev_sim = np.sort_complex(np.log(np.linalg.eigvals(Phi).astype(complex)) / dt)
A_d, _ = imp.linear_error_model(d_small, K0, D0)
ev_th = np.sort_complex(np.linalg.eigvals(A_d))
print("predicted :", ev_th)
print("simulated :", ev_sim)
print(f"max |difference| = {np.abs(ev_th - ev_sim).max():.2e}")

# =====================================================================================
# 2. Step in theta_d: nonlinear vs linear, small and large
# =====================================================================================
hdr("2. theta_d step from rest at theta = 0  (K=20I, D=5I)")
steps = {"small step": np.array([0.1, -0.07]), "large step": np.array([1.0, -0.7])}
fig, axs2 = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axs = axs2[0]
for ax, axd, (name, dth) in zip(axs, axs2[1], steps.items()):
    t, X = imp.simulate_nonlinear(np.zeros(4), T, K0, D0, th_d=lambda t, d=dth: d)
    A, _ = imp.linear_error_model(dth, K0, D0)
    t, Zl = imp.simulate_linear(np.concatenate([-dth, np.zeros(2)]), T, A, np.zeros((4, 2)))
    th_lin = Zl[:2] + dth[:, None]
    err = np.abs(X[:2] - th_lin).max() / np.linalg.norm(dth)
    tol = 0.02 * np.linalg.norm(dth)
    ts_nl = imp.settling_time(t, X[:2] - dth[:, None], tol)
    ts_li = imp.settling_time(t, Zl[:2], tol)
    print(f"{name:10s} |dtheta_d| = {np.linalg.norm(dth):.3f} rad: max|nl-lin|/|dtheta_d| = {err:.3%}, "
          f"2% settling nonlinear {ts_nl:.2f} s, linear {ts_li:.2f} s")
    for j, c in enumerate([BLUE, ORANGE]):
        ax.axhline(dth[j], color=c, lw=0.8, ls=":")
        ax.plot(t, X[j], color=c, lw=2.6, label=rf"$\theta_{j+1}$ nonlinear")
        ax.plot(t, th_lin[j], color=INK, ls="--", lw=1.0, label="linear model" if j == 0 else None)
        axd.plot(t, (X[j] - th_lin[j]) * 1e3, color=c, lw=1.4, label=rf"$\theta_{j+1}$")
    ax.set_title(f"{name}: $|\\Delta\\theta_d|$ = {np.linalg.norm(dth):.2f} rad")
    axd.set_xlabel("time (s)")
    axd.set_xlim(0, T)
    axd.legend(loc="upper right", ncol=2)
axs[0].set_ylabel("joint angle (rad)")
axs2[1][0].set_ylabel("nonlinear $-$ linear (mrad)")
axs[0].legend(loc="center right", ncol=1)
fig.tight_layout()
fig.savefig("figures/stage3_step.png", bbox_inches="tight")
plt.close(fig)

# =====================================================================================
# 3. External torque: pulse response, and static stiffness K e = tau_ext
# =====================================================================================
hdr("3. External torque pulse from rest at theta = theta_d = 0  (K=20I, D=5I)")
pulse = lambda t: np.array([4.0, -2.0]) * (0.5 <= t < 0.8)
th_d = np.zeros(2)
t, X = imp.simulate_nonlinear(np.zeros(4), T, K0, D0, th_d=lambda t: th_d, tau_ext=pulse)
A, Bw = imp.linear_error_model(th_d, K0, D0)
t, Zl = imp.simulate_linear(np.zeros(4), T, A, Bw, tau_ext=pulse)
err = np.abs(X[:2] - Zl[:2]).max()
print(f"pulse [4,-2] N.m for 0.3 s: peak |e| nonlinear = {np.abs(X[:2]).max():.3f} rad, "
      f"max |nl - lin| = {err:.4f} rad ({err / np.abs(X[:2]).max():.1%} of peak)")

fig, axs = plt.subplots(2, 1, figsize=(5.4, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
tt = np.array([pulse(s) for s in t])
for j, c in enumerate([BLUE, ORANGE]):
    axs[0].plot(t, tt[:, j], color=c, label=rf"$\tau_{{ext,{j+1}}}$")
    axs[1].plot(t, X[j], color=c, lw=2.6, label=rf"$e_{j+1}$ nonlinear")
    axs[1].plot(t, Zl[j], color=INK, ls="--", lw=1.0, label="linear model" if j == 0 else None)
axs[0].set_ylabel(r"$\tau_{ext}$ (N$\cdot$m)")
axs[1].set_ylabel(r"error $\theta-\theta_d$ (rad)")
axs[1].set_xlabel("time (s)")
axs[0].legend(loc="upper right", ncol=2)
axs[1].legend(loc="lower right", ncol=2)
axs[1].set_xlim(0, T)
fig.tight_layout()
fig.savefig("figures/stage3_pulse.png", bbox_inches="tight")
plt.close(fig)

hdr("3b. Static stiffness: constant tau_ext, steady-state error vs K^-1 tau_ext (nonlinear sim)")
for th_d, tau in [(np.zeros(2), np.array([4.0, -2.0])), (np.array([0.5, 0.5]), np.array([-3.0, 1.0]))]:
    Kmat = np.array([[20.0, 0.0], [0.0, 40.0]])      # non-identical gains to make the check non-trivial
    Dmat = np.array([[8.0, 0.0], [0.0, 8.0]])
    x0 = np.concatenate([th_d, np.zeros(2)])
    t, X = imp.simulate_nonlinear(x0, 60.0, Kmat, Dmat, th_d=lambda t, d=th_d: d, tau_ext=lambda t, f=tau: f)
    e_ss = X[:2, -1] - th_d
    pred = np.linalg.solve(Kmat, tau)
    print(f"theta_d={th_d}, tau_ext={tau}: e_ss = {e_ss}, K^-1 tau_ext = {pred}, "
          f"|diff| = {np.abs(e_ss - pred).max():.1e}, residual speed {np.abs(X[2:, -1]).max():.1e}")

# =====================================================================================
# 4. Stiffness vs damping
# =====================================================================================
hdr("4. Stiffness vs damping: step dtheta_d = [0.3, -0.2] (nonlinear sim)")
dth = np.array([0.3, -0.2])
e0n = np.linalg.norm(dth)
T4 = 20.0


def run(k, d):
    Kc, Dc = k * np.eye(2), d * np.eye(2)
    t, X = imp.simulate_nonlinear(np.zeros(4), T4, Kc, Dc, th_d=lambda t: dth)
    E = X[:2] - dth[:, None]
    return t, E


def metrics(E, t):
    ts = imp.settling_time(t, E, 0.02 * e0n)
    prog = 1 + (E * dth[:, None]).sum(0) / e0n**2          # 0 at start, 1 at target, >1 = overshoot
    return ts, max(0.0, (prog.max() - 1) * 100)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
panels = [
    ("Stiffness K (D = 10 fixed)", [(5, 10), (20, 10), (80, 10)], lambda k, d: f"K = {k}"),
    ("Damping D (K = 20 fixed)", [(20, 5), (20, 10), (20, 22)], lambda k, d: f"D = {d}"),
]
for ax, (title, cfgs, lab) in zip(axs, panels):
    for c, (k, d) in zip([BLUE, ORANGE, AQUA], cfgs):
        t, E = run(k, d)
        ax.plot(t, np.linalg.norm(E, axis=0) / e0n, color=c, label=lab(k, d))
        ts, os_ = metrics(E, t)
        modes = imp.modal_damping(dth, k, d)
        zs = ", ".join(f"{z:.2f}" for _, _, z in modes)
        print(f"K={k:3d} D={d:3d}: 2% settling {ts:5.2f} s, overshoot {os_:5.1f}%, modal zeta (light, heavy) = {zs}")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, 12)
    ax.legend(loc="upper right")
axs[0].set_ylabel(r"$\|e\| / \|e(0)\|$")
fig.tight_layout()
fig.savefig("figures/stage3_sweep.png", bbox_inches="tight")
plt.close(fig)

hdr("4b. Grid of 2% settling time (s) / overshoot (%) / worst modal zeta (nonlinear, same step)")
ks, ds = [5, 20, 80], [2.5, 5, 10, 22, 40]
print("        " + "".join(f"D={d:<18}" for d in ds))
for k in ks:
    row = f"K={k:<4d} "
    for d in ds:
        t, E = run(k, d)
        ts, os_ = metrics(E, t)
        zmin = min(z for _, _, z in imp.modal_damping(dth, k, d))
        row += f"{ts:5.1f}/{os_:4.0f}/{zmin:4.2f}     "
    print(row)

hdr("4c. Why scalar gains cannot damp both modes: eigenvalues of M(theta_d) and rate ratio")
m = np.linalg.eigvalsh(arm.M(dth))
print(f"M(theta_d) eigenvalues = {m}, ratio {m[1] / m[0]:.1f}")
print("done; figures in figures/")
