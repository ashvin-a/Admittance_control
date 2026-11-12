"""Stage 4: damping design, global stability check, gravity-model error.

Run:  .venv/bin/python sim_damping.py
Writes figures/damping_compare.png and results/damping_results.txt; every number
quoted in Secs. V-VII of main.tex comes from this script.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp

import arm
import impedance as imp
from arm import M, tau_g, P

np.set_printoptions(precision=4, suppress=True, linewidth=120)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
    "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 150,
})

out = []
def say(s=""):
    print(s); out.append(s)


def sqrtm2(A):
    """Principal square root of a 2x2 SPD matrix (closed form, fast enough for the ODE loop)."""
    s = np.sqrt(np.linalg.det(A))
    return (A + s * np.eye(2)) / np.sqrt(np.trace(A) + 2 * s)


def uniform_damping(Md, K, zeta):
    """D = 2 zeta M^1/2 (M^-1/2 K M^-1/2)^1/2 M^1/2.

    With e = M^-1/2 q the linearized closed loop becomes q'' + 2 zeta Om q' + Om^2 q = 0,
    Om = (M^-1/2 K M^-1/2)^1/2 SPD, so every mode has damping ratio zeta.
    For K = k I this reduces to D = 2 zeta sqrt(k) M^1/2.
    """
    S = sqrtm2(Md)
    Si = np.linalg.inv(S)
    Om = sqrtm2(Si @ K @ Si)
    D = 2 * zeta * S @ Om @ S
    return 0.5 * (D + D.T)


def modal_zetas(A):
    lam = np.linalg.eigvals(A)
    return np.sort(-lam.real / np.abs(lam))


def simulate(x0, t_end, K, Dfun, th_d, tau_ext=None, gravity_model=None):
    """Nonlinear closed loop; Dfun(theta) gives the damping matrix (constant or online).
    gravity_model(theta) is the controller's gravity estimate (default: exact)."""
    tau_ext = tau_ext or (lambda t: np.zeros(2))
    gm = gravity_model or tau_g

    def u_of(x):
        return -gm(x[:2]) + K @ (th_d - x[:2]) - Dfun(x[:2]) @ x[2:]

    sol = solve_ivp(lambda t, x: arm.f(x, u_of(x), tau_ext(t)), [0, t_end], x0, method="DOP853",
                    rtol=1e-9, atol=1e-11, max_step=0.005, dense_output=True)
    t = np.arange(0, t_end, 0.005)
    X = sol.sol(t)
    U = np.array([u_of(x) for x in X.T]).T
    return t, X, U


def step_metrics(t, X, th_d, th0):
    dth = th_d - th0
    E = X[:2] - th_d[:, None]
    ts = imp.settling_time(t, E, 0.02 * np.linalg.norm(dth))
    prog = 1 + (E * dth[:, None]).sum(0) / (dth @ dth)
    return ts, 100 * max(0.0, prog.max() - 1)


# ---------------------------------------------------------------------------------
say("=== 1. Uniform damping design: modal zeta of A_cl (target 0.7) ===")
for K in [20 * np.eye(2), np.diag([20.0, 40.0]), np.array([[30.0, 5.0], [5.0, 10.0]])]:
    for th_d in [np.zeros(2), np.array([1.0, -0.7]), np.array([0.0, np.pi / 2])]:
        Md = M(th_d)
        D = uniform_damping(Md, K, 0.7)
        A, _ = imp.linear_error_model(th_d, K, D)
        say(f"K={np.array2string(K, separator=',').replace(chr(10), '')} theta_d={th_d}: zeta = {modal_zetas(A)}")
K20 = 20 * np.eye(2)
A, _ = imp.linear_error_model(np.zeros(2), K20, uniform_damping(M(np.zeros(2)), K20, 0.7))
say(f"K=20I, zeta=0.7, theta_d=0: eig(A_cl) = {np.sort_complex(np.linalg.eigvals(A))}")
say(f"   D = 2*0.7*sqrt(20)*M0^1/2 = {uniform_damping(M(np.zeros(2)), K20, 0.7).round(3).tolist()}")

# ---------------------------------------------------------------------------------
say()
say("=== 2. Scalar vs mass-aware damping (K = 20 I), nonlinear simulation ===")
th_big = np.array([1.0, -0.7])
th_small = np.array([0.3, -0.2])
cases = {
    "D = 5I (proposal)": lambda thd: (lambda th: 5 * np.eye(2)),
    "D = 10I (best scalar d for k=20)": lambda thd: (lambda th: 10 * np.eye(2)),
    "mass-aware, D fixed at theta_d": lambda thd: (lambda th, Dc=uniform_damping(M(thd), K20, 0.7): Dc),
    "mass-aware, D(theta) online": lambda thd: (lambda th: uniform_damping(M(th), K20, 0.7)),
}
say(f"{'case':<34} {'large step 1.22 rad':>24} {'small step 0.36 rad':>22} {'pulse (4,-2) N.m, 0.3 s':>30}  {'peak |u| step':>13}")
say(f"{'':<34} {'settle(s) / overshoot':>24} {'settle(s) / overshoot':>22} {'peak |e|(rad) / settle(s)':>30}  {'(N.m)':>13}")
traj = {}
pulse = lambda t: np.array([4.0, -2.0]) if 0.5 <= t < 0.8 else np.zeros(2)
for name, mk in cases.items():
    t, X, U = simulate(np.zeros(4), 15, K20, mk(th_big), th_big)
    ts_b, os_b = step_metrics(t, X, th_big, np.zeros(2))
    t2, X2, _ = simulate(np.zeros(4), 15, K20, mk(th_small), th_small)
    ts_s, os_s = step_metrics(t2, X2, th_small, np.zeros(2))
    tp, Xp, _ = simulate(np.zeros(4), 15, K20, mk(np.zeros(2)), np.zeros(2), tau_ext=pulse)
    pk = np.abs(Xp[:2]).max()
    inf = np.abs(Xp[:2]).max(0)
    ts_p = tp[np.where(inf > 0.02 * pk)[0][-1]]
    traj[name] = (t, X, tp, Xp)
    say(f"{name:<34} {ts_b:>12.2f} / {os_b:5.1f}%   {ts_s:>10.2f} / {os_s:5.1f}%   {pk:>18.3f} / {ts_p:5.2f}      {np.abs(U).max():>8.1f}")

# ---------------------------------------------------------------------------------
say()
say("=== 3. Global stability check (Prop. 1): V = 1/2 thd' M thd + 1/2 e' K e along random large motions ===")
rng = np.random.default_rng(3)
th_d = np.array([0.5, 0.5])
for name, Dfun in [("D = 5I", lambda th: 5 * np.eye(2)),
                   ("mass-aware online", lambda th: uniform_damping(M(th), K20, 0.7))]:
    worst_rise, worst_final = 0.0, 0.0
    for _ in range(20):
        x0 = np.concatenate([rng.uniform(-np.pi, np.pi, 2), rng.uniform(-3, 3, 2)])
        t, X, _ = simulate(x0, 40, K20, Dfun, th_d)
        e = X[:2] - th_d[:, None]
        V = np.array([0.5 * X[2:, i] @ M(X[:2, i]) @ X[2:, i] + 0.5 * e[:, i] @ K20 @ e[:, i] for i in range(len(t))])
        worst_rise = max(worst_rise, (np.diff(V).max()) / V[0])
        worst_final = max(worst_final, np.linalg.norm(X[:, -1] - np.r_[th_d, 0, 0]))
    say(f"{name:<18}: 20 random starts (|theta|<=pi, |thd|<=3): max relative rise of V between samples "
        f"{worst_rise:.1e}, worst |x(40 s) - x*| = {worst_final:.1e}")

# ---------------------------------------------------------------------------------
say()
say("=== 4. Gravity-model error: controller believes m2 = 0.9 kg (true 1.0) ===")
p_hat = arm.Params(m2=0.9)
gm = lambda th: tau_g(th, p_hat)
for k in [20.0, 80.0]:
    K = k * np.eye(2)
    t, X, _ = simulate(np.zeros(4), 40, K, lambda th: 10 * np.eye(2) * np.sqrt(k / 20), th_big, gravity_model=gm)
    e_ss = X[:2, -1] - th_big
    pred = np.linalg.solve(K, tau_g(th_big) - gm(th_big))
    say(f"k={k:4.0f}: steady error = {e_ss.round(4)} rad, first-order prediction K^-1 (tau_g - tau_g_hat) = {pred.round(4)}")

# ---------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
styles = {
    "D = 5I (proposal)": dict(color=BLUE),
    "D = 10I (best scalar d for k=20)": dict(color=ORANGE),
    "mass-aware, D fixed at theta_d": dict(color=AQUA),
    "mass-aware, D(theta) online": dict(color=INK, ls="--", lw=1.2),
}
labels = {
    "D = 5I (proposal)": r"$\mathbf{D}=5\mathbf{I}$",
    "D = 10I (best scalar d for k=20)": r"$\mathbf{D}=10\mathbf{I}$",
    "mass-aware, D fixed at theta_d": r"uniform, $\zeta=0.7$",
    "mass-aware, D(theta) online": r"uniform, online $\mathbf{D}(\boldsymbol{\theta})$",
}
for name, (t, X, tp, Xp) in traj.items():
    e = np.linalg.norm(X[:2] - th_big[:, None], axis=0)
    axs[0].plot(t, e / e[0], label=labels[name], **styles[name])
    axs[1].plot(tp, np.linalg.norm(Xp[:2], axis=0), label=labels[name], **styles[name])
axs[0].set(xlabel="time (s)", ylabel=r"$\|\mathbf{e}\|/\|\mathbf{e}(0)\|$", xlim=(0, 10), title=r"step to $\boldsymbol{\theta}_d=(1.0,-0.7)$ rad")
axs[1].set(xlabel="time (s)", ylabel=r"$\|\mathbf{e}\|$ (rad)", xlim=(0, 10), title=r"push $(4,-2)$ N$\cdot$m for 0.3 s")
axs[0].legend(loc="upper right")
fig.tight_layout()
fig.savefig("figures/damping_compare.png", dpi=200)

with open("results/damping_results.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
print("done; figure in figures/damping_compare.png")
