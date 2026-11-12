"""Structural analysis: controllability, observability, stabilizability, detectability.

Run:  .venv/bin/python stage2_structural.py   (writes results/stage2_structural.txt)

Everything is computed from the NONLINEAR model in arm.py (numerical Jacobians), so it
does not rely on the closed forms quoted in main.tex.
"""
import itertools
import sys

import numpy as np
import scipy.linalg as sla

import arm
from arm import f, P, Cy
from impedance import linear_error_model

np.set_printoptions(precision=4, suppress=True, linewidth=130)
TOL = 1e-8


def jac(fun, z0, eps=1e-6):
    z0 = np.asarray(z0, float)
    f0 = np.atleast_1d(fun(z0))
    J = np.zeros((len(f0), len(z0)))
    for j in range(len(z0)):
        d = np.zeros(len(z0)); d[j] = eps
        J[:, j] = (np.atleast_1d(fun(z0 + d)) - np.atleast_1d(fun(z0 - d))) / (2 * eps)
    return J


def linearize(x_eq, u_eq=(0.0, 0.0), p=P):
    x_eq, u_eq = np.asarray(x_eq, float), np.asarray(u_eq, float)
    A = jac(lambda x: f(x, u_eq, p=p), x_eq)
    B = jac(lambda u: f(x_eq, u, p=p), u_eq)
    return A, B


def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def obsv(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def rank(Mx):
    return int(np.linalg.matrix_rank(Mx, tol=TOL * max(1.0, np.abs(Mx).max())))


def pbh_ctrb(A, B):
    """Per eigenvalue: (lambda, sigma_min[lambda I - A, B]). Zero => uncontrollable mode."""
    out = []
    for lam in np.linalg.eigvals(A):
        s = np.linalg.svd(np.hstack([lam * np.eye(len(A)) - A, B]), compute_uv=False)
        out.append((lam, s[-1]))
    return out


def classify(A, B, C):
    """Return dict with ranks and the four structural properties via PBH."""
    n = A.shape[0]
    rc, ro = rank(ctrb(A, B)), rank(obsv(A, C))
    pc = pbh_ctrb(A, B)
    po = [(l, s) for l, s in pbh_ctrb(A.T, C.T)]  # PBH observability = PBH ctrb of dual
    unctrl = [l for l, s in pc if s < 1e-7]
    unobs = [l for l, s in po if s < 1e-7]
    # stabilizable: every uncontrollable eigenvalue has Re < 0; detectable: dual
    stab = all(l.real < -1e-9 for l in unctrl)
    det = all(l.real < -1e-9 for l in unobs)
    return dict(rc=rc, ro=ro, ctrb=rc == n, obsv=ro == n, stab=stab, det=det,
                unctrl=unctrl, unobs=unobs, eigs=np.linalg.eigvals(A),
                smin_c=np.linalg.svd(ctrb(A, B), compute_uv=False)[-1] if B.shape[1] * n >= n else 0.0)


def fmt_eigs(e):
    e = sorted(e, key=lambda z: (round(z.real, 4), round(abs(z.imag), 4), z.imag))
    return "[" + ", ".join(f"{z.real:+.3f}{z.imag:+.3f}j" if abs(z.imag) > 1e-6 else f"{z.real:+.3f}" for z in e) + "]"


def yn(b):
    return "yes" if b else "NO "


out = []
def P_(s=""):
    print(s); out.append(s)


# ---------------------------------------------------------------------------
P_("=== 1. All equilibria with u = 0 (theta_i in {0, pi}, zero velocity), full actuation, y = theta ===")
P_("name            eigenvalues of A                                      ctrb(rank) obsv(rank) stabilizable detectable")
names = {(0, 0): "down-down", (0, np.pi): "down-up", (np.pi, 0): "up-straight", (np.pi, np.pi): "up-folded"}
B2 = np.eye(2)
for (a, b), nm in names.items():
    xe = np.array([a, b, 0.0, 0.0])
    assert np.allclose(f(xe, np.zeros(2)), 0, atol=1e-9), nm
    A, B = linearize(xe)
    r = classify(A, B, Cy)
    P_(f"{nm:<15} {fmt_eigs(r['eigs']):<52} {yn(r['ctrb'])}({r['rc']})    {yn(r['obsv'])}({r['ro']})    {yn(r['stab'])}         {yn(r['det'])}")
P_("-> controllability/observability are properties of (A,B,C), NOT of stability: the unstable")
P_("   equilibria are just as controllable and observable, so they can be stabilized by feedback.")

# ---------------------------------------------------------------------------
P_()
P_("=== 2. Hanging equilibrium: loss of an actuator (B column removed), y = theta ===")
A0, B0 = linearize(np.zeros(4))
P_("config                      ctrb rank  stabilizable   unctrl. modes")
for nm, cols in [("both motors", [0, 1]), ("shoulder only (Pendubot)", [0]), ("elbow only (Acrobot)", [1])]:
    r = classify(A0, B0[:, cols], Cy)
    P_(f"{nm:<27} {r['rc']}/4        {yn(r['stab'])}          {fmt_eigs(r['unctrl']) if r['unctrl'] else '-'}")

P_()
P_("=== 3. Hanging equilibrium: sensor sets (C rows), both motors ===")
E = np.eye(4)
sensors = {
    "theta1, theta2 (nominal)": [0, 1],
    "theta1 only": [0],
    "theta2 only": [1],
    "dtheta1, dtheta2 only": [2, 3],
    "dtheta1 only": [2],
    "dtheta2 only": [3],
    "theta1 + dtheta2": [0, 3],
}
P_("sensors                     obsv rank  detectable   unobs. modes")
for nm, rows in sensors.items():
    r = classify(A0, B0, E[rows])
    P_(f"{nm:<27} {r['ro']}/4        {yn(r['det'])}          {fmt_eigs(r['unobs']) if r['unobs'] else '-'}")
P_("-> every single sensor suffices, even velocity-only: gravity makes K_g = M0^-1 G invertible, so")
P_("   C A = [K_g-coupled rows] recovers positions from accelerations. The open loop has NO")
P_("   dissipation (eigs on j-axis), so if a mode WERE lost it would be neither decaying nor")
P_("   growing => detectability/stabilizability would fail too. Sec. 3b shows exactly that.")

# ---------------------------------------------------------------------------
P_()
P_("=== 3b. Same arm in a HORIZONTAL plane (g = 0): gravity no longer couples the joints ===")
p0 = arm.Params(g=0.0)
Ah, Bh = linearize(np.zeros(4), p=p0)
P_(f"A eigenvalues = {fmt_eigs(np.linalg.eigvals(Ah))}  (two double integrators)")
P_("case                              ctrb  obsv  stabilizable detectable  lost modes (all at s = 0)")
cases = [("both motors, y = theta", [0, 1], [0, 1]),
         ("shoulder only, y = theta", [0], [0, 1]),
         ("elbow only, y = theta", [1], [0, 1]),
         ("both motors, y = dtheta", [0, 1], [2, 3]),
         ("both motors, y = theta1", [0, 1], [0]),
         ("both motors, y = theta2", [0, 1], [1])]
for nm, cols, rows in cases:
    r = classify(Ah, Bh[:, cols], E[rows])
    lost = []
    if r["rc"] < 4: lost.append(f"{4 - r['rc']} uncontrollable")
    if r["ro"] < 4: lost.append(f"{4 - r['ro']} unobservable")
    P_(f"{nm:<33} {r['rc']}/4   {r['ro']}/4   {yn(r['stab'])}          {yn(r['det'])}        "
       f"{', '.join(lost) or '-'}")
P_("-> without gravity, one motor cannot move the other joint's double integrator independently")
P_("   (rank 2, lost modes at s=0, not stabilizable), and velocity sensing cannot recover absolute")
P_("   angle (not detectable). Gravity is what makes the underactuated / velocity-only cases work.")

# ---------------------------------------------------------------------------
P_()
P_("=== 4. Single-actuator arms at other equilibria (is the lost-actuator arm still stabilizable?) ===")
P_("equilibrium    actuator    ctrb rank  stabilizable")
for (a, b), nm in names.items():
    xe = np.array([a, b, 0.0, 0.0])
    A, B = linearize(xe)
    for an, col in [("shoulder", [0]), ("elbow", [1])]:
        r = classify(A, B[:, col], Cy)
        P_(f"{nm:<14} {an:<10}  {r['rc']}/4        {yn(r['stab'])}")

# ---------------------------------------------------------------------------
P_()
P_("=== 5. When CAN a single motor / single angle sensor lose a mode? (hanging equilibrium, g = 9.81) ===")
P_("Modes: G v = -mu M0 v. With b_j = [0; M0^-1 e_j], the left eigvec of A gives w^T b_j ∝ v^T e_j = v_j;")
P_("with y = theta_j, C x = v_j as well. So joint j's motor (or angle sensor) misses a mode iff that")
P_("mode shape has v_j = 0. Solving G e_1 ∝ M0 e_1 gives  m1*l1*l2 = 0  (elbow-only / theta2-only),")
P_("and G e_2 ∝ M0 e_2 gives  l1 = 0  (shoulder-only / theta1-only). With l1, l2 > 0 the only")
P_("candidate is m1 = 0, where M0 itself is singular (degenerate). Numerical check, sweeping m1:")
P_("m1       elbow margin min|v_2|/|v|   shoulder margin min|v_1|/|v|   Kalman rank, elbow-only (numerical)")
for m1 in [1.0, 0.1, 0.01, 1e-3, 1e-4]:
    p = arm.Params(m1=m1)
    A, B = linearize(np.zeros(4), p=p)
    G = -p.g * np.array([[(p.m1 + p.m2) * p.l1 + p.m2 * p.l2, p.m2 * p.l2], [p.m2 * p.l2, p.m2 * p.l2]])
    _, V = sla.eig(G, arm.M(np.zeros(2), p))
    vn = [np.abs(V[:, i]) / np.linalg.norm(V[:, i]) for i in range(2)]
    P_(f"{m1:<8g} {min(v[1] for v in vn):.2e}                    {min(v[0] for v in vn):.2e}"
       f"                       {rank(ctrb(A, B[:, [1]]))}/4")
P_("-> the elbow margin shrinks ~ m1/2 but is never zero for a real arm: the 'swing with elbow")
P_("   locked' mode is barely excited by elbow torque when the upper link is light. The shoulder")
P_("   margin stays O(1). The Kalman rank test already reports rank 2 at m1 = 0.01 although the")
P_("   system IS controllable: the Krylov matrix [b Ab A^2b A^3b] is badly conditioned, so use the")
P_("   PBH / modal test (or a Gramian) to decide near-degenerate cases, not a numerical rank.")

# ---------------------------------------------------------------------------
P_()
P_("=== 6. Closed loop (impedance law, K=20I, D=5I): error model (A_cl, Bw), disturbance channel ===")
for th_d in [np.zeros(2), np.array([0.5, 0.5]), np.array([1.0, -1.0])]:
    Acl, Bw = linear_error_model(th_d, 20 * np.eye(2), 5 * np.eye(2))
    Cz = np.hstack([np.eye(2), np.zeros((2, 2))])
    r = classify(Acl, Bw, Cz)
    P_(f"theta_d={th_d}: Hurwitz={yn(np.all(r['eigs'].real < 0))} ctrb(tau_ext)={r['rc']}/4 obsv(e)={r['ro']}/4 "
       f"max Re(eig)={r['eigs'].real.max():.3f}")

P_()
P_("=== 7. Closed-loop Gramians / Hankel singular values (stable => infinite-horizon Gramians exist) ===")
Acl, Bw = linear_error_model(np.zeros(2), 20 * np.eye(2), 5 * np.eye(2))
Cz = np.hstack([np.eye(2), np.zeros((2, 2))])
Wc = sla.solve_continuous_lyapunov(Acl, -Bw @ Bw.T)
Wo = sla.solve_continuous_lyapunov(Acl.T, -Cz.T @ Cz)
hsv = np.sqrt(np.sort(np.linalg.eigvals(Wc @ Wo).real)[::-1])
e3 = lambda v: "[" + ", ".join(f"{x:.3e}" for x in v) + "]"
P_(f"Wc eigenvalues = {e3(np.sort(np.linalg.eigvalsh(Wc)))}")
P_(f"Wo eigenvalues = {e3(np.sort(np.linalg.eigvalsh(Wo)))}")
P_(f"Hankel singular values = {e3(hsv)}")
P_(f"min-energy to reach e=[0.1,0], de=0 via tau_ext: x' Wc^-1 x = {np.array([0.1,0,0,0]) @ np.linalg.solve(Wc, np.array([0.1,0,0,0])):.3f}")

P_()
P_("=== 8. Stabilizability shown constructively: linearized impedance law is a state feedback u = F x ===")
A0, B0 = linearize(np.zeros(4))
G0 = jac(lambda th: arm.tau_g(th), np.zeros(2))
F = -np.hstack([G0 + 20 * np.eye(2), 5 * np.eye(2)])  # u = -tau_g - K theta - D thetadot, theta_d = 0
P_(f"eig(A + B F) = {fmt_eigs(np.linalg.eigvals(A0 + B0 @ F))}  (same as A_cl in Sec. 6, all Re < 0)")

with open("results/stage2_structural.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
