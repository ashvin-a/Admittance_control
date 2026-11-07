"""Stage 1 (and the cheap parts of stage 2): verify the model and every number
quoted in main.tex. Run:  .venv/bin/python stage1_verify.py
"""
import numpy as np
from scipy.integrate import solve_ivp

import arm
from arm import M, C, tau_g, f, B, Cy, P

np.set_printoptions(precision=4, suppress=True, linewidth=120)
rng = np.random.default_rng(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1. M and tau_g against first-principles (kinetic / potential energy) ----------
errM = errG = 0.0
for _ in range(200):
    th = rng.uniform(-np.pi, np.pi, 2)
    errM = max(errM, np.abs(M(th) - arm.M_reference(th)).max())
    errG = max(errG, np.abs(tau_g(th) - arm.tau_g_reference(th)).max())
check("M(theta) matches sum m_i J_i^T J_i", errM < 1e-6, f"(max err {errM:.1e})")
check("tau_g(theta) matches -dV/dtheta", errG < 1e-6, f"(max err {errG:.1e})")

# 2. Coriolis term: C*thd must equal the Christoffel-symbol Coriolis force ------
def christoffel_force(th, thd, eps=1e-6):
    dM = np.zeros((2, 2, 2))  # dM[i, j, k] = dM_ij / dtheta_k
    for k in range(2):
        d = np.zeros(2); d[k] = eps
        dM[:, :, k] = (M(th + d) - M(th - d)) / (2 * eps)
    Cc = np.array([[sum(0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * thd[k] for k in range(2))
                    for j in range(2)] for i in range(2)])
    return Cc

errF = errS = errSc = 0.0
for _ in range(200):
    th, thd = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
    Cc = christoffel_force(th, thd)
    errF = max(errF, np.abs(Cc @ thd - C(th, thd) @ thd).max())
    Md = (M(th + 1e-6 * thd) - M(th - 1e-6 * thd)) / 2e-6
    S = Md - 2 * C(th, thd);  errS = max(errS, np.abs(S + S.T).max())
    S = Md - 2 * Cc;          errSc = max(errSc, np.abs(S + S.T).max())
check("C(theta,thd)*thd matches Christoffel Coriolis force", errF < 1e-6, f"(max err {errF:.1e})")

# closed-form C_ch quoted in the Remark of main.tex: skew and same Coriolis force
errCh = errChS = 0.0
for _ in range(200):
    th, thd = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
    hh = P.m2 * P.l1 * P.l2 * np.sin(th[1])
    Cch = np.array([[-hh * thd[1], -hh * (thd[0] + thd[1])], [hh * thd[0], 0.0]])
    Md = (M(th + 1e-6 * thd) - M(th - 1e-6 * thd)) / 2e-6
    S = Md - 2 * Cch
    errChS = max(errChS, np.abs(S + S.T).max())
    errCh = max(errCh, np.abs(Cch @ thd - C(th, thd) @ thd).max())
check("closed-form C_ch: same force as paper C, Mdot-2C_ch skew",
      errCh < 1e-9 and errChS < 1e-6, f"(force {errCh:.1e}, skew {errChS:.1e})")
print(f"   NOTE: the paper's C is NOT the skew-symmetric (Christoffel) factorization: "
      f"max |S+S^T| of Mdot-2C = {errS:.1f} (Christoffel: {errSc:.0e}).\n"
      f"         Dynamics are unaffected; a passivity/Lyapunov proof must use the Christoffel C.")

def energy(x):
    th, thd = x[:2], x[2:]
    V = P.g * (P.m1 * arm._positions(th)[0][1] + P.m2 * arm._positions(th)[1][1])
    return 0.5 * thd @ M(th) @ thd + V

x0 = np.array([0.5, -0.8, 1.0, 0.5])
sol = solve_ivp(lambda t, x: f(x, np.zeros(2)), [0, 10], x0, rtol=1e-10, atol=1e-12)
dE = np.ptp([energy(x) for x in sol.y.T])
check("open-loop (u=0) conserves energy over 10 s", dE < 1e-6, f"(drift {dE:.1e} J)")

# 3. Equilibrium and linearization --------------------------------------------
xs, us = np.zeros(4), np.zeros(2)
check("f(0,0) = 0", np.allclose(f(xs, us), 0))

def num_jac(fun, z0, eps=1e-6):
    cols = []
    for j in range(len(z0)):
        d = np.zeros(len(z0)); d[j] = eps
        cols.append((fun(z0 + d) - fun(z0 - d)) / (2 * eps))
    return np.array(cols).T

A_num = num_jac(lambda x: f(x, us), xs)
B_num = num_jac(lambda u: f(xs, u), us)

# analytic (Eq. 5)
g = P.g
M0 = M(np.zeros(2))
G = -g * np.array([[(P.m1 + P.m2) * P.l1 + P.m2 * P.l2, P.m2 * P.l2],
                   [P.m2 * P.l2, P.m2 * P.l2]])
Kg = np.linalg.solve(M0, G)
A = np.block([[np.zeros((2, 2)), np.eye(2)], [Kg, np.zeros((2, 2))]])
Blin = np.vstack([np.zeros((2, 2)), np.linalg.solve(M0, B)])

check("analytic A matches finite-difference Jacobian", np.allclose(A, A_num, atol=1e-5))
check("analytic B_lin matches finite-difference Jacobian", np.allclose(Blin, B_num, atol=1e-5))

A_paper = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-9.81, 9.81, 0, 0], [9.81, -29.43, 0, 0]])
B_paper = np.array([[0, 0], [0, 0], [1, -2], [-2, 5]])
check("A equals matrix printed in the proposal", np.allclose(A, A_paper))
check("B_lin equals matrix printed in the proposal", np.allclose(Blin, B_paper))
check("M0 = [[5,2],[2,1]], M0^-1 = [[1,-2],[-2,5]]",
      np.allclose(M0, [[5, 2], [2, 1]]) and np.allclose(np.linalg.inv(M0), [[1, -2], [-2, 5]]))

lam = np.linalg.eigvals(A)
mu = g * (2 + np.array([np.sqrt(2), -np.sqrt(2)]))
print("   open-loop eigenvalues:", np.sort_complex(lam))
check("eigenvalues are +/- j sqrt(mu), mu = g(2 +/- sqrt2)",
      np.allclose(np.sort(np.abs(lam.imag)), np.sort(np.sqrt(np.repeat(mu, 2)))) and np.allclose(lam.real, 0, atol=1e-9),
      f"(+/-j{np.sqrt(mu.min()):.2f}, +/-j{np.sqrt(mu.max()):.2f})")

# 4. Controllability / observability (Stage 2) ---------------------------------
Ctrb = np.hstack([np.linalg.matrix_power(A, k) @ Blin for k in range(4)])
Obsv = np.vstack([Cy @ np.linalg.matrix_power(A, k) for k in range(4)])
check("(A, B_lin) controllable", np.linalg.matrix_rank(Ctrb) == 4, f"(rank {np.linalg.matrix_rank(Ctrb)}/4)")
check("(A, C_y) observable", np.linalg.matrix_rank(Obsv) == 4, f"(rank {np.linalg.matrix_rank(Obsv)}/4)")

# 5. Closed-loop claim (Sec. IV): K = 20 I, D = 5 I, theta_d = 0 -----------------
K, D = 20 * np.eye(2), 5 * np.eye(2)
Md = M(np.zeros(2))
Mi = np.linalg.inv(Md)
Acl = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi @ K, -Mi @ D]])
ev = np.sort_complex(np.linalg.eigvals(Acl))
print("   closed-loop eigenvalues:", ev)
check("A_cl eigenvalues ~ -24.4, -4.79, -0.43 +/- j1.80",
      np.allclose(sorted(ev.real), [-24.4, -4.79, -0.43, -0.43], atol=0.05)
      and abs(np.abs(ev.imag).max() - 1.80) < 0.005)

# Cross-check A_cl against the numerical Jacobian of the true nonlinear closed loop
def closed_loop(z, th_d=np.zeros(2)):
    th, thd = z[:2] + th_d, z[2:]
    u = -tau_g(th) + K @ (th_d - th) - D @ thd
    return f(np.concatenate([th, thd]), u)

Acl_num = num_jac(closed_loop, np.zeros(4))
check("A_cl matches Jacobian of nonlinear closed loop", np.allclose(Acl, Acl_num, atol=1e-5))

th_d = np.array([0.6, -0.4])
Acl2 = num_jac(lambda z: closed_loop(z, th_d), np.zeros(4))
Mi2 = np.linalg.inv(M(th_d))
Acl2_th = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi2 @ K, -Mi2 @ D]])
check("A_cl formula (M_d at theta_d != 0) matches nonlinear Jacobian", np.allclose(Acl2, Acl2_th, atol=1e-5),
      f"(theta_d = {th_d}, max re(eig) = {np.linalg.eigvals(Acl2_th).real.max():.2f})")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
