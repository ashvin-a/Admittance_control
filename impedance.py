"""Impedance controller (main.tex Sec. IV), closed-loop models, and helpers.

Control law (Eq. 6):  u = -tau_g(theta) + K (theta_d - theta) - D thetadot
Closed loop:          M thdd + (C + D) thd + K (theta - theta_d) = tau_ext
"""
import numpy as np
from scipy.integrate import solve_ivp

import arm
from arm import M, tau_g


def control(x, th_d, K, D):
    return -tau_g(x[:2]) + K @ (th_d - x[:2]) - D @ x[2:]


def simulate_nonlinear(x0, t_end, K, D, th_d=None, tau_ext=None, max_step=0.005):
    """Integrate the full nonlinear arm under the impedance law.

    th_d and tau_ext are callables of time (or None -> zeros / 0).
    Returns (t, X) with X of shape (4, N).
    """
    th_d = th_d or (lambda t: np.zeros(2))
    tau_ext = tau_ext or (lambda t: np.zeros(2))

    def rhs(t, x):
        return arm.f(x, control(x, th_d(t), K, D), tau_ext(t))

    sol = solve_ivp(rhs, [0, t_end], x0, method="DOP853", rtol=1e-9, atol=1e-11,
                    max_step=max_step, dense_output=True)
    t = np.arange(0, t_end, 0.005)
    return t, sol.sol(t)


def linear_error_model(th_d, K, D):
    """z = [e; edot], e = theta - theta_d.  zdot = A z + Bw tau_ext (Eq. 7)."""
    Mi = np.linalg.inv(M(np.asarray(th_d, float)))
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi @ K, -Mi @ D]])
    Bw = np.vstack([np.zeros((2, 2)), Mi])
    return A, Bw


def simulate_linear(z0, t_end, A, Bw, tau_ext=None, max_step=0.005):
    tau_ext = tau_ext or (lambda t: np.zeros(2))
    sol = solve_ivp(lambda t, z: A @ z + Bw @ tau_ext(t), [0, t_end], z0, method="DOP853",
                    rtol=1e-10, atol=1e-12, max_step=max_step, dense_output=True)
    t = np.arange(0, t_end, 0.005)
    return t, sol.sol(t)


def settling_time(t, e, tol):
    """Last time the error norm exceeds tol (0 if it never does)."""
    n = np.linalg.norm(e, axis=0)
    idx = np.where(n > tol)[0]
    return 0.0 if len(idx) == 0 else t[idx[-1]]


def modal_damping(th_d, k, d):
    """For K = kI, D = dI the closed loop decouples along the eigenvectors of
    M(theta_d): each eigenvalue m_i of M_d gives  m_i s^2 + d s + k = 0.
    Returns [(m_i, wn_i, zeta_i)]."""
    m = np.linalg.eigvalsh(M(np.asarray(th_d, float)))
    return [(mi, np.sqrt(k / mi), d / (2 * np.sqrt(k * mi))) for mi in m]
