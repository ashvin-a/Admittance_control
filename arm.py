"""2-DOF arm model, exactly as written in main.tex (Sec. III).

Joint angles are zero when the arm hangs straight down; theta2 is relative to
link 1. State x = [theta1, theta2, dtheta1, dtheta2]; input u = [tau1, tau2].
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81


P = Params()
B = np.eye(2)  # u -> joint torques (both joints directly actuated)
Cy = np.hstack([np.eye(2), np.zeros((2, 2))])  # y = [theta1, theta2]


def M(th, p=P):
    c2 = np.cos(th[1])
    a = p.m2 * p.l1 * p.l2 * c2
    return np.array([
        [(p.m1 + p.m2) * p.l1**2 + p.m2 * p.l2**2 + 2 * a, p.m2 * p.l2**2 + a],
        [p.m2 * p.l2**2 + a, p.m2 * p.l2**2],
    ])


def C(th, thd, p=P):
    s2 = np.sin(th[1])
    k = p.m2 * p.l1 * p.l2 * s2
    w = 2 * thd[0] + thd[1]
    return np.array([
        [0.0, -k * w],
        [0.5 * k * w, -0.5 * k * thd[0]],
    ])


def tau_g(th, p=P):
    s1, s12 = np.sin(th[0]), np.sin(th[0] + th[1])
    return -p.g * np.array([(p.m1 + p.m2) * p.l1 * s1 + p.m2 * p.l2 * s12,
                            p.m2 * p.l2 * s12])


def theta_ddot(th, thd, u, tau_ext=(0.0, 0.0), p=P):
    """Eq. (2): solve the EOM for the joint accelerations."""
    rhs = -C(th, thd, p) @ thd + tau_g(th, p) + B @ u + np.asarray(tau_ext)
    return np.linalg.solve(M(th, p), rhs)


def f(x, u, tau_ext=(0.0, 0.0), p=P):
    """Eq. (3): xdot = f(x, u) (tau_ext = 0 by default)."""
    x = np.asarray(x, float)
    return np.concatenate([x[2:], theta_ddot(x[:2], x[2:], np.asarray(u, float), tau_ext, p)])


def h(x, u=None):
    """Output y = C_y x."""
    return Cy @ np.asarray(x, float)


# ---- first-principles reference (independent of the formulas above) --------
def _positions(th, p=P):
    p1 = p.l1 * np.array([np.sin(th[0]), -np.cos(th[0])])
    p2 = p1 + p.l2 * np.array([np.sin(th[0] + th[1]), -np.cos(th[0] + th[1])])
    return p1, p2


def _jac(th, p=P, eps=1e-6):
    """Jacobians of the two point-mass positions wrt (theta1, theta2)."""
    J1 = np.zeros((2, 2)); J2 = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2); d[j] = eps
        a1, a2 = _positions(th + d, p); b1, b2 = _positions(th - d, p)
        J1[:, j] = (a1 - b1) / (2 * eps)
        J2[:, j] = (a2 - b2) / (2 * eps)
    return J1, J2


def M_reference(th, p=P):
    """M = sum_i m_i J_i^T J_i from point-mass kinetic energy."""
    J1, J2 = _jac(np.asarray(th, float), p)
    return p.m1 * J1.T @ J1 + p.m2 * J2.T @ J2


def tau_g_reference(th, p=P, eps=1e-6):
    """tau_g = -dV/dtheta with V = g (m1 y1 + m2 y2)."""
    th = np.asarray(th, float)
    V = lambda t: p.g * (p.m1 * _positions(t, p)[0][1] + p.m2 * _positions(t, p)[1][1])
    out = np.zeros(2)
    for j in range(2):
        d = np.zeros(2); d[j] = eps
        out[j] = -(V(th + d) - V(th - d)) / (2 * eps)
    return out
