"""
dynamics.py -- Lorenz ODE integrator for Homework 5.2 (next-token prediction for dynamical systems).

Lorenz system (dissipative, chaotic):
    x' = sigma (y - x),  y' = x (r - z) - y,  z' = x y - beta z
with sigma = 10, r = 28, beta = 8/3.

Main entry points:
  integrate(y0, T, dt)                       -> (t, Y) for one trajectory
  generate_trajectories(n_traj, T, dt, ...)  -> (t, Y) with Y of shape (n_traj, n_steps, 3)
"""

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

SIGMA, R, BETA = 10.0, 28.0, 8.0 / 3.0
DIM = 3
DEFAULT_TRANSIENT = 20.0  # time units discarded so samples lie on the attractor


def lorenz_rhs(t, s, sigma=SIGMA, r=R, beta=BETA):
    x, y, z = s
    return [sigma * (y - x), x * (r - z) - y, x * y - beta * z]


def random_ic(rng):
    # Broad box around the attractor; the transient is discarded later.
    return rng.uniform([-20, -25, 5], [20, 25, 45])


def integrate(y0, T: float, dt: float, rtol=1e-10, atol=1e-12, method="DOP853"):
    """Integrate one trajectory from y0 over [0, T], returning samples every dt.

    Returns (t, Y) with t of shape (n,) and Y of shape (n, 3).
    The sampling interval dt does NOT affect accuracy: the solver picks its own
    adaptive step size and only evaluates the solution at the requested times.
    """
    n = int(round(T / dt)) + 1
    t_eval = np.arange(n) * dt
    sol = solve_ivp(
        lorenz_rhs,
        (0.0, t_eval[-1]),
        y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y.T


def generate_trajectories(
    n_traj: int,
    T: float,
    dt: float,
    transient: Optional[float] = None,
    seed: int = 0,
    **kw,
):
    """Generate n_traj trajectories, each of duration T *after* discarding a transient.

    Returns
    -------
    t : (n_steps,)        time stamps (starting at 0 after the transient)
    Y : (n_traj, n_steps, 3)
    """
    if transient is None:
        transient = DEFAULT_TRANSIENT
    rng = np.random.default_rng(seed)

    n_keep = int(round(T / dt)) + 1
    n_skip = int(round(transient / dt))
    out = np.empty((n_traj, n_keep, DIM))
    for i in range(n_traj):
        y0 = random_ic(rng)
        _, Y = integrate(y0, T + transient, dt, **kw)
        out[i] = Y[n_skip : n_skip + n_keep]
    t = np.arange(n_keep) * dt
    return t, out


def _selftest():
    # ---- bounded, two-lobe switching ----
    t, Y = generate_trajectories(n_traj=5, T=200.0, dt=0.02, seed=1)
    print("Lorenz, 5 trajectories, T=200 (after 20 time-unit transient)")
    print(
        "  min/max per coordinate:",
        Y.min(axis=(0, 1)).round(1),
        Y.max(axis=(0, 1)).round(1),
    )
    for i in range(Y.shape[0]):
        sign = np.sign(Y[i, :, 0])
        switches = int(np.sum(sign[1:] != sign[:-1]))
        print(f"  traj {i}: lobe switches (sign changes of x) = {switches}")

    # ---- sensitive dependence ----
    rng = np.random.default_rng(7)
    y0 = random_ic(rng)
    _, A = integrate(y0, 40.0, 0.02)
    _, B = integrate(y0 + 1e-8, 40.0, 0.02)
    sep = np.linalg.norm(A - B, axis=1)
    t_div = np.argmax(sep > 1.0) * 0.02 if np.any(sep > 1.0) else np.inf
    print(
        f"Lorenz: 1e-8 perturbation reaches separation > 1 at t ~ {t_div:.1f} "
        f"(Lyapunov time ~ 1.1)"
    )


if __name__ == "__main__":
    _selftest()
