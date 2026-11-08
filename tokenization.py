"""
tokenization.py -- discretize Lorenz phase space into boxes (Tasks 3-4 of Homework 5.2).

A BoxGrid divides a bounding box of R^3 into n_x * n_y * n_z equal boxes. Each box is one
token, so the full vocabulary has size V = n_x * n_y * n_z. Only a small fraction of these
boxes is ever visited, because the trajectory lives near the Lorenz attractor.

Typical use:
    from dynamics import generate_trajectories
    from tokenization import BoxGrid, tokenize_trajectories

    t, Y = generate_trajectories(n_traj=200, T=100.0, dt=0.05, seed=0)
    grid = BoxGrid.fit(Y, n_bins=16)
    tokens = tokenize_trajectories(Y, grid)      # (n_traj, n_steps) integer array
    centers = grid.decode(tokens)                # back to (n_traj, n_steps, 3) box centers
"""

import numpy as np


class BoxGrid:
    """Uniform grid of boxes over [lo, hi] in each coordinate.

    Parameters
    ----------
    lo, hi : length-3 arrays, the bounding box of the grid.
    n_bins : int or length-3 sequence, number of boxes along each axis.
    """

    def __init__(self, lo, hi, n_bins):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        n = np.broadcast_to(np.asarray(n_bins, dtype=int), self.lo.shape)
        self.n_bins = n.copy()
        self.width = (self.hi - self.lo) / self.n_bins
        # strides for flattening (i, j, k) -> i * ny * nz + j * nz + k
        self._strides = np.array([self.n_bins[1] * self.n_bins[2], self.n_bins[2], 1])

    @classmethod
    def fit(cls, Y, n_bins, margin=0.5):
        """Fit the bounding box to data Y of shape (..., 3), padded by `margin` on each side."""
        flat = np.asarray(Y).reshape(-1, 3)
        lo = flat.min(axis=0) - margin
        hi = flat.max(axis=0) + margin
        return cls(lo, hi, n_bins)

    @property
    def vocab_size(self):
        return int(np.prod(self.n_bins))

    def encode(self, Y):
        """Map points Y of shape (..., 3) to integer tokens of shape (...).

        Points outside the grid are clipped to the nearest boundary box. This matters when
        encoding held-out trajectories that reach slightly beyond the training bounds.
        """
        Y = np.asarray(Y)
        idx = np.floor((Y - self.lo) / self.width).astype(int)
        idx = np.clip(idx, 0, self.n_bins - 1)
        return idx @ self._strides

    def decode(self, tokens):
        """Map integer tokens of shape (...) to box centers of shape (..., 3)."""
        tokens = np.asarray(tokens)
        i = tokens // self._strides[0]
        rem = tokens % self._strides[0]
        j = rem // self._strides[1]
        k = rem % self._strides[1]
        idx = np.stack([i, j, k], axis=-1)
        return self.lo + (idx + 0.5) * self.width


def tokenize_trajectories(Y, grid):
    """Convert trajectories Y (n_traj, n_steps, 3) to tokens (n_traj, n_steps).

    Each row is one "book". Sampling at a fixed interval dt was already done when Y was
    generated, so this is just the box lookup.
    """
    return grid.encode(Y)


def occupancy_stats(tokens, grid):
    """Summary of how much of the vocabulary is actually used."""
    visited = np.unique(tokens)
    return dict(
        vocab_size=grid.vocab_size,
        visited=len(visited),
        occupancy=len(visited) / grid.vocab_size,
    )


def self_transition_rate(tokens):
    """Fraction of consecutive steps where the token does not change.

    A rate near 1 means dt is too small for this grid: the model would mostly learn
    "stay in the same box". A rate near 0 means the trajectory jumps several boxes per step.
    """
    same = tokens[:, 1:] == tokens[:, :-1]
    return float(same.mean())


def reconstruction_error(Y, grid):
    """Mean Euclidean distance between each point and the center of its box."""
    centers = grid.decode(grid.encode(Y))
    return float(np.linalg.norm(Y - centers, axis=-1).mean())


def _selftest():
    from dynamics import generate_trajectories

    t, Y = generate_trajectories(n_traj=200, T=100.0, dt=0.05, seed=0)
    print(f"Data: {Y.shape[0]} trajectories x {Y.shape[1]} steps (dt=0.05)")

    # encode/decode round trip: decoded centers re-encode to the same tokens
    grid = BoxGrid.fit(Y, n_bins=16)
    tok = tokenize_trajectories(Y, grid)
    assert np.array_equal(grid.encode(grid.decode(tok)), tok), "round trip failed"
    assert tok.min() >= 0 and tok.max() < grid.vocab_size
    print("Round-trip encode(decode(tokens)) == tokens: OK\n")

    header = f"{'bins/axis':>9} {'V':>7} {'visited':>8} {'occupancy':>10} {'self-trans':>11} {'recon err':>10}"
    print(header)
    print("-" * len(header))
    for n in [4, 8, 16, 32, 64]:
        grid = BoxGrid.fit(Y, n_bins=n)
        tok = tokenize_trajectories(Y, grid)
        s = occupancy_stats(tok, grid)
        print(f"{n:>9} {s['vocab_size']:>7} {s['visited']:>8} {s['occupancy']:>10.3f} "
              f"{self_transition_rate(tok):>11.3f} {reconstruction_error(Y, grid):>10.3f}")

    # effect of the sampling interval on the self-transition rate (fixed grid, n=16)
    print("\nEffect of sampling interval dt (16 bins/axis, T=100 each):")
    print(f"{'dt':>6} {'self-trans':>11}")
    for dt in [0.01, 0.02, 0.05, 0.1, 0.2]:
        _, Yd = generate_trajectories(n_traj=50, T=100.0, dt=dt, seed=0)
        g = BoxGrid.fit(Y, n_bins=16)   # keep the grid fixed across dt
        print(f"{dt:>6} {self_transition_rate(tokenize_trajectories(Yd, g)):>11.3f}")


if __name__ == "__main__":
    _selftest()
