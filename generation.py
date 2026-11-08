"""
generation.py -- generate token trajectories from a MarkovModel and compare them with the
true Lorenz dynamics (Tasks 7-8 of Homework 5.2).

generate_tokens   : roll the model forward from a real seed context.
attractor_distance: distance from decoded box centers to the nearest true training point.
lobe_stats        : two-lobe switching statistics (sign of x), per decoded sequence.
divergence_steps  : steps until a generated sequence separates from its true trajectory.
tv_distance       : total variation distance between two token histograms.
"""

import numpy as np
from scipy.spatial import cKDTree


def generate_tokens(model, seed_tokens, n_steps, rng):
    """Return seed_tokens followed by n_steps sampled tokens (length m + n_steps).

    At every step the next token is drawn from the model's empirical distribution given the
    previous m tokens, so the output is a random trajectory, not the single most likely path.
    """
    m = model.m
    seq = [int(a) for a in seed_tokens]
    for _ in range(n_steps):
        ctx = seq[-m:] if m > 0 else ()
        seq.append(model.sample(ctx, rng))
    return np.array(seq)


def attractor_tree(Y_train):
    """KD-tree over the true training points (the empirical Lorenz attractor)."""
    return cKDTree(np.asarray(Y_train).reshape(-1, 3))


def attractor_distance(centers, tree):
    """Distance from each point (..., 3) to the nearest training point."""
    d, _ = tree.query(np.asarray(centers).reshape(-1, 3))
    return d


def lobe_stats(centers, dt):
    """Lobe switching of one decoded sequence (n_steps, 3), using the sign of x.

    Returns the number of switches, the mean dwell time (time units between switches) and
    the fraction of time spent in the x > 0 lobe.
    """
    s = np.sign(centers[:, 0])
    s[s == 0] = 1
    flips = np.flatnonzero(s[1:] != s[:-1])
    n = len(flips)
    dwell = np.diff(flips) * dt if n > 1 else np.array([np.nan])
    return dict(switches=n, mean_dwell=float(np.nanmean(dwell)), frac_pos=float((s > 0).mean()))


def divergence_steps(true_centers, gen_centers, threshold):
    """First step at which |generated - true| exceeds threshold (len if it never does)."""
    d = np.linalg.norm(true_centers - gen_centers, axis=1)
    over = np.flatnonzero(d > threshold)
    return int(over[0]) if len(over) else len(d)


def tv_distance(tok_a, tok_b, vocab_size):
    """Total variation distance between the token histograms of two sets of sequences."""
    pa = np.bincount(np.ravel(tok_a), minlength=vocab_size) / np.size(tok_a)
    pb = np.bincount(np.ravel(tok_b), minlength=vocab_size) / np.size(tok_b)
    return 0.5 * float(np.abs(pa - pb).sum())
