"""Tasks 7-8: generate token trajectories for m = 1, 2, 3 and compare with true Lorenz.

Usage: python3 run_generation.py [n_bins]     (default 16)
Needs lorenz_data.npz from run_context_study.py.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generation import (attractor_distance, attractor_tree, divergence_steps,
                        generate_tokens, lobe_stats, tv_distance)
from markov import MarkovModel
from tokenization import BoxGrid, tokenize_trajectories

N_BINS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_TRAIN, DIV_THRESHOLD = 150, 5.0     # separation (Lorenz units) counted as "diverged"
MS = [1, 2, 3]

d = np.load("lorenz_data.npz")
Y, dt = d["Y"], float(d["dt"])
Y_train, Y_test = Y[:N_TRAIN], Y[N_TRAIN:]
grid = BoxGrid.fit(Y_train, n_bins=N_BINS)
tok_train, tok_test = tokenize_trajectories(Y_train, grid), tokenize_trajectories(Y_test, grid)
n_steps = tok_test.shape[1]
tree = attractor_tree(Y_train)
true_centers = grid.decode(tok_test)                       # (n_test, n_steps, 3)

# ---------------------------------------------------------------- generate
rng = np.random.default_rng(0)
gen_tok, models = {}, {}
for m in MS:
    models[m] = MarkovModel(m, vocab_size=grid.vocab_size).fit(tok_train)
    gen_tok[m] = np.array([generate_tokens(models[m], t[:m], n_steps - m, rng) for t in tok_test])
gen_centers = {m: grid.decode(gen_tok[m]) for m in MS}

# ---------------------------------------------------------------- metrics
def lobes(centers):
    r = [lobe_stats(c, dt) for c in centers]
    return (np.mean([x["switches"] for x in r]), np.nanmean([x["mean_dwell"] for x in r]),
            np.mean([x["frac_pos"] for x in r]))

rows = []
dist_true = attractor_distance(true_centers, tree)
sw, dw, fp = lobes(true_centers)
rows.append(("true (test)", dist_true.mean(), np.percentile(dist_true, 99), (dist_true < 2).mean(),
             sw, dw, fp, 0.0, np.nan))
for m in MS:
    dg = attractor_distance(gen_centers[m], tree)
    sw, dw, fp = lobes(gen_centers[m])
    div = np.median([divergence_steps(t, g, DIV_THRESHOLD)
                     for t, g in zip(true_centers, gen_centers[m])]) * dt
    rows.append((f"m = {m}", dg.mean(), np.percentile(dg, 99), (dg < 2).mean(), sw, dw, fp,
                 tv_distance(gen_tok[m], tok_test, grid.vocab_size), div))

print(f"Grid {N_BINS}^3 (V = {grid.vocab_size}); {len(tok_test)} generated sequences x {n_steps} "
      f"steps ({n_steps * dt:.0f} time units each)\n")
hdr = (f"{'':>12} {'dist mean':>9} {'dist p99':>9} {'<2 units':>9} {'switches':>9} "
       f"{'dwell':>7} {'frac x>0':>9} {'TV dist':>8} {'diverge t':>10}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:>12} {r[1]:>9.2f} {r[2]:>9.2f} {r[3]:>9.3f} {r[4]:>9.1f} {r[5]:>7.2f} "
          f"{r[6]:>9.3f} {r[7]:>8.3f} {r[8]:>10.2f}")

# stuck / cyclic check: number of distinct tokens in the last 500 steps
print("\nDistinct tokens in the last 500 steps (median over sequences; low = trapped in a cycle):")
print(f"{'true':>8}: {np.median([len(set(t[-500:])) for t in tok_test]):.0f}")
for m in MS:
    print(f"{'m = ' + str(m):>8}: {np.median([len(set(t[-500:])) for t in gen_tok[m]]):.0f}")

# ---------------------------------------------------------------- figures
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})

# Figure 1: x-z projection, one sequence per panel (true vs generated)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
panels = [("True trajectory", true_centers[0], BLUE)] + \
         [(f"Generated, m = {m}", gen_centers[m][0], ORANGE) for m in MS]
for ax, (title, c, col) in zip(axes, panels):
    ax.plot(c[:, 0], c[:, 2], color=col, lw=0.5, alpha=0.8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("x")
axes[0].set_ylabel("z")
fig.suptitle(f"Lorenz x-z projection, {n_steps * dt:.0f} time units, {N_BINS}³ box grid",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("fig_xz_projection.png", dpi=200)

# Figure 2: x(t), true vs generated, first 40 time units, one panel per m
k = int(40 / dt)
t = np.arange(k) * dt
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
for ax, m in zip(axes, MS):
    ax.plot(t, true_centers[0, :k, 0], color=BLUE, lw=1.3, label="true")
    ax.plot(t, gen_centers[m][0, :k, 0], color=ORANGE, lw=1.3, label="generated")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_ylabel(f"x  (m = {m})")
axes[0].plot([], [], color=BLUE, lw=1.3, label="true")
axes[0].plot([], [], color=ORANGE, lw=1.3, label="generated")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", frameon=False, ncol=2)
axes[-1].set_xlabel("time")
fig.suptitle("x(t) from the same seed: generated and true paths separate within about one time unit",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("fig_x_vs_time.png", dpi=200)
print("\nSaved fig_xz_projection.png, fig_x_vs_time.png")
