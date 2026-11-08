"""Task 10: vary the discretization. Sweep grid resolution and measure, for m = 1, 2, 3,
what each choice costs and buys.

Usage: python3 run_discretization.py
Needs lorenz_data.npz (from run_context_study.py). Writes discretization_results.csv and
fig_discretization.png.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generation import (attractor_distance, attractor_tree, divergence_steps,
                        generate_tokens, lobe_stats, tv_distance)
from markov import MarkovModel, evaluate, persistence_baseline
from tokenization import (BoxGrid, occupancy_stats, reconstruction_error,
                          self_transition_rate, tokenize_trajectories)

BINS = [4, 8, 16, 32, 64]
MS = [1, 2, 3]
N_TRAIN, DIV_THRESHOLD = 150, 5.0

d = np.load("lorenz_data.npz")
Y, dt = d["Y"], float(d["dt"])
Y_train, Y_test = Y[:N_TRAIN], Y[N_TRAIN:]
tree = attractor_tree(Y_train)
n_steps = Y_test.shape[1]

rows = []
for nb in BINS:
    grid = BoxGrid.fit(Y_train, n_bins=nb)
    tok_train, tok_test = tokenize_trajectories(Y_train, grid), tokenize_trajectories(Y_test, grid)
    occ = occupancy_stats(tok_train, grid)
    common = dict(bins=nb, V=grid.vocab_size, visited=occ["visited"],
                  recon_err=reconstruction_error(Y_test, grid),
                  self_trans=self_transition_rate(tok_test),
                  persistence=persistence_baseline(tok_test))
    for m in MS:
        model = MarkovModel(m, vocab_size=grid.vocab_size).fit(tok_train)
        ev = evaluate(model, tok_test)
        rng = np.random.default_rng(0)
        gen = np.array([generate_tokens(model, t[:m], n_steps - m, rng) for t in tok_test])
        gc = grid.decode(gen)
        dist = attractor_distance(gc, tree)
        ls = [lobe_stats(c, dt) for c in gc]
        # divergence measured against the *raw* true trajectory, so grids are comparable
        div = np.median([divergence_steps(t, g, DIV_THRESHOLD) for t, g in zip(Y_test, gc)]) * dt
        rows.append(dict(common, m=m, contexts=model.n_contexts, coverage=ev["coverage"],
                         top1=ev["top1"], top3=ev["topk"], perplexity=ev["perplexity"],
                         dist_mean=float(dist.mean()),
                         switches=float(np.mean([x["switches"] for x in ls])),
                         tv=tv_distance(gen, tok_test, grid.vocab_size), diverge_t=div))
    print(f"done {nb} bins", flush=True)

# ---------------------------------------------------------------- table + csv
with open("discretization_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nGrid properties (test data)")
print(f"{'bins':>5} {'V':>8} {'visited':>8} {'recon err':>10} {'self-trans':>11}")
for r in [r for r in rows if r["m"] == 1]:
    print(f"{r['bins']:>5} {r['V']:>8} {r['visited']:>8} {r['recon_err']:>10.2f} {r['self_trans']:>11.3f}")

print("\nPrediction and generation")
print(f"{'bins':>5} {'m':>2} {'contexts':>9} {'coverage':>9} {'top1':>6} {'top3':>6} "
      f"{'perplex':>8} {'switches':>9} {'TV':>6} {'diverge t':>10}")
for r in rows:
    print(f"{r['bins']:>5} {r['m']:>2} {r['contexts']:>9} {r['coverage']:>9.3f} {r['top1']:>6.3f} "
          f"{r['top3']:>6.3f} {r['perplexity']:>8.2f} {r['switches']:>9.1f} {r['tv']:>6.3f} "
          f"{r['diverge_t']:>10.2f}")

# ---------------------------------------------------------------- figure
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
COL = {1: BLUE, 2: ORANGE, 3: AQUA}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6})
first = [r for r in rows if r["m"] == 1]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))

ax = axes[0]
ax.plot(BINS, [r["V"] for r in first], color=BLUE, marker="o", lw=1.6, label="V (all boxes)")
ax.plot(BINS, [r["visited"] for r in first], color=ORANGE, marker="o", lw=1.6, label="visited by data")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_title("Vocabulary", loc="left"); ax.set_ylabel("tokens"); ax.legend(frameon=False)

ax = axes[1]
ax.plot(BINS, [r["recon_err"] for r in first], color=BLUE, marker="o", lw=1.6)
ax.set_xscale("log", base=2)
ax.set_title("Geometry kept: reconstruction error", loc="left"); ax.set_ylabel("mean distance to box center")

ax = axes[2]
for m in MS:
    rr = [r for r in rows if r["m"] == m]
    ax.plot(BINS, [r["top1"] for r in rr], color=COL[m], marker="o", lw=1.6, label=f"m = {m}")
ax.set_xscale("log", base=2); ax.set_ylim(0, 1)
ax.set_title("Predictability: top-1 accuracy", loc="left"); ax.legend(frameon=False)

ax = axes[3]
for m in MS:
    rr = [r for r in rows if r["m"] == m]
    ax.plot(BINS, [r["diverge_t"] for r in rr], color=COL[m], marker="o", lw=1.6, label=f"m = {m}")
ax.set_xscale("log", base=2)
ax.set_title("Generation: time to diverge", loc="left"); ax.set_ylabel("median time units")
ax.legend(frameon=False)

for ax in axes:
    ax.set_xlabel("boxes per axis"); ax.set_xticks(BINS); ax.set_xticklabels(BINS)
    ax.minorticks_off()
fig.tight_layout()
fig.savefig("fig_discretization.png", dpi=200)
print("\nSaved discretization_results.csv, fig_discretization.png")
