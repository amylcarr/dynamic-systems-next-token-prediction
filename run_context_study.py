"""Tasks 5-6: compare context lengths m = 1, 2, 3 on Lorenz tokens.

Generates (or loads cached) trajectories, splits by trajectory into train/test, fits the
box grid on the training data only, then trains and scores one MarkovModel per m.
"""
import os
import sys

import numpy as np

from dynamics import generate_trajectories
from markov import MarkovModel, evaluate, persistence_baseline
from tokenization import BoxGrid, tokenize_trajectories

DATA = "lorenz_data.npz"
N_TRAJ, T, DT = 200, 100.0, 0.05
N_BINS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_TRAIN = 150

if os.path.exists(DATA):
    d = np.load(DATA)
    Y = d["Y"]
    print(f"Loaded cached data {Y.shape}")
else:
    _, Y = generate_trajectories(N_TRAJ, T, DT, seed=0)
    np.savez(DATA, Y=Y, dt=DT)
    print(f"Generated and cached data {Y.shape}")

Y_train, Y_test = Y[:N_TRAIN], Y[N_TRAIN:]
grid = BoxGrid.fit(Y_train, n_bins=N_BINS)          # fit on training data only
tok_train = tokenize_trajectories(Y_train, grid)
tok_test = tokenize_trajectories(Y_test, grid)

n_train_tokens = tok_train.size
print(f"\nGrid {N_BINS}^3: V = {grid.vocab_size}, visited in train = {len(np.unique(tok_train))}")
print(f"Train: {tok_train.shape[0]} books, {n_train_tokens} tokens; "
      f"test: {tok_test.shape[0]} books, {tok_test.size} tokens")
print(f"Persistence baseline (predict same box): {persistence_baseline(tok_test):.3f}\n")

hdr = (f"{'m':>2} {'contexts':>9} {'coverage':>9} {'top1':>7} {'top1|seen':>10} "
       f"{'top3':>7} {'perplexity':>11}")
print(hdr)
print("-" * len(hdr))
for m in [0, 1, 2, 3]:
    model = MarkovModel(m, vocab_size=grid.vocab_size).fit(tok_train)
    r = evaluate(model, tok_test)
    print(f"{m:>2} {model.n_contexts:>9} {r['coverage']:>9.3f} {r['top1']:>7.3f} "
          f"{r['top1_seen']:>10.3f} {r['topk']:>7.3f} {r['perplexity']:>11.2f}")
