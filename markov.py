"""
markov.py -- count-based m-step next-token model (Tasks 5-6 of Homework 5.2).

The m-step model estimates

    P(a_{n+1} = b | a_{n-m+1..n}) = #{(a_{n-m+1..n}, b)} / #{(a_{n-m+1..n})}

by counting in the training sequences. m = 1 is the first-order Markov chain from the
assignment. Counts are stored sparsely: a dict from the context tuple to a Counter of
next tokens, so memory scales with the number of *observed* contexts, not with V^m.

Unseen contexts: the plain maximum-likelihood estimate is undefined there. The model backs
off to the longest suffix of the context that was seen in training (down to m = 0, the
unigram distribution), and reports how often it had to do so (`coverage`).

Typical use:
    model = MarkovModel(m=2).fit(train_tokens)
    counts, used_order = model.lookup(context)      # raw counts for the next token
    p = model.prob(context, b)                      # smoothed probability of token b
    model.sample(context, rng)                      # draw the next token
"""

from collections import Counter, defaultdict

import numpy as np


class MarkovModel:
    """Order-m count model with back-off to shorter contexts.

    Parameters
    ----------
    m : context length (number of previous tokens used).
    vocab_size : total V of the grid; used only to smooth probabilities for scoring.
    smoothing : weight lam of a uniform distribution mixed into scored probabilities,
        p = (1 - lam) * p_MLE + lam / V. This keeps log-likelihood finite when a test
        token never followed its context in training. It does not change top-k accuracy
        and is not used by `sample`.
    """

    def __init__(self, m, vocab_size=None, smoothing=1e-3):
        self.m = m
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # tables[k][context of length k] -> Counter of next tokens, for k = 0..m
        self.tables = [defaultdict(Counter) for _ in range(m + 1)]

    # ------------------------------------------------------------------ fit
    def fit(self, sequences):
        """sequences: (n_seq, n_steps) integer array, or a list of 1-D arrays.

        Each sequence is counted separately, so contexts never straddle two "books".
        Shorter orders (used for back-off) are counted from the same data.
        """
        for seq in sequences:
            seq = [int(a) for a in seq]
            for n in range(len(seq)):
                nxt = seq[n]
                for k in range(0, self.m + 1):
                    if n - k < 0:
                        break
                    self.tables[k][tuple(seq[n - k:n])][nxt] += 1
        return self

    # --------------------------------------------------------------- lookup
    def lookup(self, context):
        """Return (Counter of next tokens, order actually used).

        Uses the longest suffix of `context` (up to m tokens) seen in training.
        Always succeeds once fitted, because order 0 is the unigram distribution.
        """
        context = tuple(int(a) for a in context)[-self.m:] if self.m > 0 else ()
        for k in range(min(self.m, len(context)), -1, -1):
            key = context[len(context) - k:] if k > 0 else ()
            c = self.tables[k].get(key)
            if c:
                return c, k
        raise RuntimeError("model has not been fitted")

    def prob(self, context, b):
        """Smoothed probability of next token b given the context (for scoring)."""
        c, _ = self.lookup(context)
        total = sum(c.values())
        p_mle = c.get(int(b), 0) / total
        lam = self.smoothing
        return (1 - lam) * p_mle + lam / self.vocab_size

    def top_k(self, context, k=3):
        c, _ = self.lookup(context)
        return [tok for tok, _ in c.most_common(k)]

    def sample(self, context, rng):
        """Draw the next token from the empirical distribution (used for generation)."""
        c, _ = self.lookup(context)
        toks = np.fromiter(c.keys(), dtype=int, count=len(c))
        cnt = np.fromiter(c.values(), dtype=float, count=len(c))
        return int(rng.choice(toks, p=cnt / cnt.sum()))

    # ----------------------------------------------------------- statistics
    @property
    def n_contexts(self):
        """Number of distinct length-m contexts seen in training."""
        return len(self.tables[self.m])


def evaluate(model, sequences, top_k=3):
    """Score a fitted model on held-out token sequences.

    For every position n >= m in every sequence, the context is the previous m tokens and
    the target is the next token. Returns a dict with:

      coverage      fraction of positions whose full length-m context was seen in training
      top1, topk    next-token accuracy (using back-off when the context is unseen)
      top1_seen     top-1 accuracy restricted to covered positions
      perplexity    exp(mean negative log-likelihood), with the model's smoothing
      n             number of scored positions
    """
    m = model.m
    n = hit1 = hitk = covered = hit1_seen = 0
    nll = 0.0
    for seq in sequences:
        seq = [int(a) for a in seq]
        for i in range(m, len(seq)):
            ctx, tgt = seq[i - m:i], seq[i]
            counts, used = model.lookup(ctx)
            best = model.top_k(ctx, top_k)
            n += 1
            seen = used == m
            covered += seen
            hit = best[0] == tgt
            hit1 += hit
            hit1_seen += hit and seen
            hitk += tgt in best
            nll -= np.log(model.prob(ctx, tgt))
    return dict(
        m=m, n=n,
        coverage=covered / n,
        top1=hit1 / n,
        top1_seen=hit1_seen / max(covered, 1),
        topk=hitk / n,
        perplexity=float(np.exp(nll / n)),
    )


def persistence_baseline(sequences):
    """Accuracy of always predicting 'same token as now' (no learning)."""
    hit = n = 0
    for seq in sequences:
        hit += int(np.sum(seq[1:] == seq[:-1]))
        n += len(seq) - 1
    return hit / n
