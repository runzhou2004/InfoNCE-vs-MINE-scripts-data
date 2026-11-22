"""
Data generator + InfoNCE and MINE estimators, with optional validation-based early stopping.

Run examples:
  python mi_start.py                                   # InfoNCE and MINE, d=5, MI=2, n=10000
  python mi_start.py --estimator mine                  # just MINE
  python mi_start.py --transform cubic                 # non-Gaussian marginals
  python mi_start.py --shuffle_control                 # true MI = 0 sanity check
  python mi_start.py --no_early_stop                   # old behavior: always use the last step
  python mi_start.py --d 20 --mi 6 --K 256             # a harder grid point

Early stopping (on by default): 80% of the n training rows fit the critic, the other 20% are a
validation set. Every `eval_every` steps the estimate on the validation set is computed, and the
best checkpoint is restored at the end. The held-out TEST set is only used for the final number.
"""
import argparse
import copy
import math

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# 1. Data generator with known ground-truth MI
# ----------------------------------------------------------------------------
def rho_for_mi(mi, d):
    """Per-coordinate correlation that gives total MI (nats) in d dimensions.
    MI = -(d/2) * ln(1 - rho^2)  =>  rho = sqrt(1 - exp(-2*MI/d))."""
    return math.sqrt(1.0 - math.exp(-2.0 * mi / d))


def sample_gaussian_pairs(n, d, rho):
    x = torch.randn(n, d)
    z = torch.randn(n, d)
    y = rho * x + math.sqrt(1.0 - rho ** 2) * z
    return x, y


def apply_transform(v, name):
    """Invertible elementwise maps: they leave MI unchanged."""
    if name == "none":
        return v
    if name == "cubic":
        return v ** 3
    if name == "asinh":
        return torch.sinh(v)  # sinh is the inverse of asinh; also invertible, heavy-tailed
    raise ValueError(name)


def make_dataset(n_train, n_test, d, mi, transform):
    rho = rho_for_mi(mi, d)
    x, y = sample_gaussian_pairs(n_train + n_test, d, rho)
    x, y = apply_transform(x, transform), apply_transform(y, transform)
    xtr, ytr, xte, yte = x[:n_train], y[:n_train], x[n_train:], y[n_train:]
    # standardize with TRAIN statistics only (heavy tails can wreck training otherwise)
    xm, xs = xtr.mean(0), xtr.std(0) + 1e-8
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-8
    xtr, xte = (xtr - xm) / xs, (xte - xm) / xs
    ytr, yte = (ytr - ym) / ys, (yte - ym) / ys
    return [t.to(DEVICE) for t in (xtr, ytr, xte, yte)]


# ----------------------------------------------------------------------------
# 2. Critic (shared by both estimators)
# ----------------------------------------------------------------------------
class Critic(nn.Module):
    """Small MLP on concatenated (x, y) -> scalar score."""

    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, xy):
        return self.net(xy).squeeze(-1)


def pair_scores(critic, x, y):
    """Scores for aligned pairs (x_i, y_i). Shape (B,)."""
    return critic(torch.cat([x, y], dim=-1))


class BestCheckpoint:
    """Keeps the critic weights with the best VALIDATION estimate seen so far.
    Step 0 (the untrained critic) counts, so the result is never worse than 'no learning'."""

    def __init__(self, critic, val_fn, every):
        self.critic, self.val_fn, self.every = critic, val_fn, every
        self.best_val, self.best_step, self.state = -float("inf"), 0, None
        self.check(0)

    def check(self, step):
        v = self.val_fn(self.critic)
        if v > self.best_val:
            self.best_val, self.best_step = v, step
            self.state = copy.deepcopy(self.critic.state_dict())

    def maybe(self, step):
        if step % self.every == 0:
            self.check(step)

    def restore(self):
        self.critic.load_state_dict(self.state)


# ----------------------------------------------------------------------------
# 3. InfoNCE
# ----------------------------------------------------------------------------
def score_matrix(critic, x, y):
    """scores[i, j] = f(x_i, y_j) for all pairs in the batch. Shape (K, K)."""
    K = x.shape[0]
    xx = x.unsqueeze(1).expand(K, K, -1)
    yy = y.unsqueeze(0).expand(K, K, -1)
    return critic(torch.cat([xx, yy], dim=-1))


def infonce_estimate(scores):
    """InfoNCE lower bound (nats). Row i: positive is y_i, negatives are the other K-1 y's.
    Value is capped at log K."""
    K = scores.shape[0]
    pos = scores.diag()
    return (pos - torch.logsumexp(scores, dim=1)).mean() + math.log(K)


def train_infonce(critic, xtr, ytr, K, steps, lr, verbose=True, ckpt=None):
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    n = xtr.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (K,), device=DEVICE)
        loss = -infonce_estimate(score_matrix(critic, xtr[idx], ytr[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (step + 1) % max(1, steps // 10) == 0:
            print(f"  step {step + 1:5d}  train InfoNCE = {-loss.item():.3f}")
        if ckpt is not None:
            ckpt.maybe(step + 1)


@torch.no_grad()
def evaluate_infonce(critic, xte, yte, K):
    """Average InfoNCE over held-out batches of size K."""
    perm = torch.randperm(xte.shape[0], device=DEVICE)
    xte, yte = xte[perm], yte[perm]
    vals = []
    for i in range(0, xte.shape[0] - K + 1, K):
        vals.append(infonce_estimate(score_matrix(critic, xte[i:i + K], yte[i:i + K])).item())
    return sum(vals) / len(vals)


# ----------------------------------------------------------------------------
# 4. MINE (Donsker-Varadhan bound with moving-average gradient correction)
# ----------------------------------------------------------------------------
def train_mine(critic, xtr, ytr, B, steps, lr, ema_alpha=0.01, verbose=True, ckpt=None):
    """DV bound:  E_joint[T] - log E_marginals[exp(T)].

    The minibatch gradient of the log term is biased. MINE fixes this by dividing by a
    moving average of E[exp(T)] instead of the current batch mean in the backward pass.
    Joint pairs are (x_i, y_i); product-of-marginals pairs use INDEPENDENT random rows
    for x and y.
    """
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    n = xtr.shape[0]
    log_ema = None  # moving average of E[exp(T)], kept in the log domain for stability
    for step in range(steps):
        idx = torch.randint(0, n, (B,), device=DEVICE)
        jdx = torch.randint(0, n, (B,), device=DEVICE)
        t_joint = pair_scores(critic, xtr[idx], ytr[idx])
        t_marg = pair_scores(critic, xtr[idx], ytr[jdx])

        log_mean_et = torch.logsumexp(t_marg, 0) - math.log(B)  # log of batch mean of exp(T)
        if log_ema is None:
            log_ema = log_mean_et.detach()
        else:
            log_ema = torch.logaddexp(math.log(1 - ema_alpha) + log_ema,
                                      math.log(ema_alpha) + log_mean_et.detach())

        # surrogate: its gradient is E[exp(T) grad T] / ema, i.e. the batch mean divided
        # by the moving average instead of the current batch's own mean
        loss = -(t_joint.mean() - torch.exp(log_mean_et - log_ema))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (step + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():  # the true (plain) DV value on this batch, for logging
                dv = t_joint.mean() - (torch.logsumexp(t_marg, 0) - math.log(B))
            print(f"  step {step + 1:5d}  train DV = {dv.item():.3f}")
        if ckpt is not None:
            ckpt.maybe(step + 1)


@torch.no_grad()
def evaluate_mine(critic, xte, yte, n_marg_factor=10, chunk=50000):
    """Plain DV on held-out data (NOT the moving-average version):
    mean T on joint pairs  -  log mean exp(T) on product-of-marginals pairs.
    The marginal term uses n_marg_factor * n_test random (x_i, y_j) pairs."""
    n = xte.shape[0]
    joint = pair_scores(critic, xte, yte).mean()
    m = n * n_marg_factor
    logsums = []
    for s in range(0, m, chunk):
        b = min(chunk, m - s)
        i = torch.randint(0, n, (b,), device=DEVICE)
        j = torch.randint(0, n, (b,), device=DEVICE)
        logsums.append(torch.logsumexp(pair_scores(critic, xte[i], yte[j]), 0))
    log_mean_exp = torch.logsumexp(torch.stack(logsums), 0) - math.log(m)
    return (joint - log_mean_exp).item()


# ----------------------------------------------------------------------------
# 5. One run = one (estimator, config, seed). The grid loop calls this.
# ----------------------------------------------------------------------------
def run_one(estimator, d, mi, n, n_test, K, transform, steps, lr, hidden, seed,
            shuffle_control=False, ema_alpha=0.01, early_stop=True, val_frac=0.2,
            eval_every=100, verbose=True):
    """Returns (test estimate, info). info has best_step and val_estimate (best_step = -1
    and val_estimate = nan when early stopping is off).

    With early_stop, n is the TOTAL training-set size: the first (1 - val_frac) of the rows
    fit the critic and the last val_frac rows choose the checkpoint."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = make_dataset(n, n_test, d, mi, transform)
    if shuffle_control:
        # break the pairing in train, validation and test so the estimator sees independent
        # data (true MI = 0); this measures its baseline bias
        ytr = ytr[torch.randperm(ytr.shape[0], device=DEVICE)]
        yte = yte[torch.randperm(yte.shape[0], device=DEVICE)]
    critic = Critic(d, hidden).to(DEVICE)

    ckpt = None
    if early_stop:
        n_fit = int(round(n * (1 - val_frac)))
        xva, yva = xtr[n_fit:], ytr[n_fit:]
        xtr, ytr = xtr[:n_fit], ytr[:n_fit]
        if estimator == "infonce":
            kv = min(K, xva.shape[0])  # validation set can be smaller than K
            val_fn = lambda c: evaluate_infonce(c, xva, yva, kv)
        else:
            val_fn = lambda c: evaluate_mine(c, xva, yva, n_marg_factor=50)
        ckpt = BestCheckpoint(critic, val_fn, eval_every)

    if estimator == "infonce":
        train_infonce(critic, xtr, ytr, K, steps, lr, verbose, ckpt)
        if ckpt is not None:
            ckpt.restore()
        est = evaluate_infonce(critic, xte, yte, K)
    elif estimator == "mine":
        train_mine(critic, xtr, ytr, K, steps, lr, ema_alpha, verbose, ckpt)
        if ckpt is not None:
            ckpt.restore()
        est = evaluate_mine(critic, xte, yte)
    else:
        raise ValueError(estimator)

    info = dict(best_step=ckpt.best_step if ckpt else -1,
                val_estimate=ckpt.best_val if ckpt else float("nan"))
    return est, info


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--estimator", default="both", choices=["infonce", "mine", "both"])
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--mi", type=float, default=2.0, help="true MI in nats")
    p.add_argument("--n", type=int, default=10000, help="training samples (incl. validation)")
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--K", type=int, default=128,
                   help="batch size (for InfoNCE this also sets the log K ceiling)")
    p.add_argument("--transform", default="none", choices=["none", "cubic", "asinh"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--ema_alpha", type=float, default=0.01, help="MINE moving-average rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_control", action="store_true")
    p.add_argument("--no_early_stop", dest="early_stop", action="store_false",
                   help="always use the last step (old behavior)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--eval_every", type=int, default=100)
    args = p.parse_args()

    # sanity check for the generator: empirical correlation should match rho
    torch.manual_seed(args.seed)
    rho = rho_for_mi(args.mi, args.d)
    xc, yc = sample_gaussian_pairs(50000, args.d, rho)
    emp = ((xc - xc.mean(0)) * (yc - yc.mean(0))).mean(0) / (xc.std(0) * yc.std(0))
    print(f"target rho = {rho:.4f}, empirical rho (mean over dims) = {emp.mean().item():.4f}")

    true_mi = 0.0 if args.shuffle_control else args.mi
    names = ["infonce", "mine"] if args.estimator == "both" else [args.estimator]
    results = {}
    for name in names:
        print(f"\n--- {name} ---")
        results[name] = run_one(name, args.d, args.mi, args.n, args.n_test, args.K,
                                args.transform, args.steps, args.lr, args.hidden, args.seed,
                                args.shuffle_control, args.ema_alpha, args.early_stop,
                                args.val_frac, args.eval_every)

    print(f"\ntrue MI = {true_mi:.3f} nats | log K = {math.log(args.K):.3f} | "
          f"early stopping = {args.early_stop}")
    for name, (est, info) in results.items():
        extra = f" (best checkpoint at step {info['best_step']})" if args.early_stop else ""
        print(f"{name:8s} held-out estimate = {est:.3f}{extra}")


if __name__ == "__main__":
    main()
