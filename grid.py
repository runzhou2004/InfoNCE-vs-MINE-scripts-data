"""
Grid runner: runs every combination of settings and appends one row per run to a CSV.

Needs mi_start.py in the same folder.

Early stopping is ON by default: 20% of each training set (size n) is held back as a validation
set to pick the best checkpoint, and the separate test set is used only for the final number.
Use --no_early_stop to reproduce the earlier "always use the last step" behavior.

Safe to resume: if you run the same command again, rows already in the CSV are skipped,
so a crash or Colab disconnect doesn't lose finished work. If you change a setting such as
--steps, use a NEW --out file. (An old CSV from before early stopping existed has different
columns and will be refused, so you can't mix them up by accident.)

Examples:
  python grid.py --dry_run --out new_results.csv        # just count the runs
  python grid.py --out new_results.csv                  # the full grid (MI 2,4,6; cubic; 5 seeds)
  python grid.py --mi 2 6 --out new_results.csv         # drop MI=4
  python grid.py --transform cubic asinh                # add the asinh transform
  python grid.py --seeds 1 --steps 300 --n_test 2000 --out tiny.csv   # quick test of the loop
"""
import argparse
import csv
import itertools
import math
import os
import time

from mi_start import DEVICE, run_one

FIELDS = ["estimator", "transform", "d", "true_mi", "n", "K", "seed", "steps", "control",
          "early_stop", "log_K", "estimate", "best_step", "val_estimate", "seconds"]


def build_runs(a):
    """One dict per run. Seeds are the OUTER loop, so if you stop early you still have
    every configuration with the same number of seeds."""
    runs = []
    for seed in range(a.seeds):
        for tf in a.transform:
            for d, mi, n, K, est in itertools.product(a.d, a.mi, a.n, a.K, a.estimators):
                runs.append(dict(estimator=est, transform=tf, d=d, true_mi=mi, n=n, K=K,
                                 seed=seed, control=0))
            if a.controls:  # shuffled pairs: true MI = 0
                for d, n, K, est in itertools.product(a.d, a.n, a.K, a.estimators):
                    runs.append(dict(estimator=est, transform=tf, d=d, true_mi=0.0, n=n, K=K,
                                     seed=seed, control=1))
    return runs


def key(r, steps, early_stop):
    return (str(r["estimator"]), str(r["transform"]), int(r["d"]), float(r["true_mi"]),
            int(r["n"]), int(r["K"]), int(r["seed"]), int(steps), int(r["control"]),
            int(early_stop))


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add(key(row, row["steps"], row["early_stop"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results.csv")
    p.add_argument("--estimators", nargs="+", default=["infonce", "mine"])
    p.add_argument("--transform", nargs="+", default=["cubic"], choices=["none", "cubic", "asinh"])
    p.add_argument("--d", nargs="+", type=int, default=[5, 20])
    p.add_argument("--mi", nargs="+", type=float, default=[2.0, 4.0, 6.0], help="true MI in nats")
    p.add_argument("--n", nargs="+", type=int, default=[1000, 10000],
                   help="training sizes (with early stopping, 20%% of these are validation)")
    p.add_argument("--K", nargs="+", type=int, default=[32, 256], help="batch size")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n_test", type=int, default=10000, help="held-out size (same for every run)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--ema_alpha", type=float, default=0.01)
    p.add_argument("--no_early_stop", dest="early_stop", action="store_false")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--no_controls", dest="controls", action="store_false",
                   help="skip the shuffled-pairs control runs")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    # refuse to append to a CSV with different columns (e.g. one from the old version)
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, newline="") as f:
            header = next(csv.reader(f))
        if header != FIELDS:
            print(f"ERROR: {a.out} has different columns than this version of grid.py writes "
                  f"(probably an older results file). Use a new --out file name.")
            return

    runs = build_runs(a)
    done = load_done(a.out)
    todo = [r for r in runs if key(r, a.steps, a.early_stop) not in done]
    print(f"device = {DEVICE} | early stopping = {a.early_stop} | total runs = {len(runs)} | "
          f"already done = {len(runs) - len(todo)} | to do = {len(todo)}")
    if a.dry_run or not todo:
        return

    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    t_start = time.time()
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            f.flush()
        for i, r in enumerate(todo, 1):
            t0 = time.time()
            info = dict(best_step=-1, val_estimate=float("nan"))
            try:
                est, info = run_one(
                    r["estimator"], r["d"],
                    a.mi[0] if r["control"] else r["true_mi"],  # control data gets shuffled anyway
                    r["n"], a.n_test, r["K"], r["transform"], a.steps, a.lr, a.hidden,
                    r["seed"], shuffle_control=bool(r["control"]), ema_alpha=a.ema_alpha,
                    early_stop=a.early_stop, val_frac=a.val_frac, eval_every=a.eval_every,
                    verbose=False)
            except Exception as e:  # don't let one failed run kill the whole grid
                print(f"  ERROR in run {r}: {e}")
                est = float("nan")
            if not math.isfinite(est):
                print(f"  WARNING: non-finite estimate for {r}")
            sec = time.time() - t0
            w.writerow(dict(r, steps=a.steps, early_stop=int(a.early_stop),
                            log_K=round(math.log(r["K"]), 4), estimate=est,
                            best_step=info["best_step"], val_estimate=info["val_estimate"],
                            seconds=round(sec, 1)))
            f.flush()  # write to disk now so nothing is lost if the session dies
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {r['estimator']:7s} {r['transform']} d={r['d']} "
                  f"MI={r['true_mi']} n={r['n']} K={r['K']} seed={r['seed']} "
                  f"ctrl={r['control']} -> {est:.3f} (best step {info['best_step']}, "
                  f"{sec:.0f}s, ~{eta / 60:.0f} min left)")


if __name__ == "__main__":
    main()
