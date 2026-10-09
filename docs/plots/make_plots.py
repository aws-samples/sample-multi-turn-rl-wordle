#!/usr/bin/env python3
"""Regenerate docs/plots/*.png from the MLflow history of the training job and
the SageMaker evaluation jobs.

    AWS_PROFILE=<profile> MLFLOW_APP_ARN=<your MLflow app ARN> \
        uv run python docs/plots/make_plots.py

Edit the constants below to point at your own runs.
"""
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

os.environ.setdefault("AWS_REGION", "us-east-1")
MLFLOW_APP = os.environ["MLFLOW_APP_ARN"]
TRAIN_RUN_ID = "2b96801b905144b69d267a070103c3e8"   # 100-step training job
EVAL_EXPERIMENT = "wordle-mtrl-eval"
N_HELD_OUT = 128

# (label, base eval run names, trained eval run names). None = use the training
# job's own validation at step 0 (base) and step 100 (trained).
SETTINGS = [
    ("Low effort\ntemp 0", None, None),
    ("Medium effort\ntemp 0", ["wordle-eval-base-t0p0-medium-20261009115036"],
     ["wordle-eval-trained-t0p0-medium-20261009115639"]),
    ("Low effort\ntemp 1.0", ["wordle-eval-base-t1p0-low-20261009111415"],
     ["wordle-eval-trained-t1p0-low-20261009111817"]),
    ("Medium effort\ntemp 1.0", ["wordle-eval-base-t1p0-medium-20261009120643"],
     ["wordle-eval-trained-t1p0-medium-20261009121246",
      "wordle-eval-trained-t1p0-medium-20261009121949"]),
]

OUT = Path(__file__).resolve().parent
ORANGE, BLUE, GRAY = "#E67E22", "#2171B5", "#B8C4CC"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def half_up(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def history(c, run_id, key):
    return {int(m.step): m.value for m in c.get_metric_history(run_id, key)}


def eval_metrics(c, names):
    exp = c.get_experiment_by_name(EVAL_EXPERIMENT)
    vals = []
    for n in names:
        runs = c.search_runs([exp.experiment_id], filter_string=f"attributes.run_name = '{n}'")
        assert len(runs) == 1, n
        m = runs[0].data.metrics
        vals.append((m["eval/reward/pass_at_1"], m["eval/reward/mean"]))
    return (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))


def training_curves(c):
    train = history(c, TRAIN_RUN_ID, "rollout/reward/mean")
    val_r = history(c, TRAIN_RUN_ID, "val/reward/mean")
    val_p = history(c, TRAIN_RUN_ID, "val/reward/pass_at_1")
    steps = sorted(train)
    smooth = [sum(train[t] for t in steps if s - 9 <= t <= s) /
              len([t for t in steps if s - 9 <= t <= s]) for s in steps]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=200)
    a1.plot(steps, [train[s] for s in steps], color=ORANGE, alpha=0.25, lw=1)
    a1.plot(steps, smooth, color=ORANGE, lw=2.5, label="Training rollouts (temp 1.0), 10-step average")
    vs = sorted(val_r)
    a1.plot(vs, [val_r[s] for s in vs], "-o", color=BLUE, lw=2.5, ms=6,
            label="Validation, held-out words (temp 0)")
    a1.axhline(0, color="#94A3B8", lw=1, ls=":")
    a1.set(title="Mean reward per game", xlabel="Training step", ylabel="Mean reward")
    a1.title.set_fontweight("bold")
    a1.legend(frameon=False, loc="lower right", fontsize=11)

    ps = sorted(val_p)
    pct = [100 * val_p[s] for s in ps]
    a2.plot(ps, pct, "-o", color=BLUE, lw=2.5, ms=6)
    for k, (s, p) in enumerate(zip(ps, pct)):
        prev_p = pct[k - 1] if k > 0 else None
        next_p = pct[k + 1] if k + 1 < len(pct) else None
        dip = prev_p is not None and p < prev_p and (next_p is None or p < next_p)
        a2.annotate(f"{half_up(p)}%", (s, p), textcoords="offset points",
                    xytext=(0, -17 if dip else 9), ha="center", fontsize=10, color="#333333")
    a2.set(title="Validation solve rate, held-out words (temp 0)", xlabel="Training step",
           ylabel="Words solved (%)", ylim=(0, 60))
    a2.title.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")
    print("wrote training_curves.png")


def eval_bars(c):
    val_p = history(c, TRAIN_RUN_ID, "val/reward/pass_at_1")
    val_r = history(c, TRAIN_RUN_ID, "val/reward/mean")
    last = max(val_p)
    rows = []
    for label, base, trained in SETTINGS:
        if base is None:
            b, t = (val_p[0], val_r[0]), (val_p[last], val_r[last])
        else:
            b, t = eval_metrics(c, base), eval_metrics(c, trained)
        rows.append((label, b, t))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=200)
    x = range(len(rows))
    for i, (label, (bp, br), (tp, tr)) in enumerate(rows):
        bp_i, tp_i = half_up(100 * bp), half_up(100 * tp)
        a1.bar(i, bp_i, 0.6, color=GRAY, edgecolor="white")
        a1.bar(i, tp_i - bp_i, 0.6, bottom=bp_i, color=BLUE, edgecolor="white")
        if bp_i >= 6:
            a1.text(i, bp_i / 2, f"{bp_i}%", ha="center", va="center", fontsize=11, color="#333333")
        else:   # too thin to hold a label: put it just above, inside the gain segment
            a1.text(i, bp_i + 1.5, f"base {bp_i}%", ha="center", va="bottom", fontsize=10, color="white")
        a1.text(i, bp_i + (tp_i - bp_i) / 2, f"+{tp_i - bp_i}", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        a1.text(i, tp_i + 2, f"{tp_i}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

        floor = -0.5
        br2, tr2 = round(br, 2), round(tr, 2)
        a2.bar(i, br2 - floor, 0.6, bottom=floor, color=GRAY, edgecolor="white")
        a2.bar(i, tr2 - br2, 0.6, bottom=br2, color=BLUE, edgecolor="white")
        a2.text(i, (floor + br2) / 2, f"{br2:+.2f}", ha="center", va="center", fontsize=11, color="#333333")
        a2.text(i, br2 + (tr2 - br2) / 2, f"+{tr2 - br2:.2f}", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        a2.text(i, tr2 + 0.03, f"{tr2:+.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    labels = [r[0] for r in rows]
    a1.set(xticks=list(x), xticklabels=labels, ylabel="Words solved (%)", ylim=(0, 100))
    a1.set_title(f"Solve rate, {N_HELD_OUT} held-out words", fontweight="bold")
    a2.set(xticks=list(x), xticklabels=labels, ylabel="Mean reward", ylim=(-0.5, 1.15))
    a2.set_title("Mean reward per game", fontweight="bold")
    a2.axhline(0, color="#555555", lw=1, ls=":")
    a2.text(0.02, 0.97, "Axis starts at \u22120.5, the outcome reward\nfor a lost game",
            transform=a2.transAxes, va="top", fontsize=10, style="italic", color="#666666")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["Base model", "Gain from MTRL training (bar top = trained model)"],
               loc="upper center", ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "eval_trained_vs_base.png")
    print("wrote eval_trained_vs_base.png")
    for label, b, t in rows:
        print(f"  {label.replace(chr(10), ' '):24s} base {100*b[0]:5.1f}% {b[1]:+.3f} | trained {100*t[0]:5.1f}% {t[1]:+.3f}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_APP)
    client = MlflowClient()
    training_curves(client)
    eval_bars(client)
