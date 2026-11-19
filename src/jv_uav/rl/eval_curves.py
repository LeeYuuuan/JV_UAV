"""Overwrite a compact evaluation history chart after each check."""
import json
from pathlib import Path


def save_evaluation_curves(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    rows = [json.loads(line) for line in (output / "eval.jsonl").read_text().splitlines() if line.strip()]
    # The first evaluation seed is shared by quick and full checks.
    seed = rows[0]["seed"]
    rows = [row for row in rows if row["seed"] == seed]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fields = [("lower_return", "Lower episode return"), ("mean_max_backlog", "Mean max backlog / packets"),
              ("frames", "Survived upper frames"), ("sensor_coverage_fraction", "Fraction of sensors visited")]
    for ax, (field, label) in zip(axes.flat, fields):
        ax.plot([r["upper_steps"] for r in rows], [r[field] for r in rows])
        ax.set(xlabel="Training upper frames", ylabel=label)
        ax.grid(alpha=.2)
    fig.suptitle(f"Deterministic joint evaluation | seed {seed}")
    fig.savefig(output / "evaluation_curves.png", dpi=140)
    plt.close(fig)

    # Keep policy and value losses in separate panels because their scales differ.
    history = output / "train.jsonl"
    if history.exists():
        records = []
        for line in history.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        panels = [(["mappo_policy_loss", "mappo_weighted_entropy"], "Upper policy and weighted entropy"),
                  (["mappo_value_loss"], "Upper value loss"),
                  (["sac_actor_loss", "sac_entropy_loss", "sac_policy_q_mean"], "Lower policy terms (Q is before minus sign)"),
                  (["sac_critic_loss"], "Lower critic loss")]
        for ax, (keys, title) in zip(axes.flat, panels):
            for key in keys:
                data = [r for r in records if key in r and (not key.startswith("mappo") or r["mappo_updated"])]
                if data:
                    ax.plot([r["upper_steps"] for r in data], [r[key] for r in data], label=key)
            ax.set(title=title, xlabel="Training upper frames")
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.savefig(output / "loss_curves.png", dpi=140)
        plt.close(fig)
