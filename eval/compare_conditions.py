"""Run and plot the A/B/C comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from eval.metrics import summarize
from training.grpo_train import run_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=9)
    parser.add_argument("--output-dir", default="outputs/eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = {}
    for condition in ("A", "B", "C"):
        records = run_condition(condition, args.episodes, "mixed")
        conditions[condition] = {"records": records, "summary": summarize(records)}

    means = [conditions[key]["summary"]["mean_total_reward"] for key in ("A", "B", "C")]
    labels = ["Random", "Single-Agent", "SPECTRA"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, color=["#b0bec5", "#ffb74d", "#4db6ac"])
    ax.set_ylabel("Mean total reward")
    ax.set_title("SPECTRA comparison: A vs B vs C")
    fig.tight_layout()
    chart_path = output_dir / "compare_conditions.png"
    fig.savefig(chart_path)
    plt.close(fig)

    json_path = output_dir / "compare_conditions.json"
    json_path.write_text(json.dumps(conditions, indent=2), encoding="utf-8")
    print(json.dumps({"chart": str(chart_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
