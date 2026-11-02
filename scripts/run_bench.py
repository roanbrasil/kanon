"""Run the simulated pass^k bench on every example and draw the figures.

    python scripts/run_bench.py --n 20 --out bench/

Two sweeps per domain:

1. per-fact error rate ε of the simulated answerer, no confidence gating
   (abstain_below = 0): how silent errors propagate to decisions;
2. the same ε with the policy's own abstain_below: how gating converts
   silent errors into escalations.

The answerer only touches facts marked ``by: llm`` or ``by: human``. Code
facts always run their real checks, so a case whose decision is reached
through code alone is correct in every trial. The fraction of such cases is
where pass^k flattens out as k grows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import kanon  # noqa: E402
from kanon import bench  # noqa: E402
from kanon.cli import load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["schema-change", "predictive-maintenance", "release-gate", "credit"]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def code_only_fraction(policy, cases) -> float:
    """Cases decided without asking any llm/human fact (perfect answerer)."""
    n = 0
    for c in cases:
        oracle = kanon.route(
            policy,
            llm=kanon.table_oracle({k: kanon.Verdict.from_bool(v) for k, v in c.truth.items()}),
            human=kanon.table_oracle({k: kanon.Verdict.from_bool(v) for k, v in c.truth.items()}),
        )
        out = kanon.solve(policy, c.data, oracle)
        if all(policy.facts[f].by == "code" for f in out.execution.verdicts):
            n += 1
    return n / len(cases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--errors", default="0,0.05,0.1,0.15,0.2,0.3")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)
    errors = [float(e) for e in args.errors.split(",")]
    ks = [1, 5, 10, 20]
    rows = []
    summary = {}

    for domain in DOMAINS:
        ex = ROOT / "examples" / domain
        policy = kanon.load_policy(ex / "policy.yaml")
        cases = load_cases(ex)
        summary[domain] = {
            "facts": len(policy.facts),
            "code_facts": sum(1 for f in policy.facts.values() if f.by == "code"),
            "judgment_facts": sum(1 for f in policy.facts.values() if f.by != "code"),
            "decisions": len(policy.decisions),
            "cases": len(cases),
            "code_only_cases": code_only_fraction(policy, cases),
            "abstain_below": policy.abstain_below,
        }
        for thr in (0.0, policy.abstain_below):
            pol = policy.model_copy(update={"abstain_below": thr})
            for err in errors:
                res = bench.run(pol, cases, bench.simulated_llm(pol, err), n=args.n, ks=ks, seed=args.seed)
                rows.append({"domain": domain, "error_rate": err, "abstain_below": thr, **res.table})
                print(f"{domain:<24} {res}")

    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    # Figure 1: pass^k vs k for ε = 0.1, no gating, one line per domain
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    for domain, color in zip(DOMAINS, colors):
        r = next(x for x in rows if x["domain"] == domain and x["error_rate"] == 0.1 and x["abstain_below"] == 0.0)
        ys = [r[f"pass^{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", color=color, linewidth=2, markersize=6, label=domain)
        ax.annotate(domain, (ks[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), fontsize=9, color=TEXT, va="center")
    ax.set_xlabel("k (independent trials that must all reach the expected decision)")
    ax.set_ylabel("pass^k")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ks)
    ax.set_title("pass^k with a simulated answerer wrong 10% of the time on judgment facts, no gating", fontsize=10, color=TEXT)
    ax.grid(axis="y", color="#e6e6e3")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_passk_by_domain.png", dpi=160)

    # Figure 2: wrong vs escalation as a function of ε, gated vs not (schema-change)
    for domain in DOMAINS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for thr, color, name in ((0.0, ORANGE, "no gating"), (summary[domain]["abstain_below"], BLUE, f"abstain below {summary[domain]['abstain_below']:.2f}")):
            rs = [x for x in rows if x["domain"] == domain and x["abstain_below"] == thr]
            ax.plot([x["error_rate"] for x in rs], [x["wrong_rate"] for x in rs], marker="o", color=color, linewidth=2, label=f"wrong decision — {name}")
            ax.plot([x["error_rate"] for x in rs], [x["escalation_rate"] for x in rs], marker="s", color=color, linewidth=2, linestyle="--", label=f"escalated — {name}")
        ax.set_xlabel("per-fact error rate of the simulated answerer (ε)")
        ax.set_ylabel("share of trials")
        ax.set_ylim(0, 0.5)
        ax.set_title(f"{domain}: silent wrong decisions vs escalations", fontsize=10, color=TEXT)
        ax.grid(axis="y", color="#e6e6e3")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_gating_{domain}.png", dpi=160)
        plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
