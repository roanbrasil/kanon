"""Command line: lint a policy, solve a case, run the bench.

    kanon lint examples/schema-change/policy.yaml
    kanon solve examples/schema-change/policy.yaml examples/schema-change/cases/add-optional-field.json
    kanon solve ... --llm ollama:qwen2.5:14b
    kanon bench examples/schema-change --error 0.1 --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import bench as benchmod
from .lint import lint
from .oracle import Oracle, Verdict, anthropic_oracle, cli_oracle, ollama_oracle, route, table_oracle
from .policy import load_policy
from .solver import Coverage, solve
from .trace import explain, to_json


def _load_case(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def _llm_from_spec(spec: str | None) -> Oracle | None:
    if not spec:
        return None
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return ollama_oracle(model or "qwen2.5:14b")
    if kind == "anthropic":
        return anthropic_oracle(model or "claude-sonnet-4-5")
    raise SystemExit(f"unknown llm spec {spec!r}; use ollama:<model> or anthropic:<model>")


def load_cases(example_dir: Path) -> list[benchmod.Case]:
    cases = []
    for path in sorted((example_dir / "cases").glob("*.*")):
        if path.suffix not in {".json", ".yaml", ".yml"}:
            continue
        data = _load_case(path)
        cases.append(
            benchmod.Case(
                id=path.stem,
                data=data["case"],
                expected=data["expected"],
                truth=data.get("truth", {}),
                note=data.get("note"),
            )
        )
    return cases


def cmd_lint(args: argparse.Namespace) -> int:
    policy = load_policy(Path(args.policy))
    report = lint(policy)
    print(report)
    return 1 if report.errors else 0


def cmd_solve(args: argparse.Namespace) -> int:
    policy = load_policy(Path(args.policy))
    doc = _load_case(Path(args.case))
    case = doc.get("case", doc)
    llm = _llm_from_spec(args.llm)
    fixture = None
    if "truth" in doc:
        # offline: answer judgment facts from the fixture's ground truth
        fixture = table_oracle({k: (Verdict.unknown("fixture: insufficient evidence", source="fixture") if v is None else Verdict.from_bool(v, "fixture ground truth", source="fixture")) for k, v in doc["truth"].items()})
    llm = llm or fixture
    human = cli_oracle if args.interactive else fixture
    oracle = route(policy, llm=llm, human=human)
    out = solve(policy, case, oracle, coverage=Coverage.FULL if args.full else Coverage.MINIMAL)
    if args.json:
        print(to_json(out))
    else:
        print(explain(out))
        if "expected" in doc:
            ok = out.decision == doc["expected"]
            print(f"\nexpected {doc['expected']!r}: {'OK' if ok else 'MISMATCH'}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    example_dir = Path(args.example)
    policy = load_policy(example_dir / "policy.yaml")
    cases = load_cases(example_dir)
    ks = [int(k) for k in args.ks.split(",")]
    rows = []
    thresholds = [float(t) for t in args.abstain.split(",")] if args.abstain else [policy.abstain_below]
    for thr in thresholds:
        pol = policy.model_copy(update={"abstain_below": thr})
        for err in [float(e) for e in args.error.split(",")]:
            res = benchmod.run(pol, cases, benchmod.simulated_llm(pol, err), n=args.n, ks=ks, seed=args.seed, label=f"ε={err:.2f} abstain<{thr:.2f}")
            print(res)
            rows.append({"policy": pol.policy, "error_rate": err, "abstain_below": thr, **res.table})
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kanon")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("lint", help="static checks on a policy")
    pl.add_argument("policy")
    pl.set_defaults(fn=cmd_lint)

    ps = sub.add_parser("solve", help="solve one case")
    ps.add_argument("policy")
    ps.add_argument("case")
    ps.add_argument("--llm", help="ollama:<model> or anthropic:<model>")
    ps.add_argument("--interactive", action="store_true", help="ask `by: human` facts at the terminal")
    ps.add_argument("--full", action="store_true", help="answer every fact, not just the needed ones")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(fn=cmd_solve)

    pb = sub.add_parser("bench", help="pass^k with a simulated noisy answerer")
    pb.add_argument("example", help="directory with policy.yaml and cases/")
    pb.add_argument("--error", default="0.0,0.05,0.1,0.2", help="comma-separated per-fact error rates")
    pb.add_argument("--n", type=int, default=20)
    pb.add_argument("--ks", default="1,5,10,20")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--abstain", help="comma-separated abstain_below thresholds to sweep (default: the policy's)")
    pb.add_argument("--out")
    pb.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
