"""pass^k with a real model: policy graph vs. one-shot "here is the policy, decide".

    # local model through Ollama (DGX Spark / Jetson / laptop)
    python scripts/live_bench.py examples/schema-change --llm ollama:qwen2.5:14b --n 10

    # Anthropic API
    ANTHROPIC_API_KEY=... python scripts/live_bench.py examples/credit --llm anthropic:claude-sonnet-4-5 --n 10

Two arms, same model, same cases, n trials each:

* ``kanon``: the policy decides; the model only answers the judgment facts
  the solver asks for, one at a time, without seeing the policy.
* ``one-shot``: the model receives the whole policy (YAML) and the case and
  must return the decision id directly.

Reports pass^k, pass@k, escalation and wrong rates for both arms and writes
``live_results.json`` next to the example. Costs n × (cases × questions)
model calls for the kanon arm and n × cases for the one-shot arm.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import kanon
from kanon import bench
from kanon.cli import _llm_from_spec, load_cases
from kanon.oracle import ollama_oracle

ONE_SHOT = """You are applying the policy below to the case below. Return ONLY a JSON
object {{"decision": "<decision id>"}} where the id is one of the policy's
decision ids, or "escalate" if the policy cannot be applied.

POLICY (YAML):
{policy}

CASE (JSON):
{case}
"""


def one_shot_decider(spec: str, policy_text: str):
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        import urllib.request

        base = "http://localhost:11434"

        def ask(prompt: str) -> str:
            body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0}}).encode()
            req = urllib.request.Request(f"{base}/api/generate", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read()).get("response", "")

    elif kind == "anthropic":
        import anthropic

        client = anthropic.Anthropic()

        def ask(prompt: str) -> str:
            msg = client.messages.create(model=model, max_tokens=200, temperature=0, messages=[{"role": "user", "content": prompt}])
            return "".join(getattr(b, "text", "") for b in msg.content)

    else:
        raise SystemExit("use --llm ollama:<model> or anthropic:<model>")

    def decide(case: dict) -> str:
        text = ask(ONE_SHOT.format(policy=policy_text, case=json.dumps(case, ensure_ascii=False, indent=2)))
        start, end = text.find("{"), text.rfind("}")
        try:
            return str(json.loads(text[start : end + 1]).get("decision", "escalate"))
        except Exception:  # noqa: BLE001
            return "unparseable"

    return decide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("example")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ks", default="1,3,5")
    args = ap.parse_args()

    ex = Path(args.example)
    policy = kanon.load_policy(ex / "policy.yaml")
    policy_text = (ex / "policy.yaml").read_text()
    cases = load_cases(ex)
    ks = [int(k) for k in args.ks.split(",") if int(k) <= args.n]
    llm = _llm_from_spec(args.llm)
    if llm is None:
        sys.exit("no model")

    # kanon arm: humans facts are answered by the same model in this bench
    def make(case, rng):
        return kanon.route(policy, llm=llm, human=llm)

    print(f"kanon arm: {len(cases)} cases × {args.n} trials")
    kanon_res = bench.run(policy, cases, make, n=args.n, ks=ks, label=f"kanon / {args.llm}")
    print(kanon_res)

    print(f"one-shot arm: {len(cases)} cases × {args.n} trials")
    decide = one_shot_decider(args.llm, policy_text)
    trials = []
    for case in cases:
        decisions = [decide(dict(case.data)) for _ in range(args.n)]
        trials.append(bench.TrialStats(case.id, case.expected, decisions))
    one_shot = bench.BenchResult(f"one-shot / {args.llm}", trials, ks)
    print(one_shot)

    out = {
        "model": args.llm,
        "n": args.n,
        "kanon": {"table": kanon_res.table, "per_case": [{"id": t.case_id, "decisions": t.decisions} for t in kanon_res.trials]},
        "one_shot": {"table": one_shot.table, "per_case": [{"id": t.case_id, "decisions": t.decisions} for t in one_shot.trials]},
    }
    (ex / "live_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {ex / 'live_results.json'}")


if __name__ == "__main__":
    main()
