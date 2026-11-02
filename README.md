# kanon

**Policy-adherent decisions for AI agents.** The policy is written once, as data,
and executed by a deterministic solver with three-valued logic. A model, a piece
of code or a person answers one atomic question at a time, never the policy.
Every decision comes with the computation that produced it.

```
policy.yaml ──► solver ──► "which fact do I need next?" ──► oracle (code | llm | human)
                  │                                             │
                  └──── verdict: yes / no / unknown + confidence + evidence
                  ▼
      decision + reasoning trace   (or an explicit escalation)
```

Why another rules engine? Because agents are being asked to apply policies —
contract rules, maintenance SOPs, release processes, credit criteria — and
"put the policy in the prompt" is right most of the time and wrong in ways you
cannot see. For that class of work the metric is **pass^k** (every one of k runs
correct), not pass@k (one of k). `kanon` is a small experiment in getting there
by construction:

* **Facts declare who answers them.** `by: code` facts run a sandboxed check
  against the case and never reach a model. `by: llm` facts carry `guidance`
  that records how ambiguity is resolved. `by: human` facts stop the solve
  until someone answers.
* **Unknown is a real answer.** Kleene logic propagates it; an answer below the
  fact's `abstain_below` confidence is treated as unknown; if the policy cannot
  be settled the outcome is an explicit `escalate` with the list of unresolved facts.
* **The trace is the computation.** Not a post-hoc explanation: the rounds,
  the verdicts with source / confidence / evidence, how each rule and decision
  evaluated, and why the winner won.
* **Policies are linted before use.** Unreachable decisions, overlaps in
  exclusive policies, gaps where no decision fires, shadowed decisions, unused
  facts — found by enumeration, with a counter-example.
* **Reliability is measured, not asserted.** `kanon bench` reports pass^k /
  pass@k / escalation / wrong-decision rates under a simulated imperfect
  answerer; `scripts/live_bench.py` runs the same against a real model, with a
  one-shot "here is the policy" baseline.

## Install

```bash
pip install -e .            # core: pydantic + pyyaml
pip install -e ".[llm]"     # + anthropic client (Ollama needs nothing extra)
pip install -e ".[dev]"     # + pytest, matplotlib
```

Python 3.11+.

## Thirty seconds

```yaml
# loan.yaml
policy: loan
abstain_below: 0.7
facts:
  good_credit:      {question: "Is the credit score 700 or above?", by: code, check: "case.score >= 700"}
  has_down_payment: {question: "Is the down payment at least 20%?",  by: code, check: "case.down >= 0.2"}
  stable_income:
    question: Has the applicant had stable employment for two years?
    by: llm
    guidance: Two employers in two years still counts if there was no gap longer than a month.
rules:
  core: good_credit and stable_income
decisions:
  - {id: approve, when: core and has_down_payment}
  - {id: review,  when: core and not has_down_payment}
  - {id: deny,    when: not core}
```

```python
import kanon

policy = kanon.load_policy("loan.yaml")
oracle = kanon.route(policy, llm=kanon.ollama_oracle("qwen2.5:14b"))
out = kanon.solve(policy, {"score": 731, "down": 0.1, "employment": "..."}, oracle)
print(kanon.explain(out))
```

```
Policy loan v1  ->  review [review]
  because: first matching decision: review

Round 1
  YES      Is the credit score 700 or above?
           by code, confidence 1.00
           evidence: case.score >= 700 -> True
Round 2
  YES      Has the applicant had stable employment for two years?
           by ollama:qwen2.5:14b, confidence 0.86
           evidence: employment history lists Acme 2023-2025 with no gap
Round 3
  NO       Is the down payment at least 20%?
           by code, confidence 1.00
           evidence: case.down >= 0.2 -> False
...
```

Command line:

```bash
kanon lint  examples/schema-change/policy.yaml
kanon solve examples/schema-change/policy.yaml examples/schema-change/cases/retype-new-version-no-ack.json
kanon solve examples/credit/policy.yaml examples/credit/cases/prime-approve.json --llm ollama:qwen2.5:14b
kanon bench examples/release-gate --error 0,0.1,0.2 --abstain 0,0.7 --n 20
```

## The policy language

| Key | Meaning |
|---|---|
| `facts.<id>.question` | the yes/no question, as the answerer sees it |
| `facts.<id>.by` | `code`, `llm` or `human` |
| `facts.<id>.check` | for `code`: a sandboxed expression over `case` (attribute or key access, comparisons, `len/any/all/sum/min/max`, comprehensions, `.get/.lower/.startswith/...`) |
| `facts.<id>.guidance` | how to interpret the question; the place where the organisation's reading of an ambiguous requirement is written down |
| `facts.<id>.abstain_below` | per-fact confidence threshold (defaults to the policy's) |
| `facts.<id>.source` | where this requirement comes from (section, clause) — carried into the trace |
| `rules.<id>` | a named expression over facts and other rules |
| `decisions[].when` | an expression; `decisions` are ordered |
| `resolution` | `first_match` (ordered, first true wins; an earlier *unknown* blocks later ones) or `exclusive` (all evaluated; exactly one may be true, else escalate) |
| `default` | decision when nothing matches; otherwise escalate |
| `context` | guidance sent with every question |

Expressions are Python-like: `a and (b or not c)`, `count(a, b, c) >= 2`.
Evaluation is left to right and short-circuits, so put the cheap or decisive
facts first. With `Coverage.FULL` every fact is answered regardless, which is
what you want when a denial must list *every* unmet requirement.

## Examples

Four policies, ~30 cases with ground truth, all passing `pytest`:

| Example | What it decides | Facts (code / judgment) | Decisions |
|---|---|---|---|
| `examples/schema-change` | can a change to an Avro/Protobuf event schema on a shared topic be merged | 7 / 3 | 6 |
| `examples/predictive-maintenance` | stop / schedule / escalate / monitor a robot cell after a monitoring cycle | 7 / 2 | 5 |
| `examples/release-gate` | can a PR be merged to the release branch | 7 / 4 | 7 |
| `examples/credit` | a small-business credit-line application | 7 / 3 | 6 |

Each case file carries the case data, the ground-truth answers for the
judgment facts (`truth`) and the expected decision, so the examples run offline
(`kanon solve` answers judgment facts from `truth` when no `--llm` is given).

## Bench

```bash
python scripts/run_bench.py --n 20          # simulated answerer, figures in bench/
python scripts/live_bench.py examples/credit --llm ollama:qwen2.5:14b --n 10
```

The simulated answerer only touches `llm`/`human` facts and is wrong with
probability ε on each. Code facts always run their real check, so a case
decided through code alone is right every time — pass^k flattens at the
share of such cases as k grows. See `bench/results.json` for the numbers
behind the figures.

## Not in scope (yet)

* No policy authoring from documents: a policy is written and reviewed by
  people, possibly with model help, and versioned in git. The lint is there
  for that review.
* No priorities beyond order: `first_match` is the whole conflict model.
* Expression language has no arithmetic over facts, only boolean logic and
  `count`. Numbers belong in `check` expressions of code facts.
* Checks run in a whitelisted AST sandbox, not a security boundary against a
  hostile policy author. Review policies like code, because they are.

## Relationship to prior work

Executing rules deterministically while a model supplies only the facts is an
old idea (expert systems, decision tables, Prolog with an external oracle).
Anterior's [symbolic-reasoning](https://github.com/anteriorai/symbolic-reasoning)
(October 2026) brought it to agent design with a path-based DSL and an MCP
server for clinical policies; this project explores a different corner of the
same space — ordered decisions with three-valued logic, declared answerers
per fact, confidence gating and reliability measurement — and shares no code
with it.

Apache-2.0.
