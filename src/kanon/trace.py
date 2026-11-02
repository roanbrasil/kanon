"""Reasoning traces: the computation itself, serialized.

Nothing here is generated after the fact. The trace lists each round of
questions, every verdict with its source, confidence and evidence, how each
rule and decision evaluated, and why the final decision (or escalation) was
reached. If the policy is the contract, the trace is the receipt.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .logic import evaluate
from .solver import Outcome


def _tri(v: bool | None) -> str:
    return {True: "YES", False: "NO", None: "UNKNOWN"}[v]


def to_dict(outcome: Outcome) -> dict[str, Any]:
    ex = outcome.execution
    pol = ex.policy
    facts = {}
    for fact_id, v in ex.verdicts.items():
        facts[fact_id] = {
            **asdict(v),
            "question": pol.facts[fact_id].question,
            "by": pol.facts[fact_id].by,
            "threshold": pol.threshold(fact_id),
            "effective": _tri(ex.effective(fact_id)),
            "gated": ex.gated(fact_id),
        }
    return {
        "policy": pol.policy,
        "version": pol.version,
        "coverage": ex.coverage.value,
        "decision": outcome.decision,
        "label": outcome.label,
        "escalated": outcome.escalated,
        "reason": outcome.resolution.reason,
        "conflict": outcome.resolution.conflict,
        "unresolved": outcome.resolution.unresolved,
        "rounds": [{"round": r.index, "asked": r.asked} for r in ex.rounds],
        "facts": facts,
        "rules": {name: _tri(ex.lookup(name)) for name in pol.rules},
        "decisions": [
            {"id": d.id, "when": d.when, "value": _tri(evaluate(d.when, ex.lookup).value)}
            for d in pol.decisions
        ],
    }


def to_json(outcome: Outcome, indent: int = 2) -> str:
    return json.dumps(to_dict(outcome), indent=indent, ensure_ascii=False)


def requirements(outcome: Outcome, decision_id: str) -> list[dict[str, Any]]:
    """Every fact ``decision_id`` depends on, with how it came out in this execution.

    Solve with ``Coverage.FULL`` first if you need every fact answered, e.g. to
    list all the reasons an application was not approved, not just the first.
    """
    ex = outcome.execution
    dec = ex.policy.decision(decision_id)
    if dec is None:
        raise KeyError(decision_id)
    out = []
    for fact_id in ex.policy.facts_in(dec.when):
        f = ex.policy.facts[fact_id]
        out.append(
            {
                "fact": fact_id,
                "question": f.question,
                "value": _tri(ex.effective(fact_id)) if fact_id in ex.verdicts else "NOT ASKED",
                "source": f.source,
            }
        )
    return out


def explain(outcome: Outcome, show_evidence: bool = True) -> str:
    """A plain-text trace a reviewer can read top to bottom."""
    ex = outcome.execution
    pol = ex.policy
    lines = [f"Policy {pol.policy} v{pol.version}  ->  {outcome.label} [{outcome.decision}]"]
    lines.append(f"  because: {outcome.resolution.reason}")
    lines.append("")
    for r in ex.rounds:
        lines.append(f"Round {r.index}")
        for fact_id in r.asked:
            v = r.verdicts[fact_id]
            f = pol.facts[fact_id]
            mark = _tri(ex.effective(fact_id))
            gate = ""
            if ex.gated(fact_id):
                gate = f"  (answered {v.value.upper()} at {v.confidence:.2f}, below threshold {pol.threshold(fact_id):.2f})"
            lines.append(f"  {mark:<8} {f.question}")
            lines.append(f"           by {v.source or f.by}, confidence {v.confidence:.2f}{gate}")
            if show_evidence and v.evidence:
                lines.append(f"           evidence: {v.evidence}")
    if pol.rules:
        lines.append("")
        lines.append("Rules")
        for name, expr in pol.rules.items():
            lines.append(f"  {_tri(ex.lookup(name)):<8} {name} = {expr}")
    lines.append("")
    lines.append("Decisions (in order)")
    for d in pol.decisions:
        v = evaluate(d.when, ex.lookup).value
        marker = "<==" if d.id == outcome.decision else "   "
        lines.append(f"  {_tri(v):<8} {d.id:<16} when {d.when}  {marker}")
    return "\n".join(lines)
