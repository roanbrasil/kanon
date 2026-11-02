"""The deterministic solver.

The solver is a pure function of the policy and the verdicts: same verdicts,
same decision, same trace, every time. Non-determinism can only enter through
the oracle, one fact at a time, and each fact's verdict is recorded with its
source, confidence and evidence.

Resolution semantics
--------------------
* ``first_match``: decisions are ordered. The outcome is the first decision
  whose condition is **true**. If an earlier decision is still **unknown**,
  a later true decision cannot win (it might be overridden), so the solver
  keeps asking; if nothing can settle it, the case is escalated.
* ``exclusive``: decisions are meant to be mutually exclusive (``kanon lint``
  checks that statically). Every decision is evaluated; exactly one true
  decision wins, zero falls back to ``default`` or escalation, and two or
  more true decisions escalate with a conflict reason.

Confidence gating
-----------------
A verdict below the fact's ``abstain_below`` threshold is recorded as given
but **treated as unknown** when evaluating the policy. The trace shows both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .logic import Tri, evaluate
from .oracle import Oracle, Question, Verdict
from .policy import ESCALATE, Policy


class Coverage(str, Enum):
    MINIMAL = "minimal"
    """Stop asking as soon as the decision is settled."""
    FULL = "full"
    """Answer every fact the policy declares, even when the decision is already settled.
    Useful to report every unmet requirement, at the cost of more questions."""


@dataclass
class Round:
    index: int
    asked: list[str]
    verdicts: dict[str, Verdict]


@dataclass
class Execution:
    policy: Policy
    case: Mapping[str, Any]
    coverage: Coverage = Coverage.MINIMAL
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    rounds: list[Round] = field(default_factory=list)

    # ---------------------------------------------------------------- facts

    def effective(self, fact_id: str) -> Tri:
        """The fact's value as the policy sees it: gated by confidence."""
        v = self.verdicts.get(fact_id)
        if v is None or v.value == "unknown":
            return None
        if v.confidence < self.policy.threshold(fact_id):
            return None
        return v.tri

    def gated(self, fact_id: str) -> bool:
        """True when a confident-looking answer was withheld by the threshold."""
        v = self.verdicts.get(fact_id)
        return v is not None and v.value != "unknown" and v.confidence < self.policy.threshold(fact_id)

    def lookup(self, name: str) -> Tri:
        if name in self.policy.rules:
            return evaluate(self.policy.rules[name], self.lookup).value
        return self.effective(name)

    def needed(self, expr: str) -> list[str]:
        """Facts that would advance ``expr``, resolving rules transitively."""
        out: list[str] = []
        for name in evaluate(expr, self.lookup).needed:
            if name in self.policy.rules:
                out.extend(n for n in self.needed(self.policy.rules[name]) if n not in out)
            elif name not in out:
                out.append(name)
        return out

    def exhausted(self, fact_id: str) -> bool:
        """A fact that was asked and came back unknown (or gated) cannot be asked again."""
        return fact_id in self.verdicts

    # ----------------------------------------------------------- decisions

    def decision_values(self) -> list[tuple[str, Tri]]:
        return [(d.id, evaluate(d.when, self.lookup).value) for d in self.policy.decisions]

    def resolve(self) -> "Resolution | None":
        """The decision if it can be settled with current knowledge, else None."""
        values = self.decision_values()
        pol = self.policy
        if pol.resolution == "first_match":
            for dec_id, v in values:
                if v is True:
                    return Resolution(dec_id, reason=f"first matching decision: {dec_id}")
                if v is None:
                    return None  # an earlier decision could still win
            return self._fallback(values)
        # exclusive
        if any(v is None for _, v in values):
            return None
        true = [d for d, v in values if v is True]
        if len(true) == 1:
            return Resolution(true[0], reason=f"only matching decision: {true[0]}")
        if len(true) > 1:
            return Resolution(ESCALATE, reason=f"conflicting decisions matched: {true}", conflict=true)
        return self._fallback(values)

    def _fallback(self, values: list[tuple[str, Tri]]) -> "Resolution":
        if self.policy.default and self.policy.default != ESCALATE:
            return Resolution(self.policy.default, reason="no decision matched; policy default applied")
        return Resolution(ESCALATE, reason="no decision matched and the policy declares no default")

    def escalation(self) -> "Resolution":
        """Called when questions are exhausted and the policy is still unresolved."""
        unknown = [f for f in self.policy.facts if self.effective(f) is None and self.exhausted(f)]
        gated = [f for f in unknown if self.gated(f)]
        reason = "unresolved facts: " + ", ".join(unknown)
        if gated:
            reason += f" (withheld for low confidence: {', '.join(gated)})"
        return Resolution(ESCALATE, reason=reason, unresolved=unknown)


@dataclass
class Resolution:
    decision: str
    reason: str
    conflict: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def escalated(self) -> bool:
        return self.decision == ESCALATE


# ---------------------------------------------------------------- stepping


def next_questions(execution: Execution) -> list[str]:
    """Facts to ask in the next round, or ``[]`` when the solve is finished."""
    ex = execution
    if ex.coverage == Coverage.FULL:
        pending = [f for f in ex.policy.facts if not ex.exhausted(f)]
        return pending
    if ex.resolve() is not None:
        return []
    wanted: list[str] = []
    for dec in ex.policy.decisions:
        value = evaluate(dec.when, ex.lookup).value
        for f in ex.needed(dec.when):
            if not ex.exhausted(f) and f not in wanted:
                wanted.append(f)
        if ex.policy.resolution == "first_match" and value is None:
            # Nothing after this decision can win until it is settled, so
            # asking further questions now would be wasted work.
            break
    return wanted


def known_for(execution: Execution, fact_id: str) -> dict[str, Verdict]:
    """Verdicts already on record that share a decision with ``fact_id``."""
    related: set[str] = set()
    for dec in execution.policy.decisions:
        facts = execution.policy.facts_in(dec.when)
        if fact_id in facts:
            related.update(facts)
    return {f: v for f, v in execution.verdicts.items() if f in related and f != fact_id}


def apply(execution: Execution, verdicts: Mapping[str, Verdict]) -> Execution:
    """Record a round of verdicts. Returns the same (mutated) execution."""
    for fact_id, v in verdicts.items():
        if fact_id not in execution.policy.facts:
            raise KeyError(f"unknown fact {fact_id!r}")
        execution.verdicts[fact_id] = v
    execution.rounds.append(Round(len(execution.rounds) + 1, list(verdicts), dict(verdicts)))
    return execution


def solve(
    policy: Policy,
    case: Mapping[str, Any],
    oracle: Oracle,
    coverage: Coverage = Coverage.MINIMAL,
    max_rounds: int = 100,
) -> "Outcome":
    ex = Execution(policy, case, coverage)
    while True:
        asked = next_questions(ex)
        if not asked:
            break
        if len(ex.rounds) >= max_rounds:
            raise RuntimeError("solver did not converge; is the oracle answering?")
        verdicts: dict[str, Verdict] = {}
        for fact_id in asked:
            q = Question(fact_id, policy.facts[fact_id], policy.context, known_for(ex, fact_id))
            verdicts[fact_id] = oracle(q, case)
        apply(ex, verdicts)
    res = ex.resolve() or ex.escalation()
    return Outcome(ex, res)


@dataclass
class Outcome:
    execution: Execution
    resolution: Resolution

    @property
    def decision(self) -> str:
        return self.resolution.decision

    @property
    def escalated(self) -> bool:
        return self.resolution.escalated

    @property
    def label(self) -> str:
        d = self.execution.policy.decision(self.decision)
        return (d.label or d.id) if d else "Escalated to a human"
