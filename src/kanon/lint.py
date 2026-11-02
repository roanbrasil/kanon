"""Static checks on a policy, before any case is ever evaluated.

Execution is correct by construction *relative to the policy*, which moves
the risk to the policy itself. ``lint`` catches the mistakes a reviewer is
unlikely to spot by reading: decisions that can never fire, facts nobody
uses, overlapping decisions in an ``exclusive`` policy, and assignments of
facts for which the policy reaches no decision at all.

Overlap, reachability and gaps are decided by enumerating all two-valued
assignments of the facts a decision set references. That is exact up to
``max_facts`` (2^20 ≈ 1M assignments, a couple of seconds); beyond that the
checks fall back to random sampling and say so.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from .logic import evaluate, referenced_names
from .policy import ESCALATE, Policy


@dataclass
class Finding:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    example: dict[str, bool] | None = None

    def __str__(self) -> str:
        s = f"{self.level.upper():<7} {self.code}: {self.message}"
        if self.example:
            facts = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in self.example.items())
            s += f"\n        e.g. {facts}"
        return s


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    exact: bool = True

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    def __str__(self) -> str:
        if not self.findings:
            return "OK: no findings"
        head = "" if self.exact else "(sampled, not exhaustive)\n"
        return head + "\n".join(str(f) for f in self.findings)


def _assignments(facts: list[str], max_facts: int, samples: int, rng: random.Random):
    if len(facts) <= max_facts:
        for bits in itertools.product([False, True], repeat=len(facts)):
            yield dict(zip(facts, bits))
    else:
        for _ in range(samples):
            yield {f: rng.random() < 0.5 for f in facts}


def lint(policy: Policy, max_facts: int = 20, samples: int = 20000, seed: int = 0) -> Report:
    report = Report()
    rng = random.Random(seed)

    used: set[str] = set()
    for d in policy.decisions:
        used.update(policy.facts_in(d.when))
    for name in policy.facts:
        if name not in used:
            report.findings.append(Finding("warning", "unused-fact", f"fact {name!r} is declared but no decision depends on it"))

    used_rules: set[str] = set()

    def mark_rules(expr: str) -> None:
        for ref in referenced_names(expr):
            if ref in policy.rules and ref not in used_rules:
                used_rules.add(ref)
                mark_rules(policy.rules[ref])

    for d in policy.decisions:
        mark_rules(d.when)
    for r in policy.rules:
        if r not in used_rules:
            report.findings.append(Finding("info", "unused-rule", f"rule {r!r} is never referenced by a decision"))

    for name, fact in policy.facts.items():
        if fact.by == "llm" and not fact.guidance:
            report.findings.append(Finding("info", "no-guidance", f"LLM fact {name!r} has no guidance; ambiguities will be resolved by the model"))

    facts = sorted(used)
    report.exact = len(facts) <= max_facts

    fired = {d.id: False for d in policy.decisions}
    overlap_example: dict[tuple[str, str], dict[str, bool]] = {}
    gap_example: dict[str, bool] | None = None
    shadowed: dict[str, bool] = {d.id: True for d in policy.decisions}  # first_match: never first true

    for assignment in _assignments(facts, max_facts, samples, rng):
        def lookup(n: str, a=assignment):
            if n in policy.rules:
                return evaluate(policy.rules[n], lookup).value
            return a.get(n)

        values = [(d.id, evaluate(d.when, lookup).value) for d in policy.decisions]
        true = [d for d, v in values if v]
        for d in true:
            fired[d] = True
        if true:
            shadowed[true[0]] = False
        if not true and gap_example is None:
            gap_example = dict(assignment)
        if len(true) > 1:
            for a, b in itertools.combinations(true, 2):
                overlap_example.setdefault((a, b), dict(assignment))

    for d, ok in fired.items():
        if not ok:
            report.findings.append(Finding("error", "unreachable-decision", f"decision {d!r} can never be true"))
    if policy.resolution == "first_match":
        for d, s in shadowed.items():
            if s and fired[d]:
                report.findings.append(Finding("warning", "shadowed-decision", f"decision {d!r} is always preceded by another true decision and never wins"))
    if policy.resolution == "exclusive":
        for (a, b), ex in overlap_example.items():
            report.findings.append(Finding("error", "overlap", f"decisions {a!r} and {b!r} can both be true in an exclusive policy", ex))
    elif overlap_example:
        pairs = sorted({f"{a}/{b}" for a, b in overlap_example})
        report.findings.append(Finding("info", "overlap", f"decisions can overlap; first_match order decides: {', '.join(pairs)}"))
    if gap_example is not None:
        if policy.default and policy.default != ESCALATE:
            report.findings.append(Finding("info", "gap", f"some cases match no decision and fall back to {policy.default!r}", gap_example))
        else:
            report.findings.append(Finding("warning", "gap", "some cases match no decision and will be escalated", gap_example))
    return report
