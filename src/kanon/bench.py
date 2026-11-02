"""pass^k and pass@k over repeated solves.

Both estimators follow τ-bench (Yao et al., 2024): with ``n`` trials of a
task and ``c`` successes,

    pass^k = C(c, k) / C(n, k)              all k trials succeed
    pass@k = 1 - C(n - c, k) / C(n, k)      at least one of k succeeds

averaged across tasks. ``pass^1 == pass@1`` is the ordinary success rate; the
distance between pass^k and pass@k as k grows is the reliability gap.

A *success* here is "the solver reached the expected decision". An
escalation is counted separately: it is not a wrong decision, it is a
decision not to decide, and the right number to watch alongside pass^k is
how often that happens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable, Mapping, Sequence

from .oracle import Oracle, Verdict, noisy_oracle, route
from .policy import ESCALATE, Policy
from .solver import Coverage, Outcome, solve


@dataclass
class Case:
    id: str
    data: Mapping[str, Any]
    expected: str
    truth: Mapping[str, bool] = field(default_factory=dict)
    """Ground-truth answers for the facts an answerer could get wrong."""
    note: str | None = None


def pass_hat_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError("k cannot exceed the number of trials")
    return comb(c, k) / comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError("k cannot exceed the number of trials")
    return 1.0 - comb(n - c, k) / comb(n, k)


@dataclass
class TrialStats:
    case_id: str
    expected: str
    decisions: list[str]

    @property
    def n(self) -> int:
        return len(self.decisions)

    @property
    def correct(self) -> int:
        return sum(1 for d in self.decisions if d == self.expected)

    @property
    def escalated(self) -> int:
        return sum(1 for d in self.decisions if d == ESCALATE)

    @property
    def wrong(self) -> int:
        return self.n - self.correct - self.escalated


@dataclass
class BenchResult:
    label: str
    trials: list[TrialStats]
    ks: Sequence[int]

    def metric(self, fn: Callable[[int, int, int], float], k: int) -> float:
        return sum(fn(t.n, t.correct, k) for t in self.trials) / len(self.trials)

    @property
    def table(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in self.ks:
            out[f"pass^{k}"] = self.metric(pass_hat_k, k)
        for k in self.ks:
            out[f"pass@{k}"] = self.metric(pass_at_k, k)
        n = sum(t.n for t in self.trials)
        out["escalation_rate"] = sum(t.escalated for t in self.trials) / n
        out["wrong_rate"] = sum(t.wrong for t in self.trials) / n
        return out

    def __str__(self) -> str:
        t = self.table
        cols = "  ".join(f"{k}={v:.3f}" for k, v in t.items())
        return f"{self.label:<28} {cols}"


def run(
    policy: Policy,
    cases: Sequence[Case],
    make_oracle: Callable[[Case, random.Random], Oracle],
    n: int = 20,
    ks: Sequence[int] = (1, 5, 10, 20),
    seed: int = 0,
    label: str = "kanon",
    coverage: Coverage = Coverage.MINIMAL,
) -> BenchResult:
    """Solve every case ``n`` times with fresh oracle randomness per trial."""
    rng = random.Random(seed)
    trials = []
    for case in cases:
        decisions = []
        for _ in range(n):
            oracle = make_oracle(case, random.Random(rng.random()))
            out: Outcome = solve(policy, case.data, oracle, coverage=coverage)
            decisions.append(out.decision)
        trials.append(TrialStats(case.id, case.expected, decisions))
    return BenchResult(label, trials, [k for k in ks if k <= n])


def simulated_llm(policy: Policy, error_rate: float) -> Callable[[Case, random.Random], Oracle]:
    """Build a router whose LLM facts are answered by a noisy simulator.

    Code facts run their real checks; only facts the policy marks ``by: llm``
    (or ``human``) can be wrong, and they are wrong with ``error_rate``.
    """

    def make(case: Case, rng: random.Random) -> Oracle:
        noisy = noisy_oracle(case.truth, error_rate, rng)
        return route(policy, llm=noisy, human=noisy)

    return make
