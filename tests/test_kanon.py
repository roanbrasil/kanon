import random
from pathlib import Path

import pytest

import kanon
from kanon import bench, logic
from kanon.cli import load_cases

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

LOAN = """
policy: loan
abstain_below: 0.7
facts:
  good_credit: {question: "Score >= 700?", by: code, check: "case.score >= 700"}
  stable_income: {question: "Two years of income?", by: llm}
  has_down_payment: {question: "20% down?", by: code, check: "case.down >= 0.2"}
rules:
  core: good_credit and stable_income
decisions:
  - {id: approve, when: core and has_down_payment}
  - {id: review, when: core and not has_down_payment}
  - {id: deny, when: not core}
"""


# ----------------------------------------------------------------- logic


def test_kleene_and_or():
    assert logic.k_and([True, None]) is None
    assert logic.k_and([False, None]) is False
    assert logic.k_or([True, None]) is True
    assert logic.k_or([False, None]) is None
    assert logic.k_not(None) is None


@pytest.mark.parametrize(
    "values,op,n,expected",
    [
        ([True, True, None], ">=", 2, True),
        ([True, None, None], ">=", 2, None),
        ([True, False, False], ">=", 2, False),
        ([None, None, None], "<=", 1, None),
        ([True, True, None], "==", 2, None),
        ([True, True, False], "==", 2, True),
    ],
)
def test_count_comparisons(values, op, n, expected):
    assert logic.k_compare_count(values, op, n) is expected


def test_evaluate_short_circuit_needed():
    known = {"a": True}
    ev = logic.evaluate("a and b and c", lambda n: known.get(n))
    assert ev.value is None and ev.needed == ["b"]
    known["a"] = False
    ev = logic.evaluate("a and b and c", lambda n: known.get(n))
    assert ev.value is False and ev.needed == []


def test_checks_are_sandboxed():
    with pytest.raises(logic.ExpressionError):
        logic.safe_eval_check("__import__('os').system('true')", {})
    with pytest.raises(logic.ExpressionError):
        logic.safe_eval_check("case.__class__", {"case": {}})
    assert logic.safe_eval_check("len(case.items) > 1 and case.x.y == 3", {"items": [1, 2], "x": {"y": 3}}) is True


# ---------------------------------------------------------------- policy


def test_policy_rejects_undeclared_names():
    with pytest.raises(ValueError, match="undeclared"):
        kanon.load_policy(LOAN.replace("not core", "not cor"))


def test_policy_rejects_rule_cycles():
    bad = LOAN.replace("core: good_credit and stable_income", "core: other\n  other: core")
    with pytest.raises(ValueError, match="cycle"):
        kanon.load_policy(bad)


# ---------------------------------------------------------------- solver


def test_solve_is_deterministic_and_traces():
    policy = kanon.load_policy(LOAN)
    case = {"score": 720, "down": 0.1}
    oracle = kanon.route(policy, llm=kanon.table_oracle({"stable_income": True}))
    a = kanon.solve(policy, case, oracle)
    b = kanon.solve(policy, case, oracle)
    assert a.decision == b.decision == "review"
    assert kanon.to_dict(a) == kanon.to_dict(b)
    assert "stable_income" in a.execution.verdicts
    assert "review" in kanon.explain(a)


def test_code_facts_never_reach_the_llm():
    policy = kanon.load_policy(LOAN)
    seen = []

    def spy(q, case):
        seen.append(q.fact_id)
        return kanon.Verdict.yes()

    out = kanon.solve(policy, {"score": 720, "down": 0.3}, kanon.route(policy, llm=spy))
    assert out.decision == "approve"
    assert seen == ["stable_income"]


def test_minimal_coverage_short_circuits():
    policy = kanon.load_policy(LOAN)
    out = kanon.solve(policy, {"score": 500, "down": 0.3}, kanon.route(policy, llm=kanon.table_oracle({})))
    # good_credit is false, so stable_income is never asked and `deny` fires
    assert out.decision == "deny"
    assert "stable_income" not in out.execution.verdicts


def test_full_coverage_asks_everything():
    policy = kanon.load_policy(LOAN)
    out = kanon.solve(
        policy, {"score": 500, "down": 0.3}, kanon.route(policy, llm=kanon.table_oracle({"stable_income": True})), coverage=kanon.Coverage.FULL
    )
    assert set(out.execution.verdicts) == set(policy.facts)


def test_unknown_escalates_instead_of_guessing():
    policy = kanon.load_policy(LOAN)
    out = kanon.solve(policy, {"score": 720, "down": 0.3}, kanon.route(policy, llm=None))
    assert out.escalated
    assert "stable_income" in out.resolution.unresolved


def test_low_confidence_is_gated():
    policy = kanon.load_policy(LOAN)
    low = kanon.table_oracle({"stable_income": kanon.Verdict.yes(confidence=0.4)})
    out = kanon.solve(policy, {"score": 720, "down": 0.3}, kanon.route(policy, llm=low))
    assert out.escalated
    assert out.execution.gated("stable_income")
    assert "withheld for low confidence" in out.resolution.reason


def test_first_match_waits_for_earlier_unknown():
    text = LOAN.replace("abstain_below: 0.7", "abstain_below: 0.0")
    policy = kanon.load_policy(text)
    # stable_income unknown: `review` cannot win while `approve` is undecided
    oracle = kanon.route(policy, llm=kanon.table_oracle({"stable_income": None}))
    out = kanon.solve(policy, {"score": 720, "down": 0.1}, oracle)
    assert out.escalated


def test_exclusive_conflict_escalates():
    text = LOAN.replace("resolution: first_match", "") + "\nresolution: exclusive\n"
    text = text.replace("- {id: review, when: core and not has_down_payment}", "- {id: review, when: core}")
    policy = kanon.load_policy(text)
    oracle = kanon.route(policy, llm=kanon.table_oracle({"stable_income": True}))
    out = kanon.solve(policy, {"score": 720, "down": 0.3}, oracle)
    assert out.escalated and out.resolution.conflict == ["approve", "review"]


def test_step_by_step_api():
    policy = kanon.load_policy(LOAN)
    ex = kanon.Execution(policy, {"score": 720, "down": 0.3})
    asked = kanon.next_questions(ex)
    assert asked == ["good_credit"]
    kanon.apply(ex, {"good_credit": kanon.Verdict.yes()})
    assert kanon.next_questions(ex) == ["stable_income"]


# ------------------------------------------------------------------ lint


def test_lint_finds_overlap_and_gap():
    text = LOAN.replace("- {id: deny, when: not core}", "") + "resolution: exclusive\n"
    text = text.replace("- {id: review, when: core and not has_down_payment}", "- {id: review, when: core}")
    report = kanon.lint(kanon.load_policy(text))
    codes = {f.code for f in report.findings}
    assert "overlap" in codes and "gap" in codes
    assert report.errors


def test_lint_unreachable_decision():
    text = LOAN.replace("- {id: deny, when: not core}", "- {id: deny, when: core and not core}")
    report = kanon.lint(kanon.load_policy(text))
    assert any(f.code == "unreachable-decision" for f in report.findings)


# ------------------------------------------------------------------ bench


def test_pass_metrics():
    assert bench.pass_hat_k(4, 4, 4) == 1.0
    assert bench.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert bench.pass_at_k(4, 1, 4) == 1.0
    assert bench.pass_at_k(4, 0, 2) == 0.0


def test_noisy_oracle_is_seeded():
    truth = {"f": True}
    a = kanon.noisy_oracle(truth, 0.5, random.Random(1))
    b = kanon.noisy_oracle(truth, 0.5, random.Random(1))
    q = kanon.Question("f", kanon.Fact(question="?"), None)
    assert [a(q, {}).value for _ in range(10)] == [b(q, {}).value for _ in range(10)]


# --------------------------------------------------------------- examples


@pytest.mark.parametrize("domain", ["schema-change", "predictive-maintenance", "release-gate", "credit"])
def test_examples_reach_expected_decisions(domain):
    policy = kanon.load_policy(EXAMPLES / domain / "policy.yaml")
    assert not kanon.lint(policy).errors
    for case in load_cases(EXAMPLES / domain):
        answers = {k: kanon.Verdict.from_bool(v) for k, v in case.truth.items()}
        oracle = kanon.route(policy, llm=kanon.table_oracle(answers), human=kanon.table_oracle(answers))
        out = kanon.solve(policy, case.data, oracle)
        assert out.decision == case.expected, f"{domain}/{case.id}: {out.resolution.reason}"
