"""Oracles answer atomic facts. They never see the policy.

A ``Verdict`` is yes / no / unknown with a confidence in [0, 1] and the
evidence the answerer relied on. ``unknown`` is a first-class answer: the
solver treats it with three-valued logic and, if the policy cannot be
resolved, escalates instead of guessing.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol

from .logic import safe_eval_check
from .policy import Fact, Policy

Value = Literal["yes", "no", "unknown"]


@dataclass(frozen=True)
class Verdict:
    value: Value
    confidence: float = 1.0
    evidence: str = ""
    source: str = ""

    @property
    def tri(self) -> bool | None:
        return {"yes": True, "no": False, "unknown": None}[self.value]

    @staticmethod
    def yes(evidence: str = "", confidence: float = 1.0, source: str = "") -> "Verdict":
        return Verdict("yes", confidence, evidence, source)

    @staticmethod
    def no(evidence: str = "", confidence: float = 1.0, source: str = "") -> "Verdict":
        return Verdict("no", confidence, evidence, source)

    @staticmethod
    def unknown(evidence: str = "", source: str = "") -> "Verdict":
        return Verdict("unknown", 0.0, evidence, source)

    @staticmethod
    def from_bool(value: bool, evidence: str = "", confidence: float = 1.0, source: str = "") -> "Verdict":
        return Verdict("yes" if value else "no", confidence, evidence, source)


@dataclass(frozen=True)
class Question:
    """One fact the solver needs answered, with everything the answerer may see."""

    fact_id: str
    fact: Fact
    policy_context: str | None
    known: Mapping[str, Verdict] = field(default_factory=dict)
    """Facts already answered on the way to this one (never the rules)."""


class Oracle(Protocol):
    def __call__(self, question: Question, case: Mapping[str, Any]) -> Verdict: ...


# -------------------------------------------------------------------- code


def code_oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
    """Answer a ``by: code`` fact by evaluating its ``check`` expression."""
    assert question.fact.check is not None
    try:
        result = safe_eval_check(question.fact.check, case)
    except (KeyError, AttributeError, TypeError, IndexError) as exc:
        return Verdict.unknown(f"check could not be evaluated on this case: {exc!r}", source="code")
    return Verdict.from_bool(bool(result), evidence=f"{question.fact.check} -> {result!r}", source="code")


# ------------------------------------------------------------------- human


def cli_oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
    """Ask at the terminal. Accepts y / n / ? (unknown)."""
    print(f"\n[{question.fact_id}] {question.fact.question}")
    if question.fact.guidance:
        print(f"  guidance: {question.fact.guidance}")
    while True:
        reply = input("  answer (y/n/?): ").strip().lower()
        if reply in {"y", "yes"}:
            return Verdict.yes("answered at the terminal", source="human")
        if reply in {"n", "no"}:
            return Verdict.no("answered at the terminal", source="human")
        if reply in {"?", "unknown"}:
            return Verdict.unknown("the reviewer could not tell", source="human")


# ------------------------------------------------------------------- tables


def table_oracle(answers: Mapping[str, Verdict | bool | None], source: str = "table") -> Oracle:
    """Answer from a fixed mapping; useful for tests, fixtures and replays."""

    def oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
        if question.fact_id not in answers:
            return Verdict.unknown("no answer on record", source=source)
        a = answers[question.fact_id]
        if isinstance(a, Verdict):
            return a
        if a is None:
            return Verdict.unknown("recorded as unknown", source=source)
        return Verdict.from_bool(bool(a), evidence="recorded answer", source=source)

    return oracle


def noisy_oracle(
    truth: Mapping[str, bool],
    error_rate: float,
    rng: random.Random,
    confidence_when_wrong: tuple[float, float] = (0.35, 0.85),
    confidence_when_right: tuple[float, float] = (0.65, 1.0),
    source: str = "noisy",
) -> Oracle:
    """Simulate an imperfect answerer: flips each answer with ``error_rate``.

    Confidence is sampled from different ranges when right and when wrong,
    so that an ``abstain_below`` threshold has something to bite on. The
    ranges overlap on purpose: a calibrated-but-imperfect answerer, not a
    cheat.
    """

    def oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
        true_value = truth[question.fact_id]
        wrong = rng.random() < error_rate
        value = (not true_value) if wrong else true_value
        lo, hi = confidence_when_wrong if wrong else confidence_when_right
        return Verdict.from_bool(value, evidence="simulated", confidence=rng.uniform(lo, hi), source=source)

    return oracle


# --------------------------------------------------------------------- LLM

_PROMPT = """You answer ONE yes/no question about a case. You do not know the policy
the question belongs to and you must not guess it; judge only the question in
front of you against the case evidence.

{policy_context}
Question: {question}
{guidance}
{known}
Case (JSON):
{case}

Reply with a single JSON object and nothing else:
{{"answer": "yes" | "no" | "unknown", "confidence": <0.0-1.0>, "evidence": "<the specific case fields or passages you relied on>"}}
Use "unknown" when the case does not contain enough evidence to decide."""


def build_prompt(question: Question, case: Mapping[str, Any]) -> str:
    known = ""
    if question.known:
        lines = [f"- {k}: {v.value}" for k, v in question.known.items()]
        known = "Facts already established for this case:\n" + "\n".join(lines) + "\n"
    return _PROMPT.format(
        policy_context=f"Context: {question.policy_context}\n" if question.policy_context else "",
        question=question.fact.question,
        guidance=f"How to interpret the question: {question.fact.guidance}\n" if question.fact.guidance else "",
        known=known,
        case=json.dumps(case, ensure_ascii=False, indent=2),
    )


def parse_llm_reply(text: str, source: str) -> Verdict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return Verdict.unknown(f"unparseable reply: {text[:200]!r}", source=source)
    try:
        data = json.loads(text[start : end + 1])
        value = str(data.get("answer", "unknown")).lower()
        if value not in {"yes", "no", "unknown"}:
            value = "unknown"
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        return Verdict(value, conf if value != "unknown" else 0.0, str(data.get("evidence", "")), source)
    except (ValueError, TypeError) as exc:
        return Verdict.unknown(f"unparseable reply ({exc}): {text[:200]!r}", source=source)


def anthropic_oracle(model: str = "claude-sonnet-4-5", max_tokens: int = 400) -> Oracle:
    """An oracle backed by the Anthropic API (``pip install kanon[llm]``)."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the `anthropic` package to use anthropic_oracle") from exc
    client = anthropic.Anthropic()

    def oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": build_prompt(question, case)}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return parse_llm_reply(text, source=f"anthropic:{model}")

    return oracle


def ollama_oracle(model: str = "qwen2.5:14b", host: str | None = None) -> Oracle:
    """An oracle backed by a local Ollama server (no extra dependencies)."""
    base = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
        body = json.dumps(
            {
                "model": model,
                "prompt": build_prompt(question, case),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(f"{base}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - local server
            data = json.loads(resp.read())
        return parse_llm_reply(data.get("response", ""), source=f"ollama:{model}")

    return oracle


# ------------------------------------------------------------------ router


def route(
    policy: Policy,
    llm: Oracle | None = None,
    human: Oracle | None = None,
    code: Oracle = code_oracle,
) -> Oracle:
    """Dispatch each question to the answerer its fact declares.

    Facts the policy marks ``by: code`` never reach the model, which is the
    point: the model only answers what is genuinely a matter of judgment.
    """
    table: dict[str, Oracle | None] = {"code": code, "llm": llm, "human": human}

    def oracle(question: Question, case: Mapping[str, Any]) -> Verdict:
        answerer = table[question.fact.by]
        if answerer is None:
            return Verdict.unknown(f"no {question.fact.by} oracle configured", source="router")
        return answerer(question, case)

    return oracle


OracleFactory = Callable[[], Oracle]
