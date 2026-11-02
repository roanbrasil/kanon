"""Policy model: the rules, written once, validated at load time.

A policy is a YAML document:

    policy: schema-change-governance
    version: 3
    context: >
      Guidance passed to every LLM-answered fact.
    abstain_below: 0.7          # LLM answers below this confidence become unknown

    facts:
      removes_field:
        question: Does the change remove an existing field?
        by: code
        check: len(case.change.removed_fields) > 0
      migration_documented:
        question: Does the changelog explain how consumers migrate?
        by: llm
        guidance: A link to a ticket is not enough; look for concrete steps.

    rules:
      breaking: removes_field or changes_type

    decisions:
      - id: approve
        label: Approved
        when: not breaking and migration_documented
      - id: review
        label: Needs human review
        when: breaking and migration_documented
      - id: reject
        label: Rejected
        when: breaking and not migration_documented

    resolution: first_match      # or: exclusive
    default: review              # decision when nothing matches (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .logic import ExpressionError, parse, referenced_names

Answerer = Literal["code", "llm", "human"]

ESCALATE = "escalate"
"""Built-in decision id used when the policy cannot be resolved."""


class Fact(BaseModel):
    """An atomic, checkable yes/no question about the case."""

    model_config = ConfigDict(extra="forbid")

    question: str
    by: Answerer = "llm"
    check: str | None = None
    """Expression over ``case`` for ``by: code`` facts."""
    guidance: str | None = None
    """How ambiguities in the question are resolved. Shown to the answerer."""
    abstain_below: float | None = None
    """Per-fact confidence threshold; overrides the policy-wide one."""
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    """Where in the source document this requirement comes from."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "Fact":
        if self.by == "code" and not self.check:
            raise ValueError("a `code` fact needs a `check` expression")
        if self.by != "code" and self.check:
            raise ValueError("only `code` facts may have a `check` expression")
        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    when: str
    label: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str | None = None


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str
    version: int | str = 1
    description: str | None = None
    context: str | None = None
    abstain_below: float = 0.0
    facts: dict[str, Fact]
    rules: dict[str, str] = Field(default_factory=dict)
    decisions: list[Decision]
    resolution: Literal["first_match", "exclusive"] = "first_match"
    default: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------ validation

    @field_validator("facts")
    @classmethod
    def _no_reserved_fact_names(cls, facts: dict[str, Fact]) -> dict[str, Fact]:
        for name in facts:
            if name in {"count", "and", "or", "not", "True", "False", ESCALATE}:
                raise ValueError(f"fact name {name!r} is reserved")
        return facts

    @model_validator(mode="after")
    def _validate_graph(self) -> "Policy":
        names = set(self.facts) | set(self.rules)
        clash = set(self.facts) & set(self.rules)
        if clash:
            raise ValueError(f"names used as both fact and rule: {sorted(clash)}")
        ids = [d.id for d in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate decision ids")
        if ESCALATE in ids:
            raise ValueError(f"decision id {ESCALATE!r} is reserved")
        if self.default is not None and self.default not in ids and self.default != ESCALATE:
            raise ValueError(f"default decision {self.default!r} is not declared")
        for rule, expr in self.rules.items():
            self._check_expr(f"rule {rule}", expr, names)
        for dec in self.decisions:
            self._check_expr(f"decision {dec.id}", dec.when, names)
        for fact_name, fact in self.facts.items():
            if fact.check:
                try:
                    parse(fact.check)
                except ExpressionError as exc:
                    raise ValueError(f"fact {fact_name}: {exc}") from exc
        self._check_rule_cycles()
        return self

    @staticmethod
    def _check_expr(where: str, expr: str, names: set[str]) -> None:
        try:
            refs = referenced_names(expr)
        except ExpressionError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        unknown = [r for r in refs if r not in names and r not in {"count"}]
        if unknown:
            raise ValueError(f"{where} references undeclared names {unknown}")

    def _check_rule_cycles(self) -> None:
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(rule: str, stack: list[str]) -> None:
            if rule in done:
                return
            if rule in visiting:
                raise ValueError(f"rule cycle: {' -> '.join(stack + [rule])}")
            visiting.add(rule)
            for ref in referenced_names(self.rules[rule]):
                if ref in self.rules:
                    visit(ref, stack + [rule])
            visiting.discard(rule)
            done.add(rule)

        for r in self.rules:
            visit(r, [])

    # ------------------------------------------------------------ helpers

    def decision(self, decision_id: str) -> Decision | None:
        for d in self.decisions:
            if d.id == decision_id:
                return d
        return None

    def threshold(self, fact_name: str) -> float:
        fact = self.facts[fact_name]
        return self.abstain_below if fact.abstain_below is None else fact.abstain_below

    def facts_in(self, expr: str) -> list[str]:
        """Facts (transitively through rules) referenced by an expression."""
        out: list[str] = []

        def walk(e: str) -> None:
            for ref in referenced_names(e):
                if ref in self.rules:
                    walk(self.rules[ref])
                elif ref in self.facts and ref not in out:
                    out.append(ref)

        walk(expr)
        return out


def load_policy(source: str | Path | dict[str, Any]) -> Policy:
    """Load a policy from a YAML path, YAML text or an already-parsed mapping."""
    if isinstance(source, dict):
        return Policy.model_validate(source)
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.endswith((".yaml", ".yml"))):
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(str(source))
    return Policy.model_validate(data)
