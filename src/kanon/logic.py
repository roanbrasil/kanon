"""Three-valued (Kleene) logic and the policy expression language.

A policy expression is Python-like infix syntax over fact and rule names:

    compatible and documented and not breaking
    count(vibration_high, temp_rising, noise_reported) >= 2
    (a or b) and not (c and d)

Values are ``True``, ``False`` or ``None`` (unknown). Unknown propagates the
way Kleene's strong logic prescribes: ``False and None`` is ``False``,
``True and None`` is ``None``, ``True or None`` is ``True``.

The same parser evaluates ``check:`` expressions of code facts against the
case data, with a whitelist of safe node types and builtins.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

Tri = bool | None


# ---------------------------------------------------------------- Kleene ops


def k_and(values: list[Tri]) -> Tri:
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def k_or(values: list[Tri]) -> Tri:
    if any(v is True for v in values):
        return True
    if any(v is None for v in values):
        return None
    return False


def k_not(value: Tri) -> Tri:
    return None if value is None else (not value)


def k_count(values: list[Tri]) -> tuple[int, int]:
    """Return (known_true, unknown)."""
    return sum(1 for v in values if v is True), sum(1 for v in values if v is None)


def k_compare_count(values: list[Tri], op: str, n: int) -> Tri:
    """Three-valued comparison of ``count(...)`` against an integer."""
    t, u = k_count(values)
    lo, hi = t, t + u  # the true count lies in [lo, hi]
    if op == ">=":
        return True if lo >= n else (False if hi < n else None)
    if op == ">":
        return True if lo > n else (False if hi <= n else None)
    if op == "<=":
        return True if hi <= n else (False if lo > n else None)
    if op == "<":
        return True if hi < n else (False if lo >= n else None)
    if op == "==":
        if lo == hi == n:
            return True
        if n < lo or n > hi:
            return False
        return None
    if op == "!=":
        r = k_compare_count(values, "==", n)
        return k_not(r)
    raise ValueError(f"unsupported count comparison {op!r}")


# ------------------------------------------------------- expression parsing

_CMP = {
    ast.GtE: ">=",
    ast.Gt: ">",
    ast.LtE: "<=",
    ast.Lt: "<",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


class ExpressionError(ValueError):
    pass


def parse(source: str) -> ast.expr:
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:  # pragma: no cover - message passthrough
        raise ExpressionError(f"cannot parse {source!r}: {exc.msg}") from exc
    return tree.body


def referenced_names(source: str) -> list[str]:
    """Names referenced by a policy expression, in evaluation order."""
    names: list[str] = []

    def walk(node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            if node.id not in names:
                names.append(node.id)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                walk(v)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            walk(node.operand)
        elif isinstance(node, ast.Compare):
            walk(node.left)
            for c in node.comparators:
                walk(c)
        elif isinstance(node, ast.Call):
            for a in node.args:
                walk(a)
        elif isinstance(node, ast.Constant):
            pass
        else:
            raise ExpressionError(f"unsupported syntax in policy expression: {ast.dump(node)}")

    walk(parse(source))
    return names


@dataclass
class Eval:
    """Result of evaluating a policy expression against partial knowledge."""

    value: Tri
    needed: list[str] = field(default_factory=list)
    """Names whose answers would advance the evaluation, in order.

    Short-circuit aware: for ``a and b and c`` with ``a`` unknown only ``a``
    is needed; once ``a`` is true, ``b`` is needed, and so on.
    """


def evaluate(source: str, lookup: Callable[[str], Tri]) -> Eval:
    """Evaluate a policy expression with Kleene semantics.

    ``lookup(name)`` returns True/False/None for a fact or rule name.
    """

    def ev(node: ast.expr) -> Eval:
        if isinstance(node, ast.Name):
            v = lookup(node.id)
            return Eval(v, [node.id] if v is None else [])
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return Eval(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = ev(node.operand)
            return Eval(k_not(inner.value), inner.needed)
        if isinstance(node, ast.BoolOp):
            parts = [ev(v) for v in node.values]
            vals = [p.value for p in parts]
            if isinstance(node.op, ast.And):
                result = k_and(vals)
            else:
                result = k_or(vals)
            needed: list[str] = []
            if result is None:
                # ask the first unknown operand (left-to-right short circuit)
                for p in parts:
                    if p.value is None:
                        needed = p.needed
                        break
            return Eval(result, needed)
        if isinstance(node, ast.Compare):
            if not (
                isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "count"
                and len(node.ops) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and isinstance(node.comparators[0].value, int)
            ):
                raise ExpressionError(
                    "comparisons are only allowed as count(a, b, ...) <op> N"
                )
            parts = [ev(a) for a in node.left.args]
            op = _CMP.get(type(node.ops[0]))
            if op is None:
                raise ExpressionError("unsupported comparison operator")
            result = k_compare_count([p.value for p in parts], op, node.comparators[0].value)
            needed = []
            if result is None:
                for p in parts:
                    if p.value is None:
                        needed = p.needed
                        break
            return Eval(result, needed)
        raise ExpressionError(f"unsupported syntax in policy expression: {ast.dump(node)}")

    return ev(parse(source))


# --------------------------------------------- safe evaluation of check exprs

_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "any": any,
    "all": all,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
}

_SAFE_METHODS = {
    "get", "keys", "values", "items",
    "lower", "upper", "strip", "startswith", "endswith", "split",
}

_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.GeneratorExp,
    ast.SetComp,
    ast.comprehension,
    ast.IfExp,
    ast.Store,
)


class _Node(dict):
    """A dict that also allows attribute access, so checks can write
    ``case.schema.added_fields`` instead of ``case["schema"]["added_fields"]``."""

    def __getattribute__(self, name: str) -> Any:
        # case keys win over dict methods, so `case.items` means the field
        if not name.startswith("_") and dict.__contains__(self, name):
            return _wrap(dict.__getitem__(self, name))
        return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        return _wrap(super().__getitem__(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return _wrap(super().get(key, default))


def _wrap(value: Any) -> Any:
    if isinstance(value, _Node):
        return value
    if isinstance(value, dict):
        return _Node(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def safe_eval_check(source: str, case: Mapping[str, Any]) -> Any:
    """Evaluate a ``check:`` expression against case data.

    Only a whitelist of node types is accepted, no attribute starting with
    ``_`` can be read, and the only callables are a handful of builtins.
    """
    tree = ast.parse(source.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ExpressionError(f"check expression uses forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ExpressionError("private attribute access is not allowed in checks")
        if isinstance(node, ast.Call):
            builtin = isinstance(node.func, ast.Name) and node.func.id in _SAFE_BUILTINS
            method = isinstance(node.func, ast.Attribute) and node.func.attr in _SAFE_METHODS
            if not (builtin or method):
                raise ExpressionError("only whitelisted builtins and methods may be called in checks")
    env = {"case": _wrap(dict(case)), "__builtins__": {}}
    env.update(_SAFE_BUILTINS)
    return eval(compile(tree, "<check>", "eval"), env)  # noqa: S307 - whitelisted AST
