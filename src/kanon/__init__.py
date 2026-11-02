"""kanon: policy-adherent decisions for AI agents.

    policy = kanon.load_policy("policy.yaml")
    oracle = kanon.route(policy, llm=kanon.ollama_oracle("qwen2.5:14b"))
    outcome = kanon.solve(policy, case, oracle)
    print(kanon.explain(outcome))
"""

from .lint import Report, lint
from .oracle import (
    Oracle,
    Question,
    Verdict,
    anthropic_oracle,
    cli_oracle,
    code_oracle,
    noisy_oracle,
    ollama_oracle,
    route,
    table_oracle,
)
from .policy import ESCALATE, Decision, Fact, Policy, load_policy
from .solver import Coverage, Execution, Outcome, Resolution, apply, next_questions, solve
from .trace import explain, requirements, to_dict, to_json

__all__ = [
    "ESCALATE",
    "Coverage",
    "Decision",
    "Execution",
    "Fact",
    "Oracle",
    "Outcome",
    "Policy",
    "Question",
    "Report",
    "Resolution",
    "Verdict",
    "anthropic_oracle",
    "apply",
    "cli_oracle",
    "code_oracle",
    "explain",
    "lint",
    "load_policy",
    "next_questions",
    "noisy_oracle",
    "ollama_oracle",
    "route",
    "requirements",
    "solve",
    "table_oracle",
    "to_dict",
    "to_json",
]

__version__ = "0.1.0"
