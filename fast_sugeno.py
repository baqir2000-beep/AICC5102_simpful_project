"""Fast, semantics-preserving zero-order Sugeno inference for Simpful 2.12.

This companion is deliberately separate from ``fast_mamdani``.  It evaluates
Sugeno rules with crisp consequents and returns the same weighted average used
by Simpful's ``Sugeno_inference``.  It does *not* defuzzify output fuzzy sets.

The implementation supports the operators used by Simpful's parsed rule trees
(AND, OR, product-AND/product-OR, NOT) and numeric rule weights.

Rule weights (weights=...):
  "simpful" (default): as in Simpful 2.12, the weight multiplies the numerator
      only, so a single rule with weight 0.5 that fires fully returns HALF its
      crisp value. Kept as the default so outputs equal Simpful's.
  "average": the corrected version. The weight multiplies the firing strength in
      both numerator and denominator, so the output is a true weighted average:
          y = sum(w_i * f_i * z_i) / sum(w_i * f_i)
      This is how Simpful's own Mamdani inference treats weights (the weight
      scales the firing strength), and it reduces to Simpful's result when every
      weight is 1. See findings_checks.py (F1) and test_fast_mamdani.py (T6).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Sequence

from simpful.rule_parsing import Clause, Functional


_OPS = {
    "AND": min,
    "OR": max,
    "AND_p": lambda left, right: left * right,
    "OR_p": lambda left, right: left + right - left * right,
}


class SugenoInferenceError(ValueError):
    """Raised when a requested Sugeno output cannot be evaluated."""


def _evaluate_cached(node, fuzzy_system, memberships: Dict[str, Dict[str, float]]) -> float:
    """Evaluate a Simpful parsed antecedent, caching each variable's terms."""
    if isinstance(node, Clause):
        variable = node._variable
        if variable not in memberships:
            try:
                memberships[variable] = fuzzy_system._lvs[variable].get_values(
                    fuzzy_system._variables[variable]
                )
            except KeyError as exc:
                raise SugenoInferenceError(
                    f"input variable {variable!r} is not defined or has no value"
                ) from exc
        try:
            return float(memberships[variable][node._term])
        except KeyError as exc:
            raise SugenoInferenceError(
                f"term {node._term!r} is not defined for {variable!r}"
            ) from exc

    if not isinstance(node, Functional):
        raise SugenoInferenceError(f"unsupported antecedent node {type(node)!r}")
    if node._A == "":
        return 1.0 - _evaluate_cached(node._B, fuzzy_system, memberships)
    try:
        operation = _OPS[node._fun]
    except KeyError as exc:
        raise SugenoInferenceError(f"unsupported operator {node._fun!r}") from exc
    return operation(
        _evaluate_cached(node._A, fuzzy_system, memberships),
        _evaluate_cached(node._B, fuzzy_system, memberships),
    )


def _outputs_to_evaluate(fuzzy_system, terms: Sequence[str] | None) -> Iterable[str]:
    available = {consequent[0] for _, consequent in fuzzy_system._rules}
    if terms is None:
        return available
    requested = set(terms)
    unknown = requested - available
    if unknown:
        raise SugenoInferenceError(
            f"requested output(s) do not occur in a consequent: {sorted(unknown)!r}"
        )
    return requested


def fast_sugeno(fuzzy_system, terms: Sequence[str] | None = None,
                weights: str = "simpful") -> dict[str, float]:
    """Infer zero-order Sugeno outputs with cached antecedent memberships.

    Only Simpful crisp output values are supported.  Functional consequents
    intentionally remain with Simpful because their expression language has
    additional template semantics not exercised by the paper's sepsis model.
    """
    if weights not in ("simpful", "average"):
        raise SugenoInferenceError(f"weights must be 'simpful' or 'average', not {weights!r}")
    outputs = _outputs_to_evaluate(fuzzy_system, terms)
    memberships: Dict[str, Dict[str, float]] = {}
    result: dict[str, float] = {}

    for output in outputs:
        numerator = 0.0
        denominator = 0.0
        for antecedent, consequent in fuzzy_system._rules:
            if consequent[0] != output:
                continue
            consequent_name = consequent[1]
            if consequent_name not in fuzzy_system._crispvalues:
                raise SugenoInferenceError(
                    "fast_sugeno supports crisp Sugeno consequents only; "
                    f"{consequent_name!r} is not a crisp output value"
                )
            weight = float(consequent[2]) if len(consequent) > 2 else 1.0
            firing = _evaluate_cached(antecedent, fuzzy_system, memberships)
            numerator += firing * float(fuzzy_system._crispvalues[consequent_name]) * weight
            denominator += firing * weight if weights == "average" else firing
        result[output] = 0.0 if denominator == 0.0 else numerator / denominator
    return result
