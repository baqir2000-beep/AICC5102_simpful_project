"""
fast_mamdani: a drop-in replacement for Simpful's Mamdani inference step.

Simpful (v2.12.0) evaluates EVERY rule's antecedent, then computes the centroid by
evaluating every rule's clipped output set at every grid point (cost = rules x
subdivisions) and approximates it with a Riemann sum (error ~ 1/subdivisions).

This module changes only the inference step, using five components (each can be
switched off for the ablation study):
  active : only the rules that can fire are evaluated. For a rule that is a plain
           AND (or product-AND) of one term per input, it can fire only if every one
           of its terms has non-zero membership. Rules are indexed by their terms
           once; at inference we list each input's non-zero terms and look up only
           those combinations. With triangular partitions at most 2 of 3 terms per
           input are non-zero, so at most 2^n of 3^n rules are touched. Rules with
           OR, NOT or a repeated variable are always evaluated normally. Skipped rules
           have firing strength exactly 0, so the result is unchanged.
           (Skipping non-active rules is known from fuzzy hardware, e.g. Chen et al.,
           "High-Speed Fuzzy Inference Processor Using Active Rules Identification",
           JCIS 2006; Simpful does not do it.)
  cache  : each input's memberships are computed once per inference (Simpful
           recomputes them for every clause of every rule) and AND/OR/NOT are applied
           directly. Same rule trees and operators as Simpful, so AND/OR/NOT, product
           operators and rule weights are supported unchanged.
  merge  : rules with the same consequent term are merged by max of their firing
           strengths (identical result under max aggregation), and rules that do not
           fire are dropped.
  exact  : for piecewise-linear output sets (triangular, trapezoidal, point-based) the
           centroid is integrated exactly between breakpoints, clip levels and line
           crossings (the same idea as PyFuzzy's polygon centroid). Non-linear output
           sets (Gaussian, sigmoid) fall back to a 200,001-point grid with the
           trapezoidal rule (not exact, but far finer than Simpful's default).
  vector : NumPy arrays instead of Python loops when a grid is used.
With exact=False the grid result equals Simpful's (checked in test_fast_mamdani.py).
Robustness option: strict=True raises an error for inputs outside the universe of
discourse and when no rule fires, instead of clamping / returning 0.
"""
import itertools
import math

import numpy as np
from simpful.fuzzy_sets import Triangular_MF, Trapezoidal_MF
from simpful.rule_parsing import Clause, Functional

_OPS = {"AND": min, "OR": max, "AND_p": lambda a, b: a * b, "OR_p": lambda a, b: a + b - a * b}
_trapz = getattr(np, "trapezoid", None) or np.trapz


class FuzzyInferenceError(Exception):
    pass


def _memberships(FS, v, mem):
    if v not in mem:
        mem[v] = FS._lvs[v].get_values(FS._variables[v])
    return mem[v]


def _eval_cached(node, FS, mem):
    if isinstance(node, Clause):
        return float(_memberships(FS, node._variable, mem)[node._term])
    if node._A == "":                      # unary operator (NOT)
        return 1.0 - _eval_cached(node._B, FS, mem)
    try:
        op = _OPS[node._fun]
    except KeyError:
        raise FuzzyInferenceError(f"unsupported operator {node._fun!r}") from None
    return op(_eval_cached(node._A, FS, mem), _eval_cached(node._B, FS, mem))


# ---------- active-rule index ----------
def _conjunction(node):
    """If node is an AND (or product-AND) of clauses, one per variable, return
    (operator, {variable: term}); otherwise None."""
    clauses, ops = {}, set()

    def walk(n):
        if isinstance(n, Clause):
            if n._variable in clauses:
                return False
            clauses[n._variable] = n._term
            return True
        if isinstance(n, Functional) and n._A != "" and n._fun in ("AND", "AND_p"):
            ops.add(n._fun)
            return walk(n._A) and walk(n._B)
        return False

    if not walk(node) or len(ops) > 1:
        return None
    return (ops.pop() if ops else "AND"), clauses


def _rule_index(FS):
    """Index rules by output -> (operator, variables) -> term tuple -> rule positions.
    Built once and rebuilt only if the rule list changes."""
    cached = FS.__dict__.get("_fast_mamdani_index")
    if cached is not None and cached[0] is FS._rules and cached[1] == len(FS._rules):
        return cached[2]
    index = {}
    for pos, (ant, cons) in enumerate(FS._rules):
        entry = index.setdefault(cons[0], {"groups": {}, "other": []})
        parsed = _conjunction(ant)
        if parsed is None:
            entry["other"].append(pos)
            continue
        op, clauses = parsed
        variables = tuple(sorted(clauses))
        key = tuple(clauses[v] for v in variables)
        entry["groups"].setdefault((op, variables), {}).setdefault(key, []).append(pos)
    FS._fast_mamdani_index = (FS._rules, len(FS._rules), index)
    return index


def _weight(cons):
    return float(cons[2]) if len(cons) > 2 else 1.0


def _active_firings(FS, output, mem, cache):
    """(term, firing * weight) for every rule that can fire; skipped rules are exactly 0."""
    entry = _rule_index(FS).get(output, {"groups": {}, "other": []})
    rules = FS._rules
    fired = []
    for (op, variables), table in entry["groups"].items():
        values = [_memberships(FS, v, mem) for v in variables]
        active = [[t for t, m in vals.items() if m > 0] for vals in values]
        n_combinations = math.prod(len(a) for a in active)
        if n_combinations == 0:
            continue
        if n_combinations <= len(table):          # enumerate the few active combinations
            keys = (k for k in itertools.product(*active) if k in table)
        else:                                      # sparse rule base: scan its rules instead
            keys = (k for k in table if all(vals.get(t, 0) > 0 for vals, t in zip(values, k)))
        for key in keys:
            degrees = [float(vals[t]) for vals, t in zip(values, key)]
            strength = min(degrees) if op == "AND" else math.prod(degrees)
            for pos in table[key]:
                ant, cons = rules[pos]
                firing = strength if cache else float(ant.evaluate(FS))
                fired.append((cons[1], firing * _weight(cons)))
    for pos in entry["other"]:
        ant, cons = rules[pos]
        firing = _eval_cached(ant, FS, mem) if cache else float(ant.evaluate(FS))
        fired.append((cons[1], firing * _weight(cons)))
    return fired


# ---------- helpers ----------
def _rules_for(FS, output):
    for ant, cons in FS._rules:
        if cons[0] == output:
            yield ant, cons[1], _weight(cons)


def _pl_vertices(fs, x0, x1):
    """Breakpoints and values of a piecewise-linear fuzzy set on [x0, x1], or None."""
    if fs._type == "pointbased":
        px = list(fs._points[:, 0])
    else:
        mf = fs._funpointer
        if isinstance(mf, Triangular_MF):
            px = [mf._a, mf._b, mf._c]
        elif isinstance(mf, Trapezoidal_MF):
            px = [mf._a, mf._b, mf._c, mf._d]
        else:
            return None
    bx = np.array(sorted({x0, x1, *[p for p in px if x0 < p < x1]}), float)
    by = np.array([float(fs.get_value(v)) for v in bx])
    return bx, by


def _eval_set(fs, x, x0, x1):
    """Vectorised membership of fuzzy set fs on array x."""
    v = _pl_vertices(fs, x0, x1)
    if v is not None:
        return np.interp(x, *v)
    f = fs._funpointer                       # Gaussian / sigmoid / custom function
    try:
        y = np.asarray(f._execute(x), dtype=float)
        if y.shape != x.shape:
            raise ValueError
    except Exception:                        # function only accepts scalars
        y = np.array([float(f._execute(float(u))) for u in x])
    return np.clip(y, 0.0, 1.0)


def _check_inputs(FS):
    for name, val in FS._variables.items():
        if name in FS._lvs:
            lo, hi = FS._lvs[name].get_universe_of_discourse()
            if not (lo <= val <= hi):
                raise FuzzyInferenceError(f"input {name}={val} is outside its universe of discourse [{lo}, {hi}]")


# ---------- exact centroid of max_k min(w_k, mu_k(x)) for piecewise-linear mu_k ----------
def _exact_centroid(sets, x0, x1):
    """sets: list of (w, bx, by). Returns (area, first moment)."""
    B = np.unique(np.concatenate([[x0, x1]] + [bx for _, bx, _ in sets]))
    agg = lambda x: np.max([np.minimum(w, np.interp(x, bx, by)) for w, bx, by in sets], axis=0)
    levels = [w for w, _, _ in sets]
    area = moment = 0.0
    for l, r in zip(B[:-1], B[1:]):
        if r <= l:
            continue
        # every set is linear on [l, r]: mu = p + q x
        lines = []
        for w, bx, by in sets:
            yl, yr = np.interp([l, r], bx, by)
            q = (yr - yl) / (r - l); lines.append((yl - q * l, q))
        cand = [l, r]
        for i, (p1, q1) in enumerate(lines):
            for lev in levels:                      # line meets a clip level
                if q1 != 0: cand.append((lev - p1) / q1)
            for p2, q2 in lines[i + 1:]:            # two lines cross
                if q1 != q2: cand.append((p2 - p1) / (q1 - q2))
        pts = np.unique(np.clip([c for c in cand if l <= c <= r], l, r))
        a, m, b = pts[:-1], (pts[:-1] + pts[1:]) / 2, pts[1:]
        fa, fm, fb = agg(a), agg(m), agg(b)
        h = b - a
        area += np.sum(h * (fa + 4 * fm + fb) / 6)
        moment += np.sum(h * (a * fa + 4 * m * fm + b * fb) / 6)
    return area, moment


# ---------- main entry point ----------
def fast_mamdani(FS, terms, subdivisions=1000, merge=True, vector=True, exact=True,
                 cache=True, active=True, strict=False):
    if strict:
        _check_inputs(FS)
    out = {}
    for output in terms:
        lv = FS._lvs[output]; x0, x1 = lv.get_universe_of_discourse()
        # 1) firing strengths
        mem = {}
        if active:
            fired = _active_firings(FS, output, mem, cache)
        elif cache:
            fired = [(term, _eval_cached(ant, FS, mem) * w) for ant, term, w in _rules_for(FS, output)]
        else:
            fired = [(term, float(ant.evaluate(FS)) * w) for ant, term, w in _rules_for(FS, output)]
        if merge:
            best = {}
            for term, v in fired:
                if v > 0: best[term] = max(best.get(term, 0.0), v)
            fired = list(best.items())
        if not any(v > 0 for _, v in fired):
            if strict:
                raise FuzzyInferenceError(f"no rule fires for output '{output}'")
            out[output] = 0.0; continue
        fsets = [(lv._FSlist[lv.get_index(t)], v) for t, v in fired]
        # 2) centroid
        verts = [_pl_vertices(fs, x0, x1) for fs, _ in fsets] if exact else None
        if exact and all(v is not None for v in verts):
            area, mom = _exact_centroid([(w, *v) for (_, w), v in zip(fsets, verts)], x0, x1)
            out[output] = mom / area
        elif vector and exact:   # non-linear output sets: fine grid, trapezoidal rule
            x = np.linspace(x0, x1, 200001)
            agg = np.max([np.minimum(w, _eval_set(fs, x, x0, x1)) for fs, w in fsets], axis=0)
            out[output] = float(_trapz(agg * x, x) / _trapz(agg, x))
        elif vector:             # Simpful's grid, same arithmetic (Riemann sum)
            x = np.linspace(x0, x1, subdivisions)
            agg = np.max([np.minimum(w, _eval_set(fs, x, x0, x1)) for fs, w in fsets], axis=0)
            out[output] = float((agg * x).sum() / agg.sum())
        else:                    # scalar loop, same arithmetic as Simpful
            num = den = 0.0
            for u in np.linspace(x0, x1, subdivisions):
                k = max(min(w, float(fs.get_value(u))) for fs, w in fsets)
                num += k * u; den += k
            out[output] = num / den
    return out
