"""Band arithmetic for agents: evaluate a whitelisted expression over named band arrays, and
lay out the text table `mpc read` prints. No predefined indices live here; the agent writes the
formula it wants."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable

import numpy as np

EPS = 1e-6


def _div(a, b):
    b = np.asarray(b, dtype=np.float64)
    small = np.abs(b) < EPS
    return np.where(small, np.nan, np.asarray(a, dtype=np.float64) / np.where(small, 1.0, b))


_BINOPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _div,
    ast.Pow: operator.pow,
}
_FUNCS: dict[str, Callable] = {"abs": np.abs, "sqrt": np.sqrt, "log": np.log, "exp": np.exp}
_CMPS: dict[type, Callable] = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}


def _compare(node: ast.Compare, bands) -> np.ndarray:
    """1 where it holds, 0 where not, NaN where a side has no data; `a < b < c` chains."""
    out, left = 1.0, _eval(node.left, bands)
    for op, right_node in zip(node.ops, node.comparators):
        right = _eval(right_node, bands)
        held = _CMPS[type(op)](left, right).astype(np.float64)
        out = out * np.where(np.isnan(left) | np.isnan(right), np.nan, held)
        left = right
    return out


def _eval(node: ast.AST, bands: dict[str, np.ndarray]):
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in bands:
            raise ValueError(f"unknown name {node.id!r}; bands are {', '.join(bands)}")
        return np.asarray(bands[node.id], dtype=np.float64)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _eval(node.operand, bands)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, bands), _eval(node.right, bands))
    if isinstance(node, ast.Compare) and all(type(op) in _CMPS for op in node.ops):
        return _compare(node, bands)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id not in _FUNCS or len(node.args) != 1:
            allowed = ", ".join(_FUNCS)
            raise ValueError(f"unknown function {node.func.id!r}; only {allowed} of one argument")
        return _FUNCS[node.func.id](_eval(node.args[0], bands))
    raise ValueError(
        f"unsupported syntax {type(node).__name__}; use + - * / ** ( ), == != < <= > >= and bands"
    )


def aliased(expr: str, known) -> tuple[str, dict[str, str]]:
    """`expr` with each known name that is not a Python name (`250m_16_days_EVI`,
    `burn-severity`) swapped for an alias, and the aliases."""
    alias = {}
    for i, n in enumerate(
        sorted((n for n in known if not n.isidentifier()), key=len, reverse=True)
    ):
        new = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", f"_asset{i}", expr)
        if new != expr:
            expr, alias[f"_asset{i}"] = new, n
    return expr, alias


def names(expr: str, known=()) -> set[str]:
    """The band names `expr` uses (function names aside); none when it does not parse."""
    expr, alias = aliased(expr, known)
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return set()
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in funcs}
    return {alias.get(n, n) for n in used}


def evaluate(expr: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate `expr` over band arrays of one shape. Division by ~0 gives NaN."""
    text, alias = aliased(expr, bands)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expr!r}: {exc.msg}") from exc
    with np.errstate(all="ignore"):
        out = _eval(tree.body, {**bands, **{a: bands[n] for a, n in alias.items()}})
    shape = next(iter(bands.values())).shape
    return np.broadcast_to(np.asarray(out, dtype=np.float64), shape).astype(np.float32)


def fmt_value(val: float) -> str:
    """'-' for NaN; else places by magnitude (0 at 100 and above, 1 from 10, 2 from 1, 3 below),
    so dB prints with one decimal and reflectance with three."""
    if not np.isfinite(val):
        return "-"
    d = 0 if abs(val) >= 100 else 1 if abs(val) >= 10 else 2 if abs(val) >= 1 else 3
    return f"{val:.{d}f}"


def series_table(
    values: dict[str, np.ndarray], dates: list[str] | None, fmt: Callable[[str, float], str]
) -> str:
    """Named columns over the selected rows as the text table an agent reads. `dates` gives
    each row's date (None for static data)."""
    cols = list(values)
    n = len(next(iter(values.values())))
    width = max(6, *(len(c) for c in cols))
    tw = max(4, *(len(s) for s in dates)) if dates else 0
    head = f"{'date':<{tw}}  " if dates else ""
    head += " ".join(f"{c:>{width}}" for c in cols)
    lines = [head.rstrip()]
    for i in range(n):
        row = f"{dates[i]:<{tw}}  " if dates else ""
        row += " ".join(fmt(c, float(values[c][i])).rjust(width) for c in cols)
        lines.append(row.rstrip())
    return "\n".join(lines)
