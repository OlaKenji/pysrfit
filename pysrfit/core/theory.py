"""Evaluate a supported subset of musrfit THEORY + FUNCTIONS syntax in pysrfit."""
from __future__ import annotations
import ast
import operator
import re
import numpy as np
from .functions import REGISTRY

# Numerical muon gyromagnetic ratio in MHz/G, as used in standard muSR models.
CONSTANTS = {"gamma_mu": 0.0135538817, "pi": np.pi}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval_ast(node, names):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_ast(node.left, names), _eval_ast(node.right, names))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_ast(node.operand, names)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("Unsupported expression; allowed: numbers, parN, mapN, funN, +, -, *, /, **")


def auxiliary_values(text: str, parameters: dict[int, float], mapping: list[int] | None = None) -> dict[str, float]:
    names = {f"par{i}": v for i, v in parameters.items()}
    names.update(CONSTANTS)
    for i, target in enumerate(mapping or [], 1):
        if target:
            names[f"map{i}"] = parameters[target]
    definitions = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        match = re.fullmatch(r"(fun\d+)\s*=\s*(.+)", stripped, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unsupported FUNCTIONS line: {raw}")
        definitions.append((match.group(1).lower(), match.group(2)))
    # Forward references supported, cycles rejected.
    for _ in range(len(definitions) + 1):
        pending = []
        progress = False
        for key, expression in definitions:
            try:
                result = _eval_ast(ast.parse(expression, mode="eval"), names)
            except KeyError:
                pending.append((key, expression))
                continue
            names[key] = float(result)
            progress = True
        if not pending:
            return names
        if not progress:
            raise ValueError(f"Unknown or cyclic auxiliary functions: {[k for k, _ in pending]}")
        definitions = pending
    raise ValueError("Unresolved FUNCTIONS")


def evaluate_theory(theory: str, t_us, parameters: dict[int, float],
                    functions_text: str = "", mapping: list[int] | None = None) -> np.ndarray:
    t = np.asarray(t_us, dtype=float)
    names = auxiliary_values(functions_text, parameters, mapping)
    terms: list[list[tuple[str, list[str]]]] = [[]]
    for raw in theory.splitlines():
        line = raw.split("#", 1)[0].split("(", 1)[0].strip()
        if not line:
            continue
        if line == "+":
            if not terms[-1]:
                raise ValueError("Empty THEORY component before '+'")
            terms.append([])
            continue
        parts = line.split()
        key = parts[0].lower()
        if key not in REGISTRY:
            raise ValueError(f"Unknown THEORY function '{parts[0]}'; load custom code if needed")
        expected = REGISTRY[key].arity
        if len(parts) - 1 != expected:
            raise ValueError(f"{parts[0]} expects {expected} parameter references, got {len(parts)-1}")
        terms[-1].append((key, parts[1:]))
    if not terms[-1]:
        raise ValueError("THEORY is empty or ends with '+'")

    def value(token):
        lower = token.lower()
        if lower.startswith("fun") or lower.startswith("map"):
            if lower not in names:
                raise ValueError(f"Unknown reference: {token}")
            return names[lower]
        try:
            idx = int(token)
            return parameters[idx]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Unknown parameter reference: {token}") from exc

    total = np.zeros_like(t, dtype=float)
    for component in terms:
        subtotal = np.ones_like(t)
        for funcname, args in component:
            result = np.asarray(REGISTRY[funcname].function(t, *(value(x) for x in args)), dtype=float)
            if result.shape != t.shape:
                raise ValueError(f"Function {funcname} returned shape {result.shape}, expected {t.shape}")
            subtotal *= result
        total += subtotal
    if not np.all(np.isfinite(total)):
        raise ValueError("Theory produced nonfinite values")
    return total
