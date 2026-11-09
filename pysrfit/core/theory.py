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
    terms: list[list[tuple]] = [[]]

    def parse_static_product(text: str, state_number: int):
        product = []
        for expression in text.split("*"):
            parts = expression.strip().split()
            if not parts:
                raise ValueError(
                    f"twoState state {state_number} contains an empty function")
            key = parts[0].lower()
            if key not in REGISTRY:
                raise ValueError(
                    f"Unknown twoState static function '{parts[0]}'")
            spec = REGISTRY[key]
            if spec.composite:
                raise ValueError("Nested composite THEORY functions are not supported")
            if len(parts) - 1 != spec.arity:
                raise ValueError(
                    f"{parts[0]} expects {spec.arity} parameter references, "
                    f"got {len(parts) - 1}")
            product.append((key, parts[1:]))
        return product

    for raw in theory.splitlines():
        line = raw.split("#", 1)[0].split("(", 1)[0].strip()
        if not line:
            continue
        if line == "+":
            if not terms[-1]:
                raise ValueError("Empty THEORY component before '+'")
            terms.append([])
            continue
        segments = [segment.strip() for segment in line.split("|")]
        parts = segments[0].split()
        key = parts[0].lower()
        if key not in REGISTRY:
            raise ValueError(f"Unknown THEORY function '{parts[0]}'; load custom code if needed")
        spec = REGISTRY[key]
        expected = spec.arity
        if len(parts) - 1 != expected:
            raise ValueError(f"{parts[0]} expects {expected} parameter references, got {len(parts)-1}")
        if spec.composite:
            if key != "twostate":
                raise ValueError(f"Unsupported composite THEORY function '{parts[0]}'")
            if len(segments) != 3:
                raise ValueError(
                    "twoState syntax is: twoState nu1 nu2 g | "
                    "staticFunction1 ... | staticFunction2 ...")
            states = (
                parse_static_product(segments[1], 1),
                parse_static_product(segments[2], 2),
            )
            terms[-1].append((key, parts[1:], states))
        else:
            if len(segments) != 1:
                raise ValueError("'|' is only valid in a twoState THEORY line")
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
        for entry in component:
            funcname, args = entry[:2]
            spec = REGISTRY[funcname]
            if spec.composite:
                state_products = entry[2]

                def static_product(times, product):
                    times = np.asarray(times, dtype=float)
                    product_value = np.ones_like(times)
                    for static_name, static_args in product:
                        static_result = np.asarray(
                            REGISTRY[static_name].function(
                                times, *(value(x) for x in static_args)),
                            dtype=float,
                        )
                        if static_result.shape != times.shape:
                            raise ValueError(
                                f"Function {static_name} returned shape "
                                f"{static_result.shape}, expected {times.shape}")
                        product_value *= static_result
                    return product_value

                result = np.asarray(
                    spec.function(
                        t,
                        *(value(x) for x in args),
                        lambda times, product=state_products[0]: static_product(times, product),
                        lambda times, product=state_products[1]: static_product(times, product),
                    ),
                    dtype=float,
                )
            else:
                result = np.asarray(
                    spec.function(t, *(value(x) for x in args)), dtype=float)
            if result.shape != t.shape:
                raise ValueError(f"Function {funcname} returned shape {result.shape}, expected {t.shape}")
            subtotal *= result
        total += subtotal
    if not np.all(np.isfinite(total)):
        raise ValueError("Theory produced nonfinite values")
    return total
