"""Tool 2 — calculator: exact arithmetic for membership and personal-training costs.

Small models get multi-step arithmetic wrong ("12 months of Elite plus a 10-pack"), so the LLM
writes the expression and this tool evaluates it. It parses the expression into a Python AST and
evaluates only numbers and arithmetic operators — never eval(), so no code can run.
"""
from __future__ import annotations

import ast
import asyncio
import operator
import re

from backend.tools.base import Tool, ToolContext, ToolError

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPRESSION_CHARS = 200


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ToolError("Exponent too large.")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round" and not node.keywords:
        args = [_eval(a) for a in node.args]
        if len(args) in (1, 2):
            return round(args[0], int(args[1])) if len(args) == 2 else round(args[0])
    raise ToolError("Only numbers, + - * / // % ** ( ) and round() are allowed.")


def evaluate(expression: str) -> float:
    expression = expression.replace("$", "").replace("×", "*")
    expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression)  # "1,000" -> "1000" (but keep round(x, 1))
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression).strip()  # "79 x 12" -> "79*12"
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError("The expression is empty or too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"'{expression}' is not a valid arithmetic expression.") from e
    try:
        return _eval(tree)
    except ZeroDivisionError as e:
        raise ToolError("Division by zero.") from e


async def run(args: dict, ctx: ToolContext) -> dict:
    value = await asyncio.to_thread(evaluate, args["expression"])
    value = round(value, 2) if isinstance(value, float) else value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"expression": args["expression"], "result": value}


TOOL = Tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression exactly. Use it for any maths on IronPulse prices: yearly cost, "
        "several months, freeze fees, prorated upgrades, personal-training packs, per-session prices."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic using numbers, + - * / ( ) and round(), e.g. \"79*12 + 460\"",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    run=run,
    timeout_s=2.0,
    examples=['{"name": "calculator", "arguments": {"expression": "79*12 + 460"}}'],
)
