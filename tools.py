import ast
import operator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> str:
    """Safely evaluates an arithmetic expression, e.g. '(4.6 - 1.4) / 4.6 * 100'."""

    def ev(node):
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported expression")

    return str(ev(ast.parse(expression, mode="eval").body))