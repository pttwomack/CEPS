"""
Provides utilities for manipulating symbolic group elements, represented as strings.
"""


def reduce(class_str: str) -> str:
    """Reduces a symbolic holonomy class string by cancelling adjacent inverse generators."""
    stack = []
    for char in class_str:
        if not stack:
            stack.append(char)
        elif char.swapcase() == stack[-1]:
            stack.pop()
        else:
            stack.append(char)
    return "".join(stack)
