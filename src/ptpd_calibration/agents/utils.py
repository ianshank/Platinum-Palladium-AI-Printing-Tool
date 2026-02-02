"""
Shared utility functions for the agents module.

This module provides common utilities used across subagents to avoid
code duplication and ensure consistent behavior.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")


def parse_json_response(
    response: str,
    model_class: type[T] | None = None,
    parse_array: bool = False,
) -> dict | list | T | None:
    """
    Extract and parse JSON from an LLM response.

    Handles common patterns like JSON embedded in markdown code blocks
    or mixed with explanatory text.

    Args:
        response: Raw LLM response text.
        model_class: Optional Pydantic model class to parse into.
        parse_array: If True, parse as JSON array instead of object.

    Returns:
        Parsed JSON as dict/list, or Pydantic model instance if model_class provided.
        Returns None if parsing fails.

    Example:
        >>> response = "Here's the plan:\\n```json\\n{\"name\": \"test\"}\\n```"
        >>> parse_json_response(response)
        {'name': 'test'}
    """
    if not response:
        return None

    # Try to find JSON in code blocks first
    json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(json_block_pattern, response)
    if matches:
        for match in matches:
            try:
                data = json.loads(match.strip())
                if parse_array and isinstance(data, list):
                    return _validate_with_model(data, model_class)
                if not parse_array and isinstance(data, dict):
                    return _validate_with_model(data, model_class)
            except json.JSONDecodeError:
                continue

    # Fall back to finding raw JSON
    if parse_array:
        start = response.find("[")
        end = response.rfind("]") + 1
    else:
        start = response.find("{")
        end = response.rfind("}") + 1

    if start >= 0 and end > start:
        try:
            data = json.loads(response[start:end])
            return _validate_with_model(data, model_class)
        except json.JSONDecodeError:
            pass

    return None


def _validate_with_model(
    data: dict | list,
    model_class: type[T] | None,
) -> dict | list | T:
    """Validate data with Pydantic model if provided."""
    if model_class is not None and isinstance(data, dict):
        return model_class(**data)
    return data


def extract_code_block(
    response: str,
    language: str = "python",
    fallback_keywords: list[str] | None = None,
) -> str:
    """
    Extract a code block from an LLM response.

    Handles markdown code blocks with or without language specifiers.

    Args:
        response: Raw LLM response text.
        language: Expected language (e.g., "python", "javascript").
        fallback_keywords: Keywords to identify code without blocks.

    Returns:
        Extracted code as string, or empty string if not found.

    Example:
        >>> response = "Here's the code:\\n```python\\ndef foo():\\n    pass\\n```"
        >>> extract_code_block(response)
        'def foo():\\n    pass'
    """
    if not response:
        return ""

    # Default fallback keywords for Python
    if fallback_keywords is None:
        fallback_keywords = ["def ", "class ", "import ", "from "]

    # Try language-specific code block first
    lang_pattern = f"```{language}\\s*\\n([\\s\\S]*?)\\n?```"
    match = re.search(lang_pattern, response)
    if match:
        return match.group(1).strip()

    # Try generic code block
    generic_pattern = r"```\s*\n([\s\S]*?)\n?```"
    match = re.search(generic_pattern, response)
    if match:
        return match.group(1).strip()

    # Fall back to keyword detection
    if any(kw in response for kw in fallback_keywords):
        return response.strip()

    return ""


def extract_imports(code: str) -> list[str]:
    """
    Extract import statements from Python code.

    Args:
        code: Python source code.

    Returns:
        List of import statement strings.

    Example:
        >>> code = "import os\\nfrom pathlib import Path\\n\\ndef foo(): pass"
        >>> extract_imports(code)
        ['import os', 'from pathlib import Path']
    """
    imports = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    return imports


def extract_functions(code: str) -> list[str]:
    """
    Extract function names from Python code.

    Args:
        code: Python source code.

    Returns:
        List of function names.

    Example:
        >>> code = "def foo(): pass\\ndef bar(): pass"
        >>> extract_functions(code)
        ['foo', 'bar']
    """
    pattern = r"def\s+(\w+)\s*\("
    return re.findall(pattern, code)


def extract_classes(code: str) -> list[str]:
    """
    Extract class names from Python code.

    Args:
        code: Python source code.

    Returns:
        List of class names.

    Example:
        >>> code = "class Foo: pass\\nclass Bar: pass"
        >>> extract_classes(code)
        ['Foo', 'Bar']
    """
    pattern = r"\bclass\s+(\w+)[\s:(]"
    return re.findall(pattern, code)


def format_bullet_list(
    items: list[str] | None,
    prefix: str = "- ",
    indent: int = 0,
) -> str:
    """
    Format a list of items as bullet points.

    Args:
        items: List of strings to format.
        prefix: Prefix for each item (default: "- ").
        indent: Number of spaces to indent each line.

    Returns:
        Formatted string with bullet points.

    Example:
        >>> format_bullet_list(["item1", "item2"])
        '- item1\\n- item2'
    """
    if not items:
        return ""
    indent_str = " " * indent
    return "\n".join(f"{indent_str}{prefix}{item}" for item in items)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to append when truncating.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_identifier(name: str) -> str:
    """
    Convert a string to a valid Python identifier.

    Args:
        name: String to sanitize.

    Returns:
        Valid Python identifier.

    Example:
        >>> sanitize_identifier("My Class Name")
        'my_class_name'
    """
    # Replace non-alphanumeric with underscore
    result = re.sub(r"[^\w]", "_", name.lower())
    # Remove leading digits
    result = re.sub(r"^\d+", "", result)
    # Remove multiple underscores
    result = re.sub(r"_+", "_", result)
    # Remove leading/trailing underscores
    return result.strip("_") or "unnamed"
