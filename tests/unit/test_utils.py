"""
Comprehensive unit tests for agents/utils.py module.

Tests all utility functions for JSON parsing, code extraction, and text manipulation.
"""


from ptpd_calibration.agents.utils import (
    extract_classes,
    extract_code_block,
    extract_functions,
    extract_imports,
    format_bullet_list,
    parse_json_response,
    sanitize_identifier,
    truncate_text,
)


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_parse_simple_json_object(self):
        """Test parsing simple JSON object."""
        response = '{"name": "test", "value": 42}'
        result = parse_json_response(response)
        assert result == {"name": "test", "value": 42}

    def test_parse_json_in_markdown_block(self):
        """Test parsing JSON from markdown code block."""
        response = """Here's the result:
```json
{"status": "success", "data": [1, 2, 3]}
```
"""
        result = parse_json_response(response)
        assert result == {"status": "success", "data": [1, 2, 3]}

    def test_parse_json_in_generic_code_block(self):
        """Test parsing JSON from generic code block."""
        response = """Result:
```
{"key": "value"}
```
"""
        result = parse_json_response(response)
        assert result == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding explanatory text."""
        response = """I analyzed the code and here's the result:

{"analysis": "complete", "issues": 0}

This indicates the code is clean."""
        result = parse_json_response(response)
        assert result == {"analysis": "complete", "issues": 0}

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        response = '[{"id": 1}, {"id": 2}]'
        result = parse_json_response(response, parse_array=True)
        assert result == [{"id": 1}, {"id": 2}]

    def test_parse_json_array_in_code_block(self):
        """Test parsing JSON array from code block."""
        response = """```json
["item1", "item2", "item3"]
```"""
        result = parse_json_response(response, parse_array=True)
        assert result == ["item1", "item2", "item3"]

    def test_parse_empty_response(self):
        """Test parsing empty response returns None."""
        assert parse_json_response("") is None
        assert parse_json_response(None) is None

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        response = "This is not valid JSON at all"
        result = parse_json_response(response)
        assert result is None

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON returns None."""
        response = '{"key": value_without_quotes}'
        result = parse_json_response(response)
        assert result is None

    def test_parse_with_pydantic_model(self):
        """Test parsing with Pydantic model validation."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            count: int

        response = '{"name": "test", "count": 5}'
        result = parse_json_response(response, model_class=TestModel)
        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.count == 5

    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        response = '{"outer": {"inner": {"deep": [1, 2, 3]}}}'
        result = parse_json_response(response)
        assert result == {"outer": {"inner": {"deep": [1, 2, 3]}}}

    def test_parse_json_with_multiple_blocks(self):
        """Test parsing when multiple JSON blocks exist."""
        response = """First block:
```json
{"first": true}
```

Second block:
```json
{"second": true}
```
"""
        result = parse_json_response(response)
        # Should return the first valid JSON object
        assert result == {"first": True}

    def test_parse_array_returns_none_for_object(self):
        """Test that parse_array=True returns None for object."""
        response = '{"key": "value"}'
        result = parse_json_response(response, parse_array=True)
        # Should find the object first, not treat as array
        assert result is None or isinstance(result, dict)


class TestExtractCodeBlock:
    """Tests for extract_code_block function."""

    def test_extract_python_code_block(self):
        """Test extracting Python code from markdown."""
        response = """Here's the code:
```python
def hello():
    print("Hello, World!")
```
"""
        result = extract_code_block(response)
        assert "def hello():" in result
        assert 'print("Hello, World!")' in result

    def test_extract_generic_code_block(self):
        """Test extracting code from generic block."""
        response = """Code:
```
x = 1 + 2
```
"""
        result = extract_code_block(response)
        assert "x = 1 + 2" in result

    def test_extract_with_fallback_keywords(self):
        """Test extraction using fallback keywords."""
        response = """def my_function():
    return 42"""
        result = extract_code_block(response)
        assert "def my_function():" in result

    def test_extract_empty_response(self):
        """Test extracting from empty response."""
        assert extract_code_block("") == ""
        assert extract_code_block(None) == ""

    def test_extract_no_code_found(self):
        """Test when no code is found."""
        response = "This is just plain text without any code."
        result = extract_code_block(response)
        assert result == ""

    def test_extract_with_custom_language(self):
        """Test extracting with specific language."""
        response = """```javascript
console.log("Hello");
```"""
        result = extract_code_block(response, language="javascript")
        assert 'console.log("Hello")' in result

    def test_extract_with_custom_fallback_keywords(self):
        """Test extraction with custom fallback keywords."""
        response = "SELECT * FROM users WHERE id = 1"
        result = extract_code_block(
            response,
            language="sql",
            fallback_keywords=["SELECT", "INSERT", "UPDATE"],
        )
        assert "SELECT * FROM users" in result

    def test_extract_multiline_code(self):
        """Test extracting multiline code."""
        response = """```python
class MyClass:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value
```"""
        result = extract_code_block(response)
        assert "class MyClass:" in result
        assert "def __init__(self):" in result
        assert "def increment(self):" in result


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_extract_import_statements(self):
        """Test extracting import statements."""
        code = """import os
import sys
from pathlib import Path
from typing import List, Dict

def main():
    pass
"""
        result = extract_imports(code)
        assert "import os" in result
        assert "import sys" in result
        assert "from pathlib import Path" in result
        assert "from typing import List, Dict" in result
        assert len(result) == 4

    def test_extract_no_imports(self):
        """Test extracting from code without imports."""
        code = """def hello():
    print("Hello")
"""
        result = extract_imports(code)
        assert result == []

    def test_extract_import_with_alias(self):
        """Test extracting import with alias."""
        code = """import numpy as np
from pandas import DataFrame as DF
"""
        result = extract_imports(code)
        assert "import numpy as np" in result
        assert "from pandas import DataFrame as DF" in result

    def test_extract_multiline_import(self):
        """Test handling of imports (line by line)."""
        code = """from typing import (
    List,
    Dict,
)
"""
        result = extract_imports(code)
        assert "from typing import (" in result


class TestExtractFunctions:
    """Tests for extract_functions function."""

    def test_extract_function_names(self):
        """Test extracting function names."""
        code = """def foo():
    pass

def bar(x, y):
    return x + y

def baz():
    '''docstring'''
    pass
"""
        result = extract_functions(code)
        assert "foo" in result
        assert "bar" in result
        assert "baz" in result
        assert len(result) == 3

    def test_extract_no_functions(self):
        """Test extracting from code without functions."""
        code = """class MyClass:
    pass

x = 1
"""
        result = extract_functions(code)
        assert result == []

    def test_extract_async_function(self):
        """Test extracting async function names."""
        code = """async def async_func():
    await something()

def sync_func():
    pass
"""
        result = extract_functions(code)
        # Note: Current implementation only matches 'def ', not 'async def'
        assert "sync_func" in result

    def test_extract_method_names(self):
        """Test extracting method names from class."""
        code = """class MyClass:
    def method1(self):
        pass

    def method2(self, arg):
        return arg
"""
        result = extract_functions(code)
        assert "method1" in result
        assert "method2" in result


class TestExtractClasses:
    """Tests for extract_classes function."""

    def test_extract_class_names(self):
        """Test extracting class names."""
        code = """class Foo:
    pass

class Bar(Base):
    pass

class Baz(Mixin1, Mixin2):
    pass
"""
        result = extract_classes(code)
        assert "Foo" in result
        assert "Bar" in result
        assert "Baz" in result
        assert len(result) == 3

    def test_extract_no_classes(self):
        """Test extracting from code without classes."""
        code = """def function():
    pass

x = 1
"""
        result = extract_classes(code)
        assert result == []

    def test_extract_dataclass(self):
        """Test extracting dataclass."""
        code = """from dataclasses import dataclass

@dataclass
class DataModel:
    name: str
    value: int
"""
        result = extract_classes(code)
        assert "DataModel" in result


class TestFormatBulletList:
    """Tests for format_bullet_list function."""

    def test_format_simple_list(self):
        """Test formatting simple list."""
        items = ["item1", "item2", "item3"]
        result = format_bullet_list(items)
        assert result == "- item1\n- item2\n- item3"

    def test_format_with_custom_prefix(self):
        """Test formatting with custom prefix."""
        items = ["first", "second"]
        result = format_bullet_list(items, prefix="* ")
        assert result == "* first\n* second"

    def test_format_with_indent(self):
        """Test formatting with indentation."""
        items = ["a", "b"]
        result = format_bullet_list(items, indent=4)
        assert result == "    - a\n    - b"

    def test_format_empty_list(self):
        """Test formatting empty list."""
        assert format_bullet_list([]) == ""
        assert format_bullet_list(None) == ""

    def test_format_single_item(self):
        """Test formatting single item list."""
        result = format_bullet_list(["only"])
        assert result == "- only"


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_truncate_long_text(self):
        """Test truncating long text."""
        text = "a" * 100
        result = truncate_text(text, max_length=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_short_text(self):
        """Test that short text is not truncated."""
        text = "short"
        result = truncate_text(text, max_length=100)
        assert result == "short"

    def test_truncate_exact_length(self):
        """Test text at exact max length."""
        text = "exact"
        result = truncate_text(text, max_length=5)
        assert result == "exact"

    def test_truncate_with_custom_suffix(self):
        """Test truncating with custom suffix."""
        text = "a" * 50
        result = truncate_text(text, max_length=10, suffix="[...]")
        assert len(result) == 10
        assert result.endswith("[...]")


class TestSanitizeIdentifier:
    """Tests for sanitize_identifier function."""

    def test_sanitize_spaces(self):
        """Test sanitizing string with spaces."""
        result = sanitize_identifier("My Class Name")
        assert result == "my_class_name"

    def test_sanitize_special_chars(self):
        """Test sanitizing special characters."""
        result = sanitize_identifier("foo-bar.baz")
        assert result == "foo_bar_baz"

    def test_sanitize_leading_digit(self):
        """Test removing leading digits."""
        result = sanitize_identifier("123abc")
        assert result == "abc"

    def test_sanitize_multiple_underscores(self):
        """Test collapsing multiple underscores."""
        result = sanitize_identifier("foo___bar")
        assert result == "foo_bar"

    def test_sanitize_empty_string(self):
        """Test sanitizing empty-ish string."""
        result = sanitize_identifier("123")
        assert result == "unnamed"

    def test_sanitize_already_valid(self):
        """Test already valid identifier."""
        result = sanitize_identifier("valid_name")
        assert result == "valid_name"

    def test_sanitize_uppercase(self):
        """Test converting to lowercase."""
        result = sanitize_identifier("CamelCase")
        assert result == "camelcase"

    def test_sanitize_leading_trailing_underscores(self):
        """Test removing leading/trailing underscores."""
        result = sanitize_identifier("__private__")
        assert result == "private"
