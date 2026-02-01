"""
Coder subagent for code generation and implementation.

Generates Python code following project conventions and best practices.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from ptpd_calibration.agents.subagents.base import (
    BaseSubagent,
    SubagentCapability,
    SubagentConfig,
    SubagentResult,
    register_subagent,
)


class CodeFile(BaseModel):
    """Generated code file."""

    filename: str
    filepath: str
    content: str
    language: str = "python"
    imports: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)


class CodeChange(BaseModel):
    """A code change or modification."""

    file_path: str
    change_type: str  # create, modify, delete
    description: str
    old_content: str | None = None
    new_content: str | None = None
    line_range: tuple[int, int] | None = None


class ImplementationResult(BaseModel):
    """Result of code implementation."""

    success: bool
    files: list[CodeFile] = Field(default_factory=list)
    changes: list[CodeChange] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


CODER_SYSTEM_PROMPT = """You are an expert Python developer specializing in clean, maintainable code.

Your role is to generate high-quality Python code following these principles:

Code Quality:
1. Type Hints: Required for all function arguments and return values
2. Docstrings: Google-style docstrings for modules, classes, and functions
3. Error Handling: Explicit error handling, avoid bare try-except
4. Line Length: 100 characters max
5. Import Organization: stdlib, third-party, local (separated by blank lines)

Architecture Principles:
1. No Hardcoding: Use configs, env vars, dependency injection
2. Single Responsibility: Each class/function does one thing well
3. Backward Compatible: Don't break existing interfaces
4. Testable: Design for easy testing with dependency injection
5. Modular: Loosely coupled, reusable components

Patterns to Follow:
- Use Pydantic for data models
- Use dataclasses for simple data containers
- Use async/await for I/O operations
- Use logging instead of print statements
- Use pathlib.Path instead of os.path

Code Style:
- Use descriptive variable names
- Keep functions small and focused
- Prefer composition over inheritance
- Use explicit imports (avoid `import *`)
- Include meaningful comments only where logic is not self-evident

Output Format:
When generating code, output complete, runnable Python files.
Include all necessary imports at the top.
Use markdown code blocks with language identifier.
"""


@register_subagent
class CoderAgent(BaseSubagent):
    """
    Coder subagent for code generation.

    Generates Python code following project conventions,
    with proper type hints, docstrings, and error handling.
    """

    AGENT_TYPE = "coder"
    CAPABILITIES = [SubagentCapability.CODING, SubagentCapability.ANALYSIS]
    DESCRIPTION = "Generates Python code following project conventions"

    def __init__(self, config: SubagentConfig | None = None):
        """Initialize the coder agent."""
        super().__init__(config)
        self._generated_files: dict[str, CodeFile] = {}

    def capabilities(self) -> list[SubagentCapability]:
        """Return coder capabilities."""
        return self.CAPABILITIES

    async def run(self, task: str, context: dict | None = None) -> SubagentResult:
        """
        Generate code for the given task.

        Args:
            task: Description of what to implement.
            context: Optional context including:
                - specification: Detailed specification
                - existing_code: Code to modify or extend
                - patterns: Patterns to follow
                - constraints: Technical constraints

        Returns:
            SubagentResult containing generated code.
        """
        self._start_execution(task)

        try:
            ctx = context or {}

            if ctx.get("existing_code"):
                # Modify existing code
                result_data = await self.modify_code(
                    existing_code=ctx["existing_code"],
                    modification=task,
                    constraints=ctx.get("constraints"),
                )
            else:
                # Generate new code
                result_data = await self.generate_code(
                    specification=task,
                    patterns=ctx.get("patterns"),
                    constraints=ctx.get("constraints"),
                )

            result = SubagentResult(
                success=result_data.success,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                result=result_data.model_dump(),
                metadata={
                    "num_files": len(result_data.files),
                    "num_changes": len(result_data.changes),
                },
                artifacts=[
                    {
                        "type": "code_file",
                        "filename": f.filename,
                        "filepath": f.filepath,
                        "content": f.content,
                    }
                    for f in result_data.files
                ],
            )

        except Exception as e:
            result = SubagentResult(
                success=False,
                agent_id=self.id,
                agent_type=self.AGENT_TYPE,
                task=task,
                error=str(e),
            )

        return self._complete_execution(result)

    async def generate_code(
        self,
        specification: str,
        patterns: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> ImplementationResult:
        """
        Generate new code from specification.

        Args:
            specification: What to implement.
            patterns: Patterns to follow.
            constraints: Technical constraints.

        Returns:
            ImplementationResult with generated files.
        """
        patterns_text = "\n".join(f"- {p}" for p in (patterns or []))
        constraints_text = "\n".join(f"- {c}" for c in (constraints or []))

        prompt = f"""Generate Python code for the following specification:

{specification}

{"Patterns to follow:" + chr(10) + patterns_text if patterns else ""}

{"Constraints:" + chr(10) + constraints_text if constraints else ""}

Requirements:
1. Include all necessary imports
2. Add type hints to all functions
3. Include Google-style docstrings
4. Handle errors appropriately
5. Follow the code style guidelines

Generate complete, runnable Python code.
"""

        response = await self._llm_complete(prompt, system=CODER_SYSTEM_PROMPT)

        # Extract code from response
        code_content = self._extract_code(response)

        if not code_content:
            return ImplementationResult(
                success=False,
                notes=["Failed to generate valid code"],
            )

        # Create code file
        code_file = CodeFile(
            filename="generated_code.py",
            filepath="src/generated_code.py",
            content=code_content,
            imports=self._extract_imports(code_content),
            classes=self._extract_classes(code_content),
            functions=self._extract_functions(code_content),
        )

        self._generated_files[code_file.filename] = code_file

        return ImplementationResult(
            success=True,
            files=[code_file],
            notes=["Code generated successfully"],
        )

    async def modify_code(
        self,
        existing_code: str,
        modification: str,
        constraints: list[str] | None = None,
    ) -> ImplementationResult:
        """
        Modify existing code.

        Args:
            existing_code: Current code to modify.
            modification: Description of changes to make.
            constraints: Technical constraints.

        Returns:
            ImplementationResult with changes.
        """
        constraints_text = "\n".join(f"- {c}" for c in (constraints or []))

        prompt = f"""Modify the following code according to the specification:

Current Code:
```python
{existing_code}
```

Modification Required:
{modification}

{"Constraints:" + chr(10) + constraints_text if constraints else ""}

Requirements:
1. Maintain backward compatibility
2. Keep existing functionality intact
3. Add type hints if missing
4. Follow the code style guidelines
5. Explain the changes made

Output the complete modified code.
"""

        response = await self._llm_complete(prompt, system=CODER_SYSTEM_PROMPT)

        # Extract code from response
        new_code = self._extract_code(response)

        if not new_code:
            return ImplementationResult(
                success=False,
                notes=["Failed to generate valid modified code"],
            )

        # Create change record
        change = CodeChange(
            file_path="modified_code.py",
            change_type="modify",
            description=modification,
            old_content=existing_code,
            new_content=new_code,
        )

        code_file = CodeFile(
            filename="modified_code.py",
            filepath="src/modified_code.py",
            content=new_code,
            imports=self._extract_imports(new_code),
            classes=self._extract_classes(new_code),
            functions=self._extract_functions(new_code),
        )

        return ImplementationResult(
            success=True,
            files=[code_file],
            changes=[change],
            notes=["Code modified successfully"],
        )

    async def implement_function(
        self,
        name: str,
        description: str,
        parameters: list[dict],
        return_type: str,
        examples: list[str] | None = None,
    ) -> str:
        """
        Generate a single function.

        Args:
            name: Function name.
            description: What the function does.
            parameters: List of parameter dicts with name, type, description.
            return_type: Return type annotation.
            examples: Usage examples.

        Returns:
            Generated function code.
        """
        params_text = "\n".join(
            f"    - {p['name']} ({p['type']}): {p.get('description', '')}"
            for p in parameters
        )
        examples_text = "\n".join(f">>> {e}" for e in (examples or []))

        prompt = f"""Generate a Python function with the following specification:

Name: {name}
Description: {description}

Parameters:
{params_text}

Returns: {return_type}

{"Examples:" + chr(10) + examples_text if examples else ""}

Include:
- Type hints
- Google-style docstring
- Proper error handling
- Clear implementation

Output only the function code (no class wrapper).
"""

        response = await self._llm_complete(prompt, system=CODER_SYSTEM_PROMPT)
        return self._extract_code(response)

    async def implement_class(
        self,
        name: str,
        description: str,
        attributes: list[dict],
        methods: list[dict],
        base_classes: list[str] | None = None,
    ) -> str:
        """
        Generate a class implementation.

        Args:
            name: Class name.
            description: Class purpose.
            attributes: List of attribute dicts.
            methods: List of method specifications.
            base_classes: Parent classes to inherit from.

        Returns:
            Generated class code.
        """
        attrs_text = "\n".join(
            f"    - {a['name']} ({a['type']}): {a.get('description', '')}"
            for a in attributes
        )
        methods_text = "\n".join(
            f"    - {m['name']}({', '.join(m.get('params', []))}): {m.get('description', '')}"
            for m in methods
        )
        bases_text = f"({', '.join(base_classes)})" if base_classes else ""

        prompt = f"""Generate a Python class with the following specification:

Name: {name}{bases_text}
Description: {description}

Attributes:
{attrs_text}

Methods:
{methods_text}

Include:
- Type hints for all attributes and methods
- Google-style docstrings
- __init__ with proper initialization
- Property decorators where appropriate

Output the complete class code.
"""

        response = await self._llm_complete(prompt, system=CODER_SYSTEM_PROMPT)
        return self._extract_code(response)

    async def refactor_code(
        self,
        code: str,
        improvements: list[str],
    ) -> ImplementationResult:
        """
        Refactor code with specified improvements.

        Args:
            code: Code to refactor.
            improvements: List of improvements to make.

        Returns:
            ImplementationResult with refactored code.
        """
        improvements_text = "\n".join(f"- {i}" for i in improvements)

        prompt = f"""Refactor the following code with these improvements:

Code:
```python
{code}
```

Improvements:
{improvements_text}

Requirements:
1. Maintain the same functionality
2. Keep backward compatibility
3. Improve code quality
4. Add missing type hints and docstrings

Output the refactored code.
"""

        response = await self._llm_complete(prompt, system=CODER_SYSTEM_PROMPT)
        new_code = self._extract_code(response)

        if not new_code:
            return ImplementationResult(
                success=False,
                notes=["Failed to refactor code"],
            )

        return ImplementationResult(
            success=True,
            files=[
                CodeFile(
                    filename="refactored_code.py",
                    filepath="src/refactored_code.py",
                    content=new_code,
                    imports=self._extract_imports(new_code),
                    classes=self._extract_classes(new_code),
                    functions=self._extract_functions(new_code),
                )
            ],
            changes=[
                CodeChange(
                    file_path="refactored_code.py",
                    change_type="modify",
                    description="Refactored: " + ", ".join(improvements),
                    old_content=code,
                    new_content=new_code,
                )
            ],
            notes=["Code refactored successfully"],
        )

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Check if response is valid Python
        if any(kw in response for kw in ["def ", "class ", "import "]):
            return response.strip()

        return ""

    def _extract_imports(self, code: str) -> list[str]:
        """Extract import statements."""
        imports = []
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line)
        return imports

    def _extract_classes(self, code: str) -> list[str]:
        """Extract class names."""
        classes = []
        for line in code.split("\n"):
            if line.startswith("class "):
                name = line.split("class ")[1].split("(")[0].split(":")[0].strip()
                classes.append(name)
        return classes

    def _extract_functions(self, code: str) -> list[str]:
        """Extract function names."""
        functions = []
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("def "):
                name = stripped.split("def ")[1].split("(")[0].strip()
                functions.append(name)
        return functions

    def get_generated_file(self, filename: str) -> CodeFile | None:
        """Get a previously generated file."""
        return self._generated_files.get(filename)
