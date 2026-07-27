"""YAML prompt-template loading with validation and path safety."""

from pathlib import Path

import yaml

from app.utils.errors import GenerationError


class PromptLoader:
    """Load named YAML templates from the package prompt template directory."""

    _required_fields = {"system_instructions", "response_rules", "citation_instructions"}

    def __init__(self, template_directory: Path | None = None) -> None:
        self._template_directory = template_directory or Path(__file__).parent / "prompt_templates"

    def load(self, template_name: str) -> dict[str, str]:
        """Return a validated YAML template by its simple name."""
        if not template_name or Path(template_name).name != template_name:
            raise GenerationError("Prompt template name must not contain a path")
        template_path = self._template_directory / f"{template_name}.yaml"
        if not template_path.is_file():
            raise GenerationError(f"Prompt template is missing: {template_path}")
        try:
            contents = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise GenerationError(f"Unable to load prompt template '{template_name}': {error}") from error
        if not isinstance(contents, dict) or not self._required_fields.issubset(contents):
            raise GenerationError(f"Prompt template '{template_name}' is invalid")
        if not all(isinstance(contents[field], str) and contents[field].strip() for field in self._required_fields):
            raise GenerationError(f"Prompt template '{template_name}' contains empty instructions")
        return {field: str(value) for field, value in contents.items()}
