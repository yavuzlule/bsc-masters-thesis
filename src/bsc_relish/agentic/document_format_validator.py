from pydantic_ai import Agent
from bsc_relish.agentic.models import ValidationResult
import re
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

model = OpenAIChatModel(
    model_name="qwen2.5", # try small gemma
    provider=provider
)

document_validator = Agent(
    name="DocumentFormatValidatorAgent",
    output_type=ValidationResult,
    retries=2,
    model=model,
    system_prompt="""
        You validate a normalized document. Use each tool to validate.

        Return ValidationResult only. An example schema that you MUST follow:
        {
            "valid": true,
            "errors": [],
            "warnings": []
        }

        Rules:
        - content must exist and be non-empty
        - no binary characters (\x00)
        - whitespace should be normalized
        """
)



@document_validator.tool
def validate_content(
    ctx: RunContext,
    content: str
) -> ValidationResult:

    errors = []
    warnings = []

    if not content.strip():
        errors.append("Document content is empty")

    if "\x00" in content:
        errors.append("Binary data detected")

    if re.search(r"\n{5,}", content):
        warnings.append("Excessive newlines detected")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }