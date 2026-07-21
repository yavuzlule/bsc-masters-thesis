from groq import BaseModel
from pydantic import ConfigDict, field_validator
from bsc_relish.agentic.models import DocumentInput, BoundaryExtractionResult
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from bsc_relish.agentic.models import NormalizedDocument
from pydantic_ai import RunContext
from pydantic import BaseModel, ConfigDict, field_validator


provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

model = OpenAIChatModel(
    model_name="qwen2.5",
    provider=provider,
)

class TitleOutput(BaseModel):
    title: str

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    @field_validator("title")
    @classmethod
    def validate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("title must be a string")

        v = v.strip()

        if not v:
            raise ValueError("title cannot be empty")

        return v


title_extractor = Agent(
    name="TitleExtractor",
    output_type=TitleOutput,
    model=model,
    retries=10,
    system_prompt="""
Extract the FIRST title from the input text.

Return rules:
- Return only the first meaningful title segment.
- Preserve original wording exactly.
- Do not include any recipe or repeated trailing content if identifiable.
- If no boundary return empty string.
- Output must be valid JSON:
  {"title": "<title>"}
"""
)
