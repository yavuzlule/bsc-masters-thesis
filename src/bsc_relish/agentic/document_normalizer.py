from bsc_relish.agentic.models import DocumentInput
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from bsc_relish.agentic.models import NormalizedDocument
from pydantic_ai import RunContext


provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

model = OpenAIChatModel(
    model_name="qwen2.5",
    provider=provider,
    
)

document_normalizer = Agent(
    name="DocumentNormalizerAgent",
    output_type=NormalizedDocument,
    model=model,
    retries=3,
    system_prompt="""
        You MUST return valid JSON matching this schema:

        {
        "content": "string",
        "source_type": "string",
        "metadata": {}
        }

        Rules:
        - Output must be valid JSON only
        - No markdown
        - No explanations
        - No extra text

        
        based on the given text.
        

        the content of the JSON should be the EXACT same text as string WITHOUT any changes to the original.
        """

        
        )

from pathlib import Path
from pypdf import PdfReader
import re


@document_normalizer.tool
def read_txt(ctx: RunContext, path: str) -> str:
    return Path(path).read_text(
        encoding="utf-8",
        errors="ignore"
    )


@document_normalizer.tool
def read_pdf(ctx: RunContext, path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


@document_normalizer.tool
def normalize_whitespace(ctx: RunContext, text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()