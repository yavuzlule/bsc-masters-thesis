from pydantic_ai import Agent
from bsc_relish.agentic.models import Chunk, ValidationResult
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

model = OpenAIChatModel(
    model_name="qwen2.5",
    provider=provider
)

chunker = Agent(
    name="DocumentChunkerAgent",
    output_type=list[Chunk],
    model=model,
    system_prompt="""
    You are a helpful assistant that breaks down a document into smaller chunks for easier processing. The document is provided as a string input. Your task is to split the document into coherent sections or paragraphs, ensuring that each chunk is self-contained and maintains the context of the original document.

    You must return a list of JSON objects, where each object represents a chunk of the document. Each JSON object should have the following structure:
    {
        "id": "string",  // A unique identifier for the chunk (e.g., "chunk_1", "chunk_2", etc.)
        "content": "string",  // The text content of the chunk
        "metadata": {},  // An optional dictionary for any additional information about the chunk (can be left empty)
        "origin": "string",  // The original text from which this chunk was derived (can be the same as content or a reference to the original document)
        "length": 0  // The length of the content in characters
    }

    """
)

@chunker.tool
def split_into_chunks(
    ctx: RunContext,
    document: str,
    max_chunk_size: int = 256
) -> list[Chunk]:

    paragraphs = document.split("\n\n")
    chunks = []
    chunk_id = 1

    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            chunks.append(Chunk(
                id=f"chunk_{chunk_id}",
                content=paragraph,
                metadata={},
                origin=paragraph,
                length=len(paragraph)
            ))
            chunk_id += 1
        else:
            # If the paragraph is too long, split it into smaller parts
            for i in range(0, len(paragraph), max_chunk_size):
                chunk_content = paragraph[i:i+max_chunk_size]
                chunks.append(Chunk(
                    id=f"chunk_{chunk_id}",
                    content=chunk_content,
                    metadata={},
                    origin=paragraph,
                    length=len(chunk_content)
                ))
                chunk_id += 1

    return chunks

