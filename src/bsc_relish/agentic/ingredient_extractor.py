from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from bsc_relish.agentic.models import NormalizedDocument, IngredientsList
from typing import List

provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

class IngredientOutput(BaseModel):
    ingredients: List[str]


model = OpenAIChatModel(
    model_name="qwen2.5",  # or qwen2.5-7b-instruct if needed
    provider=provider,
)

class IngredientOutput(BaseModel):
    ingredients: List[str]

    model_config = {
        "extra": "forbid"  # no other keys allowed
    }

    @field_validator("ingredients")
    @classmethod
    def validate_and_normalize_ingredients(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise TypeError("'ingredients' must be a list")

        cleaned = []
        seen = set()

        for item in v:
            if not isinstance(item, str):
                raise TypeError("Each ingredient must be a string")

            ingredient = item.strip().lower()

            if not ingredient:
                raise ValueError("Empty ingredient not allowed")

            if ingredient in seen:
                continue

            seen.add(ingredient)
            cleaned.append(ingredient)

        if not cleaned:
            raise ValueError("At least one ingredient is required")

        return cleaned
    
ingredient_extractor = Agent(
    name="IngredientExtractor",
    model=model,
    output_type=IngredientOutput,
    retries=3,  # keep low; failures should be fixed, not masked
    system_prompt="""
You are a strict information extraction system.

TASK:
Extract only food ingredients from the input text.

INPUT:
A recipe section that may contain ingredients, instructions, and noise.

STRICT RULES:
- Output ONLY valid JSON matching the schema
- Do NOT output Python dictionaries
- Do NOT include explanations, markdown, or extra keys
- Return lowercase ingredient names only
- Remove duplicates
- Keep ingredient names concise but complete (e.g., "olive oil")
- Do NOT infer ingredients not explicitly present
- Ignore:
  - instructions (mix, bake, stir, chop)
  - quantities (1 cup, 2 tbsp, 200g)
  - units (cup, tbsp, ml, g)
  - temperatures and times
  - cookware unless it is a named ingredient

OUTPUT SCHEMA:
{
  "ingredients": ["string"]
}

EXAMPLE:

Input:
Cook elbow macaroni in boiling water. Add cheddar cheese and butter.

Output:
{
  "ingredients": ["elbow macaroni", "cheddar cheese", "butter"]
}

Input:
Ingredients: flour, sugar, milk, eggs. Mix and bake.

Output:
{
  "ingredients": ["flour", "sugar", "milk", "eggs"]
}
"""
)