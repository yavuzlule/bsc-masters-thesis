from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from bsc_relish.agentic.models import NormalizedDocument, IngredientsList
provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)


from pydantic import BaseModel, field_validator, model_validator
from typing import List


class ActionOutput(BaseModel):
    actions: List[str]

    model_config = {
        "extra": "forbid"  # no other keys allowed
    }

    @field_validator("actions")
    @classmethod
    def validate_and_normalize_actions(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise TypeError("'actions' must be a list")

        cleaned = []
        seen = set()

        for item in v:
            if not isinstance(item, str):
                raise TypeError("Each action must be a string")

            action = item.strip().lower()

            if not action:
                raise ValueError("Empty action not allowed")

            if action in seen:
                continue

            seen.add(action)
            cleaned.append(action)

        if not cleaned:
            raise ValueError("At least one action is required")

        return cleaned
    

model = OpenAIChatModel(
    model_name="qwen2.5",
    provider=provider
)

action_extractor = Agent(
    name="ActionExtractor",
    output_type=ActionOutput,
    model=model,
    retries=3,
    system_prompt="""
You extract cooking actions from recipe text.

INPUT:
A full recipe section (may include ingredients, instructions, or mixed text).

TASK:
Extract ONLY cooking actions / steps.

ACTIONS INCLUDE:
- verbs describing cooking processes (e.g., mix, bake, boil, stir, chop, whisk, fry, simmer, preheat)
- explicit preparation steps (e.g., "let rest", "marinate", "knead dough")

IGNORE:
- ingredients (food items)
- quantities, units, times, temperatures
- cookware unless it is part of an action phrase (e.g., "heat pan")
- descriptive or narrative text

RULES:
- Output must be VALID JSON ONLY
- Do NOT include any text outside JSON
- Keep actions concise and normalized (lowercase verb phrases)
- Deduplicate actions
- Do NOT include ingredients or objects alone (e.g., "oven" is invalid, "preheat oven" is valid)
- Do NOT infer missing steps

OUTPUT FORMAT (STRICT):
{
  "actions": ["string", "string", ...]
}

EXAMPLES:

Input:
"Chop onions and fry them in a pan. Then add tomatoes and simmer."

Output:
{
  "actions": ["chop onions", "fry", "add tomatoes", "simmer"]
}

Input:
"Ingredients: flour, eggs. Mix well and bake at 180C."

Output:
{
  "actions": ["mix", "bake"]
}
"""
)