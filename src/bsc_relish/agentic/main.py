import json
import os
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import RunContext
import random


class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    confidence: float


class RecipeBundle(BaseModel):
    recipes: List[Recipe]



def roberta_extract(text: str) -> List[dict]:
    """
    Simulated RoBERTa extraction tool.
    Replace this later with real HF model inference.
    """

    

    candidates = [
        {
            "title": "Spaghetti Carbonara",
            "ingredients": [
                "200g spaghetti",
                "2 eggs",
                "100g pancetta",
                "parmesan"
            ],
            "instructions": [
                "Boil pasta",
                "Cook pancetta",
                "Mix eggs and cheese",
                "Combine everything"
            ],
            "confidence": 0.93
        },
        {
            "title": "Tomato Soup",
            "ingredients": [
                "tomatoes",
                "onion",
                "garlic",
                "olive oil"
            ],
            "instructions": [
                "Cook vegetables",
                "Blend",
                "Simmer"
            ],
            "confidence": 0.85
        }
    ]

    # simulate variability
    return random.sample(candidates, k=random.randint(1, 2))

provider = OpenAIProvider(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

model = OpenAIChatModel(
    model_name="qwen2.5",
    provider=provider
)

agent = Agent[None, RecipeBundle](model)



@agent.tool
def extract_recipes(ctx: RunContext[None], text: str) -> List[dict]:
    """
    Tool wrapping RoBERTa extraction.
    """
    return roberta_extract(text)



SYSTEM_PROMPT = """
Return ONLY valid JSON matching this schema:

{
  "recipes": [
    {
      "title": "...",
      "ingredients": ["..."],
      "instructions": ["..."],
      "confidence": 0.0
    }
  ]
}

No explanations. No markdown. No extra text.
"""



def save_json(data, folder="outputs", filename="recipes.json"):
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def run_agent(book_text: str) -> RecipeBundle:
    result = agent.run_sync(
        SYSTEM_PROMPT + "\n\nTEXT:\n" + book_text,
        deps=None
    )
    return result.output


if __name__ == "__main__":

    book_text = """
    Chapter 1: Italian Classics

    This book contains traditional recipes such as pasta and soups.
    """

    result = agent.run_sync(SYSTEM_PROMPT + "\n\n" + book_text)

    data = json.loads(result.output)
    save_json(data)
    print(json.dumps(data, indent=2))
    