from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal, List, Optional


class DocumentInput(BaseModel):
    source: str
    source_type: str  # string | txt | pdf


class NormalizedDocument(BaseModel):
    content: str
    source_type: Literal["text", "pdf", "string"]
    metadata: dict = Field(default_factory=dict)


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IngredientsList(BaseModel):
    ingredients: list[str]


class Chunk(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    origin: str = ""
    length: int = 0

class Section(BaseModel):
    type: str  # ingredients | instructions | notes | unknown
    content: str


class Recipe(BaseModel):
    recipe_id: str
    title: str | None
    sections: list[Section]


class BoundaryExtractionResult(BaseModel):
    recipes: list[Recipe]

class BoundaryInput(BaseModel):
    text: str


class BoundaryOutput(BaseModel):
    recipe: Optional[str]
    start: Optional[int]
    end: Optional[int]
