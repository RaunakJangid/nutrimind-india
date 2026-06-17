from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Intent = Literal["rda_lookup", "diet_check", "general_question", "unknown"]


class QueryEntities(BaseModel):
    nutrient: str | None = None
    age_months: int | None = None
    foods: list[str] = Field(default_factory=list)
    servings: dict[str, float] = Field(default_factory=dict)
    intent: Intent = "unknown"
    confidence: float = 0.0


class FoodContribution(BaseModel):
    food: str
    mapped_to: str
    servings: float
    grams: float
    value_per_100g: float
    nutrient_amount: float
    unit: str
    source: str
    confidence: float = 1.0


class CalculationResult(BaseModel):
    nutrient: str
    age_months: int
    age_group: str
    required_value: float
    required_unit: str
    consumed_value: float = 0.0
    consumed_unit: str
    gap_value: float
    gap_percent: float
    food_details: list[FoodContribution] = Field(default_factory=list)
    unknown_foods: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    rda_source: dict[str, Any]
    intent: Intent
    conflicts: list["Conflict"] = Field(default_factory=list)


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str


class VerificationResult(BaseModel):
    verified: bool
    checks: list[CheckResult] = Field(default_factory=list)
    fail_reason: str | None = None


class SemanticChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float = 0.0
    chapter: str | None = None
    section: str | None = None
    page: int | None = None
    title: str | None = None


class RetrievalResult(BaseModel):
    rda: dict[str, Any] | None = None
    foods: list[dict[str, Any]] = Field(default_factory=list)
    semantic: list[SemanticChunk] = Field(default_factory=list)
    age_group: str | None = None
    unknown_foods: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Conflict(BaseModel):
    type: str
    sources: list[str]
    description: str
    resolution: str


class MergedContext(BaseModel):
    rda: dict[str, Any] | None = None
    foods: list[dict[str, Any]] = Field(default_factory=list)
    semantic: list[SemanticChunk] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    context_text: str = ""
    confidence: float = 0.0


class SynthesisResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    model_backend: str
    prompt_version: str = "v1"
