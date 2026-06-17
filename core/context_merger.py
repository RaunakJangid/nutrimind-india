from __future__ import annotations

import re
from typing import Any

from core.models import Conflict, MergedContext, RetrievalResult


NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|ug|mcg)", re.I)


class ContextMerger:
    def merge(self, retrieval: RetrievalResult) -> MergedContext:
        conflicts = self.detect_conflicts(retrieval)
        confidence_parts = []
        if retrieval.rda:
            confidence_parts.append(float(retrieval.rda.get("confidence", 1.0)))
        confidence_parts.extend(float(food.get("confidence", 1.0)) for food in retrieval.foods)
        confidence_parts.extend(max(0.0, min(1.0, chunk.score)) for chunk in retrieval.semantic)
        confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.0
        return MergedContext(
            rda=retrieval.rda,
            foods=retrieval.foods,
            semantic=retrieval.semantic,
            conflicts=conflicts,
            context_text=self._build_context_text(retrieval, conflicts),
            confidence=confidence,
        )

    def detect_conflicts(self, retrieval: RetrievalResult) -> list[Conflict]:
        conflicts: list[Conflict] = []
        if retrieval.rda:
            rda_value = float(retrieval.rda["value"])
            rda_unit = str(retrieval.rda["unit"]).lower()
            nutrient = str(retrieval.rda["nutrient"]).replace("_", " ")
            for chunk in retrieval.semantic:
                text = chunk.text.lower()
                if nutrient not in text:
                    continue
                for raw_value, raw_unit in NUMBER_RE.findall(text):
                    value = float(raw_value)
                    unit = raw_unit.lower()
                    if unit == rda_unit and abs(value - rda_value) > max(0.5, rda_value * 0.1):
                        conflicts.append(
                            Conflict(
                                type="structured_vs_semantic",
                                sources=[str(retrieval.rda["source"]), chunk.source],
                                description=f"Structured RDA has {rda_value:g} {rda_unit}, text mentions {value:g} {unit}.",
                                resolution="Structured RDA value is used for calculations; text is citation context only.",
                            )
                        )
        return conflicts

    def _build_context_text(self, retrieval: RetrievalResult, conflicts: list[Conflict]) -> str:
        lines: list[str] = []
        if retrieval.rda:
            rda = retrieval.rda
            lines.append(f"ICMR RDA: {rda['nutrient']} for {rda['age_group']} = {rda['value']:g} {rda['unit']} [Source: {rda['source']}]")
        for food in retrieval.foods:
            lines.append(
                f"IFCT: {food['food']} maps to {food['mapped_to']}; {food['nutrient']} = "
                f"{food['value_per_100g']:g} {food['unit']} per 100g [Source: {food['source']}]"
            )
        for chunk in retrieval.semantic:
            lines.append(f"ICMR-NIN Text ({chunk.title or chunk.chunk_id}, score {chunk.score:.2f}): {chunk.text}")
        for conflict in conflicts:
            lines.append(f"Conflict: {conflict.description} Resolution: {conflict.resolution}")
        return "\n".join(lines)


def merge_context(retrieval: RetrievalResult) -> MergedContext:
    return ContextMerger().merge(retrieval)
