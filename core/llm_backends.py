from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, context: dict) -> str:
        raise NotImplementedError


class GeminiBackend(LLMBackend):
    name = "gemini"

    def generate(self, prompt: str, context: dict) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError("GEMINI_API_KEY is not configured")

        # Migrated from deprecated google-generativeai to google-genai SDK
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        return response.text.strip()


class LlamaBackend(LLMBackend):
    """
    Uses FreeLLMAPI (OpenAI-compatible endpoint at localhost:3001/v1).
    """
    name = "llama"

    def generate(self, prompt: str, context: dict) -> str:
        base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:3001/v1").rstrip("/")
        api_key = os.getenv(
            "LLAMA_API_KEY",
            "freellmapi-62ac52438261a2c3f941ed0d2cd42518e7daa3b40b01d451",
        )
        payload = json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"FreeLLMAPI backend unavailable: {exc}") from exc
        return str(data["choices"][0]["message"]["content"]).strip()


class DeterministicBackend(LLMBackend):
    name = "deterministic"

    def generate(self, prompt: str, context: dict) -> str:
        calculation = context.get("calculation")
        if not calculation:
            chunks = context.get("merged_context", {}).get("semantic", [])
            if chunks:
                text = chunks[0].get("text", "").strip()
                return f"{text} [Source: ICMR-NIN Text]"
            return "I found general nutrition context, but not enough structured data for a verified calculation. [Source: ICMR-NIN Text]"
        nutrient = calculation["nutrient"].replace("_", " ")
        required = calculation["required_value"]
        unit = calculation["required_unit"]
        consumed = calculation["consumed_value"]
        gap = calculation["gap_value"]
        return (
            f"For this age group, the daily {nutrient} requirement is {required:g} {unit} [Source: ICMR RDA]. "
            f"The listed foods provide about {consumed:.1f} {unit} using IFCT values [Source: IFCT]. "
            f"The remaining gap is {gap:.1f} {unit}."
        )


def get_backend(name: str | None = None) -> LLMBackend:
    backend = (name or os.getenv("MODEL_BACKEND", "gemini")).lower()
    if backend == "gemini":
        return GeminiBackend()
    if backend == "llama":
        return LlamaBackend()
    return DeterministicBackend()