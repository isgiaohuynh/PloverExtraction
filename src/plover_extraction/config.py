"""Dependency-free loading for the project's deliberately small YAML subset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import os

load_dotenv()  

from .ontology import PLOVER_ONTOLOGY

DEFAULT_SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+|\n+"


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Parse top-level scalars, JSON-style lists, and ``|`` block strings."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    result: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace() or ":" not in line:
            raise ValueError(f"Invalid top-level YAML at line {index}: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if value.strip() in {"|", "|-", ">", ">-"}:
            block: list[str] = []
            while index < len(lines) and (not lines[index] or lines[index][:1].isspace()):
                block_line = lines[index]
                index += 1
                block.append(block_line[2:] if block_line.startswith("  ") else block_line)
            result[key] = "\n".join(block).rstrip()
        else:
            result[key] = _scalar(value)
    return result


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model: str
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 2


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    llm: LLMConfig
    ontology: dict[str, Any]
    system_prompt: str
    user_prompt_template: str
    sentence_split_pattern: str = DEFAULT_SENTENCE_SPLIT_PATTERN


def load_config(path: str | Path) -> PipelineConfig:
    """Load model settings, label definitions, and prompts from YAML."""
    raw = _load_simple_yaml(path)
    try:
        llm = LLMConfig(**{key: raw[key] for key in (
            "model", "api_base", "api_key_env", "temperature",
            "timeout_seconds", "max_retries",
        )})
        config = PipelineConfig(
            llm=llm,
            ontology=json.loads(raw["ontology_json"]),
            system_prompt=raw["system_prompt"],
            user_prompt_template=raw["user_prompt_template"],
            sentence_split_pattern=raw.get(
                "sentence_split_pattern", DEFAULT_SENTENCE_SPLIT_PATTERN
            ),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid configuration: {error}") from error
    if set(config.ontology) != set(PLOVER_ONTOLOGY):
        raise ValueError("ontology_json must define every canonical Plover category")
    for category, canonical_types in PLOVER_ONTOLOGY.items():
        item = config.ontology[category]
        if not isinstance(item, dict) or not str(item.get("description", "")).strip():
            raise ValueError(f"Missing description for category {category}")
        modes = item.get("modes")
        expected = {value for value in canonical_types if value is not None}
        if not isinstance(modes, dict) or set(modes) != expected:
            raise ValueError(f"modes for {category} do not match canonical ontology")
        if not all(isinstance(value, str) and value.strip() for value in modes.values()):
            raise ValueError(f"Every mode in {category} must have a description")
        if item.get("allow_category_only") is not (None in canonical_types):
            raise ValueError(f"Invalid allow_category_only for {category}")
    for name in ("model", "system_prompt", "user_prompt_template"):
        if not getattr(config.llm, name, None) and not getattr(config, name, None):
            raise ValueError(f"{name} must be a non-empty string")
    if config.llm.max_retries < 0 or config.llm.timeout_seconds <= 0:
        raise ValueError("LLM retries and timeout must be non-negative/positive")
    return config
