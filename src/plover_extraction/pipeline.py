"""End-to-end raw-text Plover extraction pipeline."""

import json
from string import Template

from .config import PipelineConfig
from .llm import LLMClient
from .models import EventMention, ExtractionRequest, ExtractionResult, Sentence
from .sentences import split_sentences


def build_user_prompt(
    request: ExtractionRequest, config: PipelineConfig, sentences: list[Sentence]
) -> str:
    """Render the externally configured extraction prompt."""
    sentence_data = [
        {"id": sentence.sentence_id, "text": sentence.text}
        for sentence in sentences
    ]
    try:
        return Template(config.user_prompt_template).substitute(
            doc_id=json.dumps(request.doc_id, ensure_ascii=False),
            ontology=json.dumps(config.ontology, ensure_ascii=False),
            sentences=json.dumps(sentence_data, ensure_ascii=False),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid user_prompt_template placeholder: {error}") from error


async def extract_document(
    request: ExtractionRequest, config: PipelineConfig, client: LLMClient
) -> ExtractionResult:
    """Split raw text, query the LLM, and validate its output against the request."""
    sentences, _ = split_sentences(
        request.raw_text, config.sentence_split_pattern
    )
    result = await client.generate_json(
        config.system_prompt, build_user_prompt(request, config, sentences)
    )
    print("\n\nGenerated JSON:", json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not isinstance(result, dict) or not isinstance(result.get("events"), list):
        raise ValueError("LLM output must be an object containing an events array")
    known_ids = {sentence.sentence_id for sentence in sentences}
    mentions: list[EventMention] = []
    for raw in result["events"]:
        if not isinstance(raw, dict):
            raise ValueError("Each LLM event must be an object")
        raw["doc_id"] = request.doc_id  
        allowed = {
            "doc_id", "actor_text", "recipient_text", "category", "mode",
            "evidence_sentence_ids",
        }
        if set(raw) != allowed:
            raise ValueError(f"LLM event fields must be exactly: {', '.join(sorted(allowed))}")
        if raw["doc_id"] != request.doc_id:
            raise ValueError("LLM event doc_id must match the requested document")
        evidence_value = raw["evidence_sentence_ids"]
        if not isinstance(evidence_value, list) or not all(
            isinstance(value, str) for value in evidence_value
        ):
            raise ValueError("evidence_sentence_ids must be an array of strings")
        evidence = tuple(evidence_value)
        unknown = set(evidence) - known_ids
        if unknown:
            raise ValueError(f"LLM returned unknown sentence IDs: {', '.join(sorted(unknown))}")
        mentions.append(EventMention(
            doc_id=request.doc_id,
            actor_text=raw["actor_text"],
            recipient_text=raw["recipient_text"],
            category=raw["category"],
            mode=raw["mode"].split("||") if raw["mode"] is not None else None,
            evidence_sentence_ids=evidence,
            evidence={
                sentence.sentence_id: sentence.text
                for sentence in sentences
                if sentence.sentence_id in evidence
            },
        ))
    return ExtractionResult(request.doc_id, tuple(sentences), tuple(mentions))
