"""End-to-end raw-text Plover extraction pipeline."""
import json
from string import Template
import logging

from src.plover_extraction.config import PipelineConfig, load_config
from src.plover_extraction.llm import LLMClient
from src.plover_extraction.models import EventMention, ExtractionRequest, ExtractionResult, Sentence
from src.workflow.state import ExtractionState

logger = logging.getLogger(__name__)

def build_user_prompt(
    doc_id: str, 
    config: PipelineConfig, 
    sentences: str
) -> str:
    """Render the externally configured extraction prompt."""
    try:
        return Template(config.user_prompt_template).substitute(
            doc_id=json.dumps(doc_id, ensure_ascii=False),
            ontology=json.dumps(config.ontology, ensure_ascii=False),
            sentences=json.dumps(sentences, ensure_ascii=False),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid user_prompt_template placeholder: {error}") from error

def validate_event(
    raw: object,
    doc_id: str,
    sentence_map: dict[str, str],
) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Event must be an object")

    required_fields = {
        "actor_text",
        "recipient_text",
        "category",
        "mode",
        "evidence_sentence_ids",
    }

    missing = required_fields - set(raw)
    unexpected = set(raw) - (required_fields | {"doc_id"})

    if missing or unexpected:
        raise ValueError(
            f"Invalid fields: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    if "doc_id" in raw and raw["doc_id"] != doc_id:
        raise ValueError("doc_id must match the document")

    evidence_ids = raw["evidence_sentence_ids"]

    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or not all(isinstance(value, str) for value in evidence_ids)
    ):
        raise ValueError(
            "evidence_sentence_ids must be a non-empty array of strings"
        )

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Duplicate evidence sentence IDs")

    unknown_ids = set(evidence_ids) - set(sentence_map)
    if unknown_ids:
        raise ValueError(
            f"Unknown sentence IDs: {sorted(unknown_ids)}"
        )

    evidence = {
        sentence_id: sentence_map[sentence_id]
        for sentence_id in evidence_ids
    }

    raw_mode = raw["mode"]

    if raw_mode is None:
        mode = None
    elif isinstance(raw_mode, str):
        mode = [value.strip() for value in raw_mode.split("||")]
    elif isinstance(raw_mode, list) and all(
        isinstance(value, str) for value in raw_mode
    ):
        mode = [value.strip() for value in raw_mode]
    else:
        raise ValueError("mode must be a string, list of strings, or null")

    if mode is not None:
        if not mode or any(not value for value in mode):
            raise ValueError("mode must not contain empty values")

        if len(mode) != len(set(mode)):
            raise ValueError("Duplicate modes")

    roles = {}

    for role in ("actor_text", "recipient_text"):
        value = raw[role]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{role} must be a non-empty string")

        value = value.strip()

        if value.lower() == "unknown":
            value = "unknown"
        elif not any(value in text for text in evidence.values()):
            raise ValueError(
                f"{role}={value!r} is not an exact span "
                f"in cited evidence {evidence_ids}"
            )

        roles[role] = value

    if not isinstance(raw["category"], str):
        raise ValueError("category must be a string")

    mention = EventMention(
        doc_id=doc_id,
        actor_text=roles["actor_text"],
        recipient_text=roles["recipient_text"],
        category=raw["category"],
        mode=mode,
        evidence_sentence_ids=tuple(evidence_ids),
        evidence=evidence,
    )

    return mention.to_dict()


async def raw_extract(
    state: ExtractionState,
    pipeline_config: PipelineConfig,
    client: LLMClient,
) -> ExtractionState:
    doc_id = state["doc_id"]
    result = None
    events = []
    errors = []

    try:
        sentence_map = {
            sentence["sentence_id"]: sentence["text"]
            for sentence in state["sentences"]
        }

        result = await client.generate_json(
            pipeline_config.system_prompt,
            build_user_prompt(
                doc_id,
                pipeline_config,
                state["joined_sentences"],
            ),
        )

        if (
            not isinstance(result, dict)
            or set(result) != {"events"}
            or not isinstance(result["events"], list)
        ):
            raise ValueError(
                "LLM output must contain exactly an events array"
            )

    except Exception as error:
        logger.exception("Document extraction failed: %s", doc_id)

        return {
            "raw_extraction": result,
            "events": [],
            "status": "failed",
            "errors": [{
                "doc_id": doc_id,
                "stage": "raw_extract",
                "error_type": type(error).__name__,
                "message": str(error),
            }],
        }

    # Event-level failure: keep the other valid events.
    for index, raw in enumerate(result["events"]):
        try:
            events.append(
                validate_event(raw, doc_id, sentence_map)
            )

        except (ValueError, TypeError, KeyError) as error:
            logger.warning(
                "Rejected event: doc_id=%s event_index=%s error=%s",
                doc_id,
                index,
                error,
            )

            cited_ids = (
                raw.get("evidence_sentence_ids")
                if isinstance(raw, dict)
                else None
            )

            errors.append({
                "doc_id": doc_id,
                "stage": "event_validation",
                "event_index": index,
                "error_type": type(error).__name__,
                "message": str(error),
                "raw_event": raw,
                "cited_evidence": {
                    sentence_id: sentence_map[sentence_id]
                    for sentence_id in cited_ids
                    if isinstance(sentence_id, str)
                    and sentence_id in sentence_map
                } if isinstance(cited_ids, list) else {},
            })

    if not errors:
        status = "success"
    elif events:
        status = "partial"
    else:
        status = "failed"

    return {
        "raw_extraction": result,
        "events": events,
        "status": status,
        "errors": errors,
    }

# def raw_extract(state: ExtractionState, 
#                 pipeline_config: PipelineConfig,
#                 client: LLMClient):
    
#     doc_id = state['doc_id']
#     sentences = state['sentences']
    
#     result = client.generate_json(
#         pipeline_config.system_prompt,
#         build_user_prompt(
#             doc_id,
#             pipeline_config,
#             state["joined_sentences"],
#         ))
    
#     print("\n\nGenerated JSON:", json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    
#     if (
#         not isinstance(result, dict)
#         or set(result) != {"events"}
#         or not isinstance(result["events"], list)
#     ):
#         raise ValueError(
#             "LLM output must contain exactly an events array"
#         )
    
#     sentence_map = {
#         sentence["sentence_id"]: sentence["text"]
#         for sentence in sentences
#     }

#     required_fields = {
#         "actor_text",
#         "recipient_text",
#         "category",
#         "mode",
#         "evidence_sentence_ids",
#     }

#     allowed_fields = required_fields | {"doc_id"}
#     events = []

#     for index,raw in enumerate(result["events"]):

#         prefix = f"Event {index}"

#         if not isinstance(raw, dict):
#             raise ValueError("Each LLM event must be an object")

#         if (
#             not required_fields.issubset(raw)
#             or set(raw) - allowed_fields
#         ):
#             raise ValueError(
#                 f"{prefix}: missing or unexpected fields"
#             )

#         evidence_ids = raw["evidence_sentence_ids"]

#         if (
#             not isinstance(evidence_ids, list)
#             or not evidence_ids
#             or not all(
#                 isinstance(value, str) for value in evidence_ids
#             )
#         ):
#             raise ValueError(
#                 f"{prefix}: evidence_sentence_ids must be "
#                 "a non-empty array of strings"
#             )
        
#         if len(evidence_ids) != len(set(evidence_ids)):
#             raise ValueError(
#                 f"{prefix}: duplicate evidence sentence IDs"
#             )

#         unknown_ids = set(evidence_ids) - set(sentence_map)
#         if unknown_ids:
#             raise ValueError(
#                 f"{prefix}: unknown sentence IDs: "
#                 f"{sorted(unknown_ids)}"
#             )

#         evidence = {
#             sentence_id: sentence_map[sentence_id]
#             for sentence_id in evidence_ids
#         }

#         mode = raw["mode"]

#         if mode is None:
#             mode = None

#         elif isinstance(mode, str):
#             mode = [
#                 value.strip()
#                 for value in mode.split("||")
#             ]

#             if any(not value for value in mode):
#                 raise ValueError(
#                     f"{prefix}: mode contains an empty mode"
#                 )

#             if len(mode) != len(set(mode)):
#                 raise ValueError(
#                     f"{prefix}: duplicate modes"
#                 )
#         else:
#             raise ValueError(
#                 f"{prefix}: mode must be a string or null"
#             )

#         roles = {}

#         for role in ("actor_text", "recipient_text"):
#             value = raw[role]

#             if not isinstance(value, str) or not value.strip():
#                 raise ValueError(
#                     f"{prefix}: {role} must be a non-empty string"
#                 )

#             value = value.strip()

#             if value.lower() == "unknown":
#                 value = "unknown"
#             elif not any(
#                 value in text for text in evidence.values()
#             ):
#                 raise ValueError(
#                     f"{prefix}: {role} must be an exact span "
#                     "in the cited evidence {evidence_ids}"
#                 )

#             roles[role] = value

#         mention = EventMention(
#             doc_id=doc_id,
#             actor_text=roles["actor_text"],
#             recipient_text=roles["recipient_text"],
#             category=raw["category"],
#             mode=mode,
#             evidence_sentence_ids=tuple(evidence_ids),
#             evidence=evidence,
#         )
#         events.append(mention.to_dict())

#     return {
#         "raw_extraction": result,
#         "events": events,
#     }
