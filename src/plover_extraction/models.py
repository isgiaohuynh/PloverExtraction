"""Validated input and output models for extraction pipelines."""

from dataclasses import dataclass
from typing import Any

from .ontology import PloverCategory, validate_mode


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Sentence:
    sentence_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentence_id", _required(self.sentence_id, "sentence_id"))
        object.__setattr__(self, "text", _required(self.text, "text"))


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """A raw document from which to extract all supported event labels."""

    doc_id: str
    raw_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _required(self.doc_id, "doc_id"))
        object.__setattr__(self, "raw_text", _required(self.raw_text, "raw_text"))


@dataclass(frozen=True, slots=True)
class EventMention:
    doc_id: str
    actor_text: str
    recipient_text: str
    category: PloverCategory
    mode: list | None
    evidence_sentence_ids: tuple[str, ...]
    evidence: dict[str, str]

    def __post_init__(self) -> None:
        for field in ("doc_id", "actor_text", "recipient_text"):
            object.__setattr__(self, field, _required(getattr(self, field), field))
        validate_mode(self.category, self.mode)
        if not self.evidence_sentence_ids:
            raise ValueError("evidence_sentence_ids must not be empty")
        ids = tuple(_required(value, "evidence_sentence_ids item") for value in self.evidence_sentence_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_sentence_ids must not contain duplicates")
        object.__setattr__(self, "evidence_sentence_ids", ids)
        if set(self.evidence) != set(ids):
            raise ValueError("evidence must contain exactly the evidence_sentence_ids")
        object.__setattr__(self, "evidence", {
            sentence_id: _required(text, f"evidence[{sentence_id}]")
            for sentence_id, text in self.evidence.items()
        })

    def to_dict(self) -> dict[str, Any]:
        """Return the exact public JSON shape."""
        return {
            "doc_id": self.doc_id,
            "actor_text": self.actor_text,
            "recipient_text": self.recipient_text,
            "category": self.category,
            "mode": self.mode,
            "evidence_sentence_ids": list(self.evidence_sentence_ids),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    doc_id: str
    sentences: tuple[Sentence, ...]
    events: tuple[EventMention, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sentences": {
                sentence.sentence_id: sentence.text for sentence in self.sentences
            },
            "events": [event.to_dict() for event in self.events],
        }
