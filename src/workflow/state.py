from typing import TypedDict, Annotated, Literal, Any
import operator

class SentenceState(TypedDict):
    sentence_id: str
    text: str


class ExtractionState(TypedDict, total=False):
    doc_id: str
    raw_text: str

    sentences: list[SentenceState]
    joined_sentences: str

    raw_extraction: object

    events: list[dict[str, object]]
    status: Literal["success", "partial", "failed"]
    errors: list[dict[str, Any]]