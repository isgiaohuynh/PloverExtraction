"""End-to-end raw-text Plover extraction pipeline."""
import csv
from pathlib import Path
import uuid
from collections.abc import Iterator

from src.plover_extraction.config import  PipelineConfig
from src.plover_extraction.sentences import split_sentences
from src.plover_extraction.models import EventMention, ExtractionRequest


from src.workflow.state import ExtractionState

def _document_id(value: str | None) -> str:
    """Preserve a supplied document ID, or create one when missing."""
    if value and value.strip():
        return value.strip()
    return str(uuid.uuid4())


def prepare_csv(path: str | Path) -> Iterator[ExtractionRequest]:
    """Yield one document per CSV row."""
    path = Path(path)

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())

        text_column = next(
            (
                name
                for name in ("raw_article", "raw_text")
                if name in fields
            ),
            None,
        )

        if text_column is None:
            raise ValueError(
                "CSV must contain a raw_article or raw_text column"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_text = row.get(text_column)

            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(
                    f"CSV row {row_number}: empty {text_column}"
                )

            source_id = next(
                (
                    value
                    for name in ("doc_id", "article_id")
                    if (value := row.get(name))
                    and value.strip()
                ),
                None,
            )

            yield ExtractionRequest(
                doc_id=_document_id(source_id),
                raw_text=raw_text,
            )

def sentence_indexing(raw_text, pipeline_config: PipelineConfig):
    sentences, joined_sentences = split_sentences(raw_text, pipeline_config.sentence_split_pattern)
    print("\n\njoined_sentences",joined_sentences)

    return sentences, joined_sentences

    # return {"sentences": [
    #         {
    #             "sentence_id": sentence.sentence_id,
    #             "text": sentence.text,
    #         }
    #         for sentence in sentences],

    #         "joined_sentences": joined_sentences}


def prepare_doc(state: ExtractionState, pipeline_config: PipelineConfig):
    source_id = state.get("doc_id")

    if source_id is not None and not isinstance(source_id, str):
        raise ValueError("doc_id must be a string")

    sentences, joined_sentences = sentence_indexing(
        state.get("raw_text",""),
        pipeline_config
    )

    if not sentences or not joined_sentences.strip():
        raise ValueError(
            f"Document {source_id}: no sentences after splitting"
        )


    return {
        "sentences": [
            {
                "sentence_id": sentence.sentence_id,
                "text": sentence.text,
            }
            for sentence in sentences
        ],
        "joined_sentences": joined_sentences,
    }

