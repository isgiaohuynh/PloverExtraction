#!/usr/bin/env python3
"""Extract Plover events from a CSV file, text file, or raw-text argument."""

import argparse
import asyncio
import csv
import json
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

from src.plover_extraction.config import load_config
from src.plover_extraction.llm import OpenAICompatibleClient
from src.plover_extraction.models import ExtractionRequest
from src.plover_extraction.pipeline import extract_document


def _document_id(value: str | None) -> str:
    return value.strip() if value and value.strip() else str(uuid.uuid4())


def requests_from_csv(path: Path) -> Iterator[ExtractionRequest]:
    """Yield requests from CSV rows containing ``raw_text`` or ``raw_article``."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        text_column = next((name for name in ("raw_text", "raw_article") if name in fields), None)
        if text_column is None:
            raise ValueError("CSV must contain a raw_text or raw_article column")

        for row_number, row in enumerate(reader, 2):
            raw_text = row.get(text_column, "")
            if not raw_text or not raw_text.strip():
                raise ValueError(f"CSV row {row_number} has empty {text_column}")
            source_id = row.get("doc_id") or row.get("article_id")
            yield ExtractionRequest(_document_id(source_id), raw_text)


def parse_requests(
    input_path: Path | None, raw_text: str | None, doc_id: str | None
) -> Iterator[ExtractionRequest]:
    """Create requests from exactly one supported input source."""
    if raw_text is not None:
        yield ExtractionRequest(_document_id(doc_id), raw_text)
    elif input_path is not None and input_path.suffix.lower() == ".csv":
        if doc_id is not None:
            raise ValueError("--doc-id cannot override IDs for a multi-row CSV")
        yield from requests_from_csv(input_path)
    elif input_path is not None:
        yield ExtractionRequest(
            _document_id(doc_id), input_path.read_text(encoding="utf-8")
        )
    else:
        yield ExtractionRequest(_document_id(doc_id), sys.stdin.read())


async def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = load_config(args.config)
    client = OpenAICompatibleClient(config.llm)
    results = []
    try:
        for request in parse_requests(args.input, args.text, args.doc_id):
            result = await extract_document(request, config, client)
            results.append(result.to_dict())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        await client.close()

    output = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def main() -> None:
#     text = '''Chebrov, director of the Kamchatka branch of the Russian Academy of Sciences "Unified Geophysics Agency", which is responsible for earthquake monitoring, said on July 31 that after a strong magnitude 8.7 earthquake occurred in the waters of the Kamchatka Peninsula on July 30, about 1,000 aftershocks were monitored on the peninsula.Russian experts generally believe that this strong earthquake may be the last stage of earthquake activity in the region in more than a year.
# According to the Russian News Agency today, Chebrov confirmed that the magnitude of these aftershocks ranged from the minimum magnitude that the instrument could measure to about 7 magnitude.It is expected that in the next few months, "there will be a sense of aftershocks near the Kamchatka Peninsula, with the total number reaching thousands of times."
# In addition, Russian emergency rescue departments confirmed on the 31st that the strong earthquake caused some coastal areas of Kamchatka Peninsula and Sakhalin Island to be hit by a tsunami of 5 meters with the largest wave height, and about 2,700 people were evacuated to the Kuril Islands to avoid the tsunami.On that day, the water accumulation in the area hit by the tsunami had subsided.
# Israeli-Kazakhstan negotiations are on the verge of breakdown, Israeli military considers expanding military operations
# '''

    
    parser = argparse.ArgumentParser(
        description="Extract Plover events from CSV, a text file, raw text, or stdin"
    )
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--doc-id", help="Document ID for raw text or a text file")
    parser.add_argument("--text", help="Raw text to extract instead of reading a file")
    parser.add_argument("--output", type=Path, help="Write JSON to this file")
    parser.add_argument("input", nargs="?", type=Path, help="CSV or UTF-8 text file")
    args = parser.parse_args()
    if args.text is not None and args.input is not None:
        parser.error("use either --text or an input file, not both")

    asyncio.run(_run(args, parser))


if __name__ == "__main__":
    main()
