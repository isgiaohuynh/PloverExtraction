#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
import asyncio

from src.plover_extraction.config import load_config
from src.plover_extraction.llm import OpenAICompatibleClient
from src.plover_extraction.models import ExtractionRequest

from src.workflow.graph import sub_graph


def _document_id(
    value: str | None,
    source_key: str,
    raw_text: str,
) -> str:
    if value and value.strip():
        return value.strip()

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                [source_key, raw_text],
                ensure_ascii=False,
            ),
        )
    )


def requests_from_csv(path: Path) -> Iterator[ExtractionRequest]:
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
                "CSV must contain raw_article or raw_text"
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
                    if (value := row.get(name)) and value.strip()
                ),
                None,
            )

            yield ExtractionRequest(
                doc_id=_document_id(
                    source_id,
                    f"{path.resolve()}#row={row_number}",
                    raw_text,
                ),
                raw_text=raw_text,
            )


def parse_requests(
    input_path: Path | None,
    raw_text: str | None,
    doc_id: str | None,
) -> Iterator[ExtractionRequest]:
    if input_path is not None and input_path.suffix.lower() == ".csv":
        if doc_id is not None:
            raise ValueError(
                "--doc-id cannot override IDs for a multi-row CSV"
            )
        yield from requests_from_csv(input_path)
        return

    if raw_text is not None:
        text = raw_text
        source_key = "text-argument"
    elif input_path is not None:
        text = input_path.read_text(encoding="utf-8")
        source_key = str(input_path.resolve())
    else:
        text = sys.stdin.read()
        source_key = "stdin"

    yield ExtractionRequest(
        doc_id=_document_id(doc_id, source_key, text),
        raw_text=text,
    )


def atomic_write_json(path: Path, value: object) -> None:
    """Replace the destination only after writing the complete JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_signature(
    requests: list[ExtractionRequest],
    config_path: Path,
) -> str:
    """Prevent mixing results from different inputs or YAML settings."""
    digest = hashlib.sha256()
    #digest.update(config_path.read_bytes())

    for request in requests:
        digest.update(
            json.dumps(
                [request.doc_id, request.raw_text],
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def checkpoint_data(signature: str, results: list[dict]) -> dict:
    completed = [
        result["doc_id"]
        for result in results
        if result.get("status", "success") == "success"
    ]

    return {
        "version": 1,
        "run_signature": signature,
        "completed_doc_ids": completed,
        "completed_count": len(completed),
        "retry_doc_ids": [
            result["doc_id"]
            for result in results
            if result.get("status", "success") != "success"
        ],
    }


def load_progress(
    output_path: Path,
    checkpoint_path: Path,
    signature: str,
    requests: list[ExtractionRequest],
) -> list[dict]:
    results = []

    if output_path.exists():
        results = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        if not isinstance(results, list):
            raise ValueError("Output JSON must be an array")

    if checkpoint_path.exists():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("version") != 1
            #or checkpoint.get("run_signature") != signature
        ):
            raise ValueError(
                "Invalid checkpoint format"
                # "Checkpoint does not match input/config. "
                # "Use new output and checkpoint paths for a new run."
            )
    elif results:
        raise ValueError(
            "Output contains results but checkpoint is missing. "
            #"Use new output and checkpoint paths."
        )

    request_map = {
        request.doc_id: request.raw_text for request in requests
    }
    completed_ids = set()

    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Invalid document in output JSON")

        doc_id = result.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in request_map:
            raise ValueError("Output contains an unexpected doc_id")

        if doc_id in completed_ids:
            raise ValueError(f"Duplicate output doc_id: {doc_id}")

        if result.get("raw_text") != request_map[doc_id]:
            raise ValueError(f"Output text mismatch: {doc_id}")

        if not isinstance(result.get("events"), list):
            raise ValueError(f"Invalid events in output: {doc_id}")

        completed_ids.add(doc_id)

    # Output is authoritative: a crash may happen after saving output
    # but before updating checkpoint.
    atomic_write_json(
        checkpoint_path,
        checkpoint_data(signature, results),
    )
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LangGraph extraction with resume support"
    )
    parser.add_argument("--config", type=Path, default=Path("config.example.yaml"))
    parser.add_argument("--doc-id")
    parser.add_argument("--text")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/extraction_results.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Default: <output filename>.checkpoint.json",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="CSV or UTF-8 text file",
    )
    args = parser.parse_args()

    if args.text is not None and args.input is not None:
        parser.error("Use either --text or an input file")

    checkpoint_path = args.checkpoint or args.output.with_name(
        f"{args.output.stem}.checkpoint.json"
    )

    # Protect input/config files from accidental overwriting.
    protected_paths = {args.config.resolve()}
    if args.input is not None:
        protected_paths.add(args.input.resolve())

    if (
        args.output.resolve() == checkpoint_path.resolve()
        or args.output.resolve() in protected_paths
        or checkpoint_path.resolve() in protected_paths
    ):
        parser.error(
            "Output/checkpoint paths must be distinct "
            "and must not overwrite input/config"
        )

    client = None

    try:
        requests = list(
            parse_requests(args.input, args.text, args.doc_id)
        )

        request_ids = [request.doc_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(
                "Input contains duplicate doc_ids; each row needs a unique ID"
            )

        pipeline_config = load_config(args.config)
        signature = run_signature(requests, args.config)

        results = load_progress(
            args.output,
            checkpoint_path,
            signature,
            requests,
        )
        completed_ids = {
            result["doc_id"]
            for result in results
            if result.get("status", "success") == "success"
        }

        pending = [
            request
            for request in requests
            if request.doc_id not in completed_ids
        ]

        if pending:
            client = OpenAICompatibleClient(pipeline_config.llm)
            graph = sub_graph(pipeline_config, client)


        results_by_id = {
            result["doc_id"]: result for result in results
        }

        for position, request in enumerate(pending, start=1):
            print(
                f"[{position}/{len(pending)}] doc_id={request.doc_id}",
                flush=True,
            )

            try:
                state = await graph.ainvoke({
                    "doc_id": request.doc_id,
                    "raw_text": request.raw_text,
                })

                result = {
                    "doc_id": state["doc_id"],
                    "raw_text": state["raw_text"],
                    "sentences": {
                        item["sentence_id"]: item["text"]
                        for item in state["sentences"]
                    },
                    "events": state["events"],
                    "status": state["status"],
                    "errors": state.get("errors", []),
                    "raw_extraction": state.get("raw_extraction"),
                }

            except Exception as error:
                # Catch document failures; KeyboardInterrupt vẫn dừng được.
                print(
                    f"Document failed: {request.doc_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

                result = {
                    "doc_id": request.doc_id,
                    "raw_text": request.raw_text,
                    "sentences": {},
                    "events": [],
                    "status": "failed",
                    "errors": [{
                        "doc_id": request.doc_id,
                        "stage": "graph",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }],
                    "raw_extraction": None,
                }

            results_by_id[request.doc_id] = result
            results = list(results_by_id.values())

            # Giữ ngoài try/except: lỗi lưu file cần được báo và dừng.
            atomic_write_json(args.output, results)
            atomic_write_json(
                checkpoint_path,
                checkpoint_data(signature, results),
            )

            print(
                f"status={result['status']} | "
                f"events={len(result['events'])} | "
                f"errors={len(result['errors'])}",
                flush=True,
            )

        print(f"Saved {len(results)} documents to {args.output}")

    except KeyboardInterrupt:
        parser.exit(
            130,
            "\nStopped. Rerun the same command to resume.\n",
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        if client is not None:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())