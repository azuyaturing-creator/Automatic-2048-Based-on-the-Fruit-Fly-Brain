"""Validate the neuPrint token and save a safe connection report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from neuprint import Client, fetch_meta

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
TOKEN_ENV_VAR = "NEUPRINT_APPLICATION_CREDENTIALS"
DEFAULT_REPORT = Path("outputs/connection/connection_report.json")
META_KEYS = (
    "dataset",
    "description",
    "info",
    "lastDatabaseEdit",
    "latestMutationId",
    "hideDataSet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check neuPrint access for the MaleCNS v1.0 dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Path of the JSON report (default: {DEFAULT_REPORT})",
    )
    return parser.parse_args()


def redact_message(message: str, token: str) -> str:
    """Keep token values out of user-facing error messages."""
    redacted = message
    if token:
        redacted = redacted.replace(token, "[REDACTED]")
        stripped_token = token.strip()
        if stripped_token:
            redacted = redacted.replace(stripped_token, "[REDACTED]")
    return redacted


def safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep only stable, non-sensitive dataset metadata."""
    return {key: meta[key] for key in META_KEYS if key in meta}


def build_report(client: Client, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "server": SERVER,
        "dataset_requested": DATASET,
        "dataset_selected": client.dataset,
        "token_present": True,
        "dataset_info": safe_meta(meta),
    }


def write_report(report: dict[str, Any], output: Path, token: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if token and token in serialized:
        raise RuntimeError("Refusing to write a report that contains the token.")
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()

    if not token:
        print(f"ERROR: {TOKEN_ENV_VAR} is not set.", file=sys.stderr)
        print(
            "Set it in your terminal, or on Windows run: "
            "setx NEUPRINT_APPLICATION_CREDENTIALS \"your-token\"",
            file=sys.stderr,
        )
        return 2

    try:
        client = Client(SERVER, dataset=DATASET, token=token, progress=False)
        meta = fetch_meta(client=client)
        report = build_report(client, meta)
        write_report(report, args.output, token)
    except requests.exceptions.Timeout as exc:
        print(
            "ERROR: neuPrint request timed out. "
            f"{redact_message(str(exc), token)}",
            file=sys.stderr,
        )
        return 3
    except requests.exceptions.RequestException as exc:
        print(
            "ERROR: could not connect to neuPrint. "
            f"{redact_message(str(exc), token)}",
            file=sys.stderr,
        )
        return 3
    except RuntimeError as exc:
        print(f"ERROR: {redact_message(str(exc), token)}", file=sys.stderr)
        return 4
    except Exception as exc:  # Keep the CLI failure readable without a stack trace.
        print(f"ERROR: unexpected failure: {redact_message(str(exc), token)}", file=sys.stderr)
        return 5

    description = str(meta.get("description", "")).splitlines()[0]
    print("OK: neuPrint connection succeeded.")
    print(f"Dataset: {client.dataset}")
    if description:
        print(f"Description: {description}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
