"""Fetch one MaleCNS neuron and its upstream/downstream connections."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from neuprint import Client, fetch_adjacencies, fetch_neurons

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
TOKEN_ENV_VAR = "NEUPRINT_APPLICATION_CREDENTIALS"
DEFAULT_OUTPUT_DIR = Path("outputs/neurons")
NEURON_COLUMNS = [
    "bodyId",
    "instance",
    "type",
    "pre",
    "post",
    "upstream",
    "downstream",
]
CONNECTION_COLUMNS = [
    "direction",
    "source",
    "target",
    "source_instance",
    "source_type",
    "target_instance",
    "target_type",
    "partner_body_id",
    "partner_instance",
    "partner_type",
    "weight",
    "roi_count",
    "rois",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one MaleCNS neuron, its upstream connections, and its downstream connections."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--body-id",
        type=int,
        help="Exact neuron bodyId, for example 12781 for DNge104_R.",
    )
    selector.add_argument(
        "--query",
        help="Neuron type or instance name, for example DNge104.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def redact_message(message: str, token: str) -> str:
    redacted = message
    if token:
        redacted = redacted.replace(token, "[REDACTED]")
        stripped_token = token.strip()
        if stripped_token:
            redacted = redacted.replace(stripped_token, "[REDACTED]")
    return redacted


def as_dataframe(result: Any) -> pd.DataFrame:
    """Handle both (neurons, roi_counts) and plain DataFrame returns."""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    if isinstance(result, pd.DataFrame):
        return result
    raise RuntimeError("neuprint-python returned an unexpected neuron result type.")


def normalize_neuron_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return normalized
    for column in NEURON_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[NEURON_COLUMNS]


def json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return int(value) if value.is_integer() else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dataframe_to_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: json_value(value) for key, value in row.items()} for _, row in frame.iterrows()]


def metadata_map(neurons: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if neurons.empty:
        return result
    for row in neurons.itertuples(index=False):
        body_id = int(row.bodyId)
        instance = getattr(row, "instance", None)
        neuron_type = getattr(row, "type", None)
        result[body_id] = {
            "instance": str(instance) if pd.notna(instance) else "unknown",
            "type": str(neuron_type) if pd.notna(neuron_type) else "unknown",
        }
    return result


def aggregate_connections(
    connections: pd.DataFrame,
    neurons: pd.DataFrame,
    center_body_id: int,
    direction: str,
) -> pd.DataFrame:
    if connections.empty:
        empty = pd.DataFrame(columns=CONNECTION_COLUMNS)
        return empty

    grouped = (
        connections.groupby(["bodyId_pre", "bodyId_post"], sort=True)
        .agg(
            weight=("weight", "sum"),
            roi_count=("roi", "size"),
            rois=("roi", lambda values: ";".join(sorted(set(str(value) for value in values)))),
        )
        .reset_index()
    )

    meta = metadata_map(neurons)

    def meta_for(body_id: int) -> tuple[str, str]:
        item = meta.get(int(body_id), {})
        return item.get("instance", "unknown"), item.get("type", "unknown")

    records: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        source = int(row.bodyId_pre)
        target = int(row.bodyId_post)
        source_instance, source_type = meta_for(source)
        target_instance, target_type = meta_for(target)
        if direction == "downstream":
            partner = target
        elif direction == "upstream":
            partner = source
        else:
            raise ValueError(f"Unknown direction: {direction}")
        partner_instance, partner_type = meta_for(partner)
        records.append(
            {
                "direction": direction,
                "source": source,
                "target": target,
                "source_instance": source_instance,
                "source_type": source_type,
                "target_instance": target_instance,
                "target_type": target_type,
                "partner_body_id": partner,
                "partner_instance": partner_instance,
                "partner_type": partner_type,
                "weight": int(row.weight),
                "roi_count": int(row.roi_count),
                "rois": row.rois,
            }
        )

    return pd.DataFrame.from_records(records, columns=CONNECTION_COLUMNS)


def fetch_direction(
    client: Client,
    center_body_id: int,
    direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if direction == "downstream":
        sources: list[int] | None = [center_body_id]
        targets: list[int] | None = None
    elif direction == "upstream":
        sources = None
        targets = [center_body_id]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    neurons, connections = fetch_adjacencies(
        sources=sources,
        targets=targets,
        client=client,
        properties=["type", "instance"],
    )
    return neurons, connections


def output_base_name(neuron: pd.Series) -> str:
    body_id = int(neuron["bodyId"])
    instance = neuron.get("instance", None)
    raw_name = str(instance) if pd.notna(instance) and str(instance).strip() else f"neuron_{body_id}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_")
    slug = slug or f"neuron_{body_id}"
    return f"{slug}_{body_id}"


def write_json(data: dict[str, Any], path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, default=json_value) + "\n"
    if token and token in serialized:
        raise RuntimeError("Refusing to write a JSON file that contains the token.")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def print_candidates(candidates: pd.DataFrame) -> None:
    columns = [column for column in NEURON_COLUMNS if column in candidates.columns]
    print("SELECTION_REQUIRED: the query matched multiple neurons.")
    print(candidates[columns].to_string(index=False))
    print("Rerun the script with one exact --body-id from the table above.")


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

        if args.body_id is not None:
            query = args.body_id
        else:
            query = args.query

        neuron_result = fetch_neurons(
            query,
            client=client,
            omit_rois=True,
            returned_columns=NEURON_COLUMNS,
        )
        neurons = normalize_neuron_frame(as_dataframe(neuron_result))

        if neurons.empty:
            print(f"ERROR: no neurons matched query: {query}", file=sys.stderr)
            return 3

        if args.body_id is None and len(neurons) > 1:
            print_candidates(neurons)
            return 4

        if args.body_id is not None:
            exact = neurons[neurons["bodyId"] == args.body_id]
            if exact.empty:
                print(f"ERROR: bodyId {args.body_id} was not found.", file=sys.stderr)
                return 3
            neurons = exact

        neuron = neurons.iloc[0]
        center_body_id = int(neuron["bodyId"])

        upstream_neurons, upstream_connections = fetch_direction(
            client, center_body_id, "upstream"
        )
        downstream_neurons, downstream_connections = fetch_direction(
            client, center_body_id, "downstream"
        )

        upstream_summary = aggregate_connections(
            upstream_connections, upstream_neurons, center_body_id, "upstream"
        )
        downstream_summary = aggregate_connections(
            downstream_connections, downstream_neurons, center_body_id, "downstream"
        )
        connections = pd.concat(
            [upstream_summary, downstream_summary], ignore_index=True
        )
        if not connections.empty:
            connections = connections.sort_values(
                ["direction", "weight", "partner_body_id"],
                ascending=[True, False, True],
                ignore_index=True,
            )

        base_name = output_base_name(neuron)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        neuron_path = args.output_dir / f"{base_name}_neuron.json"
        connections_path = args.output_dir / f"{base_name}_connections.csv"
        report_path = args.output_dir / f"{base_name}_report.json"

        neuron_records = dataframe_to_json_records(neurons)
        report = {
            "status": "ok",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET,
            "query": {"body_id": center_body_id},
            "neuron": neuron_records[0] if neuron_records else None,
            "connection_summary": {
                "upstream_partners": int(len(upstream_summary)),
                "downstream_partners": int(len(downstream_summary)),
                "upstream_total_weight": int(upstream_summary["weight"].sum()) if not upstream_summary.empty else 0,
                "downstream_total_weight": int(downstream_summary["weight"].sum()) if not downstream_summary.empty else 0,
                "upstream_roi_rows": int(len(upstream_connections)),
                "downstream_roi_rows": int(len(downstream_connections)),
            },
            "outputs": {
                "neuron": str(neuron_path),
                "connections": str(connections_path),
                "report": str(report_path),
            },
        }

        neuron_json = {
            "status": "ok",
            "fetched_at": report["fetched_at"],
            "dataset": DATASET,
            "neuron": report["neuron"],
        }

        write_json(neuron_json, neuron_path, token)
        write_json(report, report_path, token)
        connections.to_csv(connections_path, index=False, encoding="utf-8-sig")

        print("OK: neuron query succeeded.")
        print(f"Neuron: {neuron.get('instance', center_body_id)} ({center_body_id})")
        print(f"Upstream partners: {len(upstream_summary)}")
        print(f"Downstream partners: {len(downstream_summary)}")
        print(f"Neuron JSON: {neuron_path}")
        print(f"Connections CSV: {connections_path}")
        print(f"Report JSON: {report_path}")
        return 0

    except requests.exceptions.Timeout as exc:
        print(
            "ERROR: neuPrint request timed out. "
            f"{redact_message(str(exc), token)}",
            file=sys.stderr,
        )
        return 5
    except requests.exceptions.RequestException as exc:
        print(
            "ERROR: could not connect to neuPrint. "
            f"{redact_message(str(exc), token)}",
            file=sys.stderr,
        )
        return 5
    except RuntimeError as exc:
        print(f"ERROR: {redact_message(str(exc), token)}", file=sys.stderr)
        return 6
    except Exception as exc:
        print(
            f"ERROR: unexpected failure: {redact_message(str(exc), token)}",
            file=sys.stderr,
        )
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
