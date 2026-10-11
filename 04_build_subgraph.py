"""Build a size-controlled MaleCNS neuron-level subgraph for later LIF simulation."""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_OUTPUT_DIR = Path("outputs/subgraphs")
DEFAULT_BODY_ID = 12781
DEFAULT_MAX_NODES = 200
METADATA_COLUMNS = ["bodyId", "instance", "type"]
RAW_EDGE_COLUMNS = ["source", "target", "weight"]
EDGE_COLUMNS = [
    "source",
    "target",
    "weight",
    "source_instance",
    "source_type",
    "target_instance",
    "target_type",
]
NODE_COLUMNS = [
    "body_id",
    "instance",
    "type",
    "is_center",
    "in_degree",
    "out_degree",
    "total_degree",
    "input_weight",
    "output_weight",
    "strength",
    "degree_rank",
    "is_hub",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MaleCNS direct-connection subgraph capped by node count."
    )
    parser.add_argument(
        "--body-id",
        type=int,
        default=DEFAULT_BODY_ID,
        help=f"Exact starting neuron bodyId (default: {DEFAULT_BODY_ID}).",
    )
    parser.add_argument(
        "--max-nodes",
        type=positive_int,
        default=DEFAULT_MAX_NODES,
        help=f"Maximum number of nodes, including the center (default: {DEFAULT_MAX_NODES}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for subgraph files (default: {DEFAULT_OUTPUT_DIR}).",
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
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    if isinstance(result, pd.DataFrame):
        return result
    raise RuntimeError("neuprint-python returned an unexpected metadata result type.")


def normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in METADATA_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[METADATA_COLUMNS]


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


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: json_value(value) for key, value in row.items()} for _, row in frame.iterrows()]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "neuron"


def neuron_label(metadata: pd.Series) -> str:
    instance = metadata.get("instance")
    if instance is not None and pd.notna(instance) and str(instance).strip():
        return str(instance).strip()
    neuron_type = metadata.get("type")
    if neuron_type is not None and pd.notna(neuron_type) and str(neuron_type).strip():
        return str(neuron_type).strip()
    return "neuron"


def fetch_center_metadata(client: Client, body_id: int) -> pd.Series | None:
    result = fetch_neurons(
        [body_id],
        client=client,
        returned_columns=METADATA_COLUMNS,
    )
    frame = normalize_metadata(as_dataframe(result))
    if frame.empty:
        return None
    if len(frame) != 1:
        raise RuntimeError(
            f"Expected one metadata row for bodyId {body_id}, but received {len(frame)}."
        )
    return frame.iloc[0]


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
        omit_rois=True,
    )
    return as_dataframe(neurons), as_dataframe(connections)


def aggregate_direction_edges(
    connections: pd.DataFrame,
    center_body_id: int,
    direction: str,
) -> pd.DataFrame:
    if connections.empty:
        return pd.DataFrame(columns=RAW_EDGE_COLUMNS)

    required = {"bodyId_pre", "bodyId_post", "weight"}
    missing = required.difference(connections.columns)
    if missing:
        raise RuntimeError(
            "neuprint-python connection table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    grouped = (
        connections.groupby(["bodyId_pre", "bodyId_post"], as_index=False)
        .agg(weight=("weight", "sum"))
    )
    grouped["weight"] = grouped["weight"].astype(int)

    if direction == "upstream":
        grouped = grouped[grouped["bodyId_post"] == center_body_id]
    elif direction == "downstream":
        grouped = grouped[grouped["bodyId_pre"] == center_body_id]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    edges = pd.DataFrame(
        {
            "source": grouped["bodyId_pre"].astype(int),
            "target": grouped["bodyId_post"].astype(int),
            "weight": grouped["weight"].astype(int),
        }
    )
    return edges[RAW_EDGE_COLUMNS].reset_index(drop=True)


def candidate_partners(
    edges: pd.DataFrame,
    center_body_id: int,
    direction: str,
    limit: int,
) -> list[int]:
    if limit <= 0 or edges.empty:
        return []

    if direction == "upstream":
        candidates = edges[edges["target"] == center_body_id].copy()
        candidates["partner"] = candidates["source"]
    elif direction == "downstream":
        candidates = edges[edges["source"] == center_body_id].copy()
        candidates["partner"] = candidates["target"]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    candidates = candidates[candidates["partner"] != center_body_id]
    if candidates.empty:
        return []

    grouped = (
        candidates.groupby("partner", as_index=False)
        .agg(weight=("weight", "sum"))
        .sort_values(["weight", "partner"], ascending=[False, True])
    )
    return [int(value) for value in grouped["partner"].head(limit).tolist()]


def select_partner_ids(
    upstream_edges: pd.DataFrame,
    downstream_edges: pd.DataFrame,
    center_body_id: int,
    max_nodes: int,
) -> tuple[list[int], list[int]]:
    partner_slots = max(0, max_nodes - 1)
    if partner_slots == 0:
        return [], []

    upstream_limit = (partner_slots + 1) // 2
    downstream_limit = partner_slots // 2

    upstream_available = len(candidate_partners(upstream_edges, center_body_id, "upstream", partner_slots))
    downstream_available = len(
        candidate_partners(downstream_edges, center_body_id, "downstream", partner_slots)
    )

    if upstream_available < upstream_limit:
        downstream_limit += upstream_limit - upstream_available
    elif downstream_available < downstream_limit:
        upstream_limit += downstream_limit - downstream_available

    upstream_limit = min(upstream_limit, partner_slots)
    downstream_limit = min(downstream_limit, partner_slots)

    upstream_ids = candidate_partners(
        upstream_edges, center_body_id, "upstream", upstream_limit
    )
    downstream_ids = candidate_partners(
        downstream_edges, center_body_id, "downstream", downstream_limit
    )
    return upstream_ids, downstream_ids


def metadata_map(frames: list[pd.DataFrame]) -> dict[int, dict[str, str]]:
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result: dict[int, dict[str, str]] = {}
    if combined.empty:
        return result

    for _, row in normalize_metadata(combined).iterrows():
        if pd.isna(row["bodyId"]):
            continue
        body_id = int(row["bodyId"])
        instance = row["instance"]
        neuron_type = row["type"]
        result[body_id] = {
            "instance": str(instance).strip() if pd.notna(instance) and str(instance).strip() else "unknown",
            "type": str(neuron_type).strip() if pd.notna(neuron_type) and str(neuron_type).strip() else "unknown",
        }
    return result


def build_edges(
    upstream_edges: pd.DataFrame,
    downstream_edges: pd.DataFrame,
    selected_ids: set[int],
    metadata: dict[int, dict[str, str]],
) -> pd.DataFrame:
    combined = pd.concat([upstream_edges, downstream_edges], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    combined = combined[
        combined["source"].isin(selected_ids) & combined["target"].isin(selected_ids)
    ]
    if combined.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    grouped = (
        combined.groupby(["source", "target"], as_index=False)
        .agg(weight=("weight", "sum"))
        .sort_values(["source", "target"])
    )
    grouped["weight"] = grouped["weight"].astype(int)

    def meta(body_id: int, field: str) -> str:
        return metadata.get(int(body_id), {}).get(field, "unknown")

    edges = pd.DataFrame(
        {
            "source": grouped["source"].astype(int),
            "target": grouped["target"].astype(int),
            "weight": grouped["weight"].astype(int),
            "source_instance": [meta(value, "instance") for value in grouped["source"]],
            "source_type": [meta(value, "type") for value in grouped["source"]],
            "target_instance": [meta(value, "instance") for value in grouped["target"]],
            "target_type": [meta(value, "type") for value in grouped["target"]],
        }
    )
    return edges[EDGE_COLUMNS].reset_index(drop=True)


def build_nodes(
    selected_ids: set[int],
    center_body_id: int,
    edges: pd.DataFrame,
    metadata: dict[int, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for body_id in sorted(selected_ids):
        incoming = edges[edges["target"] == body_id]
        outgoing = edges[edges["source"] == body_id]
        input_weight = int(incoming["weight"].sum()) if not incoming.empty else 0
        output_weight = int(outgoing["weight"].sum()) if not outgoing.empty else 0
        rows.append(
            {
                "body_id": int(body_id),
                "instance": metadata.get(int(body_id), {}).get("instance", "unknown"),
                "type": metadata.get(int(body_id), {}).get("type", "unknown"),
                "is_center": bool(body_id == center_body_id),
                "in_degree": int(len(incoming)),
                "out_degree": int(len(outgoing)),
                "total_degree": int(len(incoming) + len(outgoing)),
                "input_weight": input_weight,
                "output_weight": output_weight,
                "strength": int(input_weight + output_weight),
                "degree_rank": 0,
                "is_hub": False,
            }
        )

    nodes = pd.DataFrame.from_records(rows, columns=NODE_COLUMNS)
    if nodes.empty:
        return nodes

    ordered = nodes.sort_values(
        ["total_degree", "strength", "body_id"],
        ascending=[False, False, True],
    ).copy()
    ordered["degree_rank"] = range(1, len(ordered) + 1)
    hub_count = max(1, int(math.ceil(len(ordered) * 0.1)))
    ordered["is_hub"] = ordered["degree_rank"] <= hub_count
    return ordered.sort_values("body_id").reset_index(drop=True)


def validate_subgraph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    max_nodes: int,
) -> dict[str, Any]:
    node_ids = set(nodes["body_id"].astype(int).tolist())
    edge_pairs = list(
        zip(edges["source"].astype(int), edges["target"].astype(int))
    )
    return {
        "node_count_within_limit": int(len(nodes)) <= max_nodes,
        "edge_count_positive": int(len(edges)) > 0,
        "all_edge_endpoints_exist": bool(
            set(edges["source"].astype(int)).issubset(node_ids)
            and set(edges["target"].astype(int)).issubset(node_ids)
        ),
        "duplicate_source_target_edges": int(len(edge_pairs) - len(set(edge_pairs))),
        "all_weights_positive": bool((edges["weight"] > 0).all()) if not edges.empty else False,
    }


def write_json(data: dict[str, Any], path: Path, token: str) -> None:
    serialized = json.dumps(data, ensure_ascii=False, indent=2, default=json_value) + "\n"
    if token and token in serialized:
        raise RuntimeError("Refusing to write a JSON file that contains the token.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> int:
    args = parse_args()
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        print(f"ERROR: {TOKEN_ENV_VAR} is not set.", file=sys.stderr)
        print(
            "Set it in your terminal, or on Windows run: "
            'setx NEUPRINT_APPLICATION_CREDENTIALS "your-token"',
            file=sys.stderr,
        )
        return 2

    if args.max_nodes < 3:
        print("ERROR: --max-nodes must be at least 3.", file=sys.stderr)
        return 4

    client = Client(SERVER, DATASET)

    try:
        center_metadata = fetch_center_metadata(client, args.body_id)
        if center_metadata is None:
            print(f"ERROR: no neuron found for bodyId {args.body_id}.", file=sys.stderr)
            return 3

        upstream_neurons, upstream_connections = fetch_direction(
            client, args.body_id, "upstream"
        )
        downstream_neurons, downstream_connections = fetch_direction(
            client, args.body_id, "downstream"
        )

        upstream_edges = aggregate_direction_edges(
            upstream_connections, args.body_id, "upstream"
        )
        downstream_edges = aggregate_direction_edges(
            downstream_connections, args.body_id, "downstream"
        )
        if upstream_edges.empty and downstream_edges.empty:
            print(
                f"ERROR: no direct connections found for bodyId {args.body_id}.",
                file=sys.stderr,
            )
            return 4

        upstream_ids, downstream_ids = select_partner_ids(
            upstream_edges, downstream_edges, args.body_id, args.max_nodes
        )
        selected_ids = {args.body_id, *upstream_ids, *downstream_ids}

        metadata = metadata_map(
            [
                pd.DataFrame([center_metadata.to_dict()]),
                upstream_neurons,
                downstream_neurons,
            ]
        )
        edges = build_edges(upstream_edges, downstream_edges, selected_ids, metadata)
        nodes = build_nodes(selected_ids, args.body_id, edges, metadata)
        if nodes.empty or edges.empty:
            print("ERROR: selected subgraph has no usable nodes or edges.", file=sys.stderr)
            return 4

        validation = validate_subgraph(nodes, edges, args.max_nodes)
        if not all(
            [
                validation["node_count_within_limit"],
                validation["edge_count_positive"],
                validation["all_edge_endpoints_exist"],
                validation["duplicate_source_target_edges"] == 0,
                validation["all_weights_positive"],
            ]
        ):
            raise RuntimeError(f"Subgraph validation failed: {validation}")

        label = neuron_label(center_metadata)
        base_name = f"{safe_name(label)}_{args.body_id}"
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        subgraph_path = output_dir / f"{base_name}_subgraph.json"
        nodes_path = output_dir / f"{base_name}_nodes.csv"
        edges_path = output_dir / f"{base_name}_edges.csv"
        summary_path = output_dir / f"{base_name}_summary.json"

        fetched_at = datetime.now(timezone.utc).isoformat()
        parameters = {
            "max_nodes": int(args.max_nodes),
            "selection_strategy": "center plus top-weight direct upstream and downstream partners",
            "partner_allocation": "balanced, with unused slots moved to the other direction",
            "edge_aggregation": "sum of ROI-level neuPrint weights per source-target pair",
            "hub_rule": "top 10% of nodes by total_degree, then strength, then body_id",
        }
        outputs = {
            "subgraph": str(subgraph_path),
            "nodes": str(nodes_path),
            "edges": str(edges_path),
            "summary": str(summary_path),
        }

        subgraph = {
            "status": "ok",
            "fetched_at": fetched_at,
            "dataset": DATASET,
            "query": {"body_id": int(args.body_id), "name": label},
            "parameters": parameters,
            "nodes": records(nodes),
            "edges": records(edges),
            "validation": validation,
            "outputs": outputs,
        }

        hubs = nodes[nodes["is_hub"]].sort_values("degree_rank")
        summary = {
            "status": "ok",
            "fetched_at": fetched_at,
            "dataset": DATASET,
            "query": {"body_id": int(args.body_id), "name": label},
            "parameters": parameters,
            "counts": {
                "nodes": int(len(nodes)),
                "edges": int(len(edges)),
                "upstream_partners_selected": int(len(upstream_ids)),
                "downstream_partners_selected": int(len(downstream_ids)),
                "upstream_edges_selected": int(
                    (edges["target"] == args.body_id).sum()
                ),
                "downstream_edges_selected": int(
                    (edges["source"] == args.body_id).sum()
                ),
            },
            "hub_neurons": records(
                hubs[
                    [
                        "body_id",
                        "instance",
                        "type",
                        "in_degree",
                        "out_degree",
                        "total_degree",
                        "input_weight",
                        "output_weight",
                        "strength",
                        "degree_rank",
                    ]
                ]
            ),
            "validation": validation,
            "outputs": outputs,
        }

        write_json(subgraph, subgraph_path, token)
        write_json(summary, summary_path, token)
        nodes.to_csv(nodes_path, index=False, encoding="utf-8-sig")
        edges.to_csv(edges_path, index=False, encoding="utf-8-sig")

        print("OK: subgraph built.")
        print(f"Neuron: {label} ({args.body_id})")
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        print(f"Upstream partners selected: {len(upstream_ids)}")
        print(f"Downstream partners selected: {len(downstream_ids)}")
        print(f"Hubs: {int(nodes['is_hub'].sum())}")
        print(f"Subgraph JSON: {subgraph_path}")
        print(f"Nodes CSV: {nodes_path}")
        print(f"Edges CSV: {edges_path}")
        print(f"Summary JSON: {summary_path}")
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
