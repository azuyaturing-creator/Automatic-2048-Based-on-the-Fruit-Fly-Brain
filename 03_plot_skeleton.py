"""Fetch and render one MaleCNS neuron skeleton as a PNG."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import navis
import requests
from neuprint import Client, fetch_neurons
from navis.interfaces.neuprint import fetch_skeletons

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
TOKEN_ENV_VAR = "NEUPRINT_APPLICATION_CREDENTIALS"
DEFAULT_OUTPUT_DIR = Path("outputs/skeletons")
DEFAULT_BODY_ID = 12781
METADATA_COLUMNS = ["bodyId", "instance", "type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one MaleCNS neuron skeleton and save it as a PNG."
    )
    parser.add_argument(
        "--body-id",
        type=int,
        default=DEFAULT_BODY_ID,
        help=f"Exact neuron bodyId (default: {DEFAULT_BODY_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the PNG file (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def redact_message(message: str, token: str) -> str:
    if not token:
        return message
    redacted = message.replace(token, "[REDACTED]")
    stripped_token = token.strip()
    if stripped_token:
        redacted = redacted.replace(stripped_token, "[REDACTED]")
    return redacted


def as_dataframe(result: Any):
    """Handle both DataFrame and (neurons, roi_counts) returns."""
    import pandas as pd

    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    if isinstance(result, pd.DataFrame):
        return result
    raise RuntimeError("neuprint-python returned an unexpected metadata result type.")


def normalize_metadata(frame):
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("neuprint-python metadata is not a DataFrame.")
    normalized = frame.copy()
    for column in METADATA_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[METADATA_COLUMNS]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "neuron"


def neuron_label(neuron_metadata) -> str:
    instance = neuron_metadata.get("instance")
    if instance is not None and str(instance).strip():
        return str(instance).strip()
    neuron_type = neuron_metadata.get("type")
    if neuron_type is not None and str(neuron_type).strip():
        return str(neuron_type).strip()
    return "neuron"


def main() -> int:
    args = parse_args()
    token = (
        __import__("os").environ.get(TOKEN_ENV_VAR, "").strip()
    )
    if not token:
        print(
            f"ERROR: {TOKEN_ENV_VAR} is not set.",
            file=sys.stderr,
        )
        print(
            "Set it in your terminal, or on Windows run: "
            "setx NEUPRINT_APPLICATION_CREDENTIALS \"your-token\"",
            file=sys.stderr,
        )
        return 2

    client = Client(SERVER, DATASET)

    try:
        metadata_result = fetch_neurons(
            [args.body_id],
            client=client,
            returned_columns=METADATA_COLUMNS,
        )
        metadata_frame = normalize_metadata(as_dataframe(metadata_result))
        if metadata_frame.empty:
            print(f"ERROR: no neuron found for bodyId {args.body_id}.", file=sys.stderr)
            return 3
        metadata = metadata_frame.iloc[0].to_dict()

        skeletons = fetch_skeletons(
            [args.body_id],
            heal=False,
            parallel=False,
            client=client,
        )
        if skeletons is None or len(skeletons) == 0:
            print(
                f"ERROR: no skeleton found for bodyId {args.body_id}.",
                file=sys.stderr,
            )
            return 3
        if len(skeletons) != 1:
            print(
                f"ERROR: expected one skeleton for bodyId {args.body_id}, "
                f"but received {len(skeletons)}.",
                file=sys.stderr,
            )
            return 6

        skeleton = skeletons[0]
        label = neuron_label(metadata)
        output_name = f"{safe_name(label)}_{args.body_id}_skeleton.png"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / output_name

        fig, ax = navis.plot2d(
            skeleton,
            method="2d",
            view=("x", "-z"),
            soma=True,
            connectors=False,
            linewidth=1.2,
            color="#2563eb",
        )
        try:
            fig.set_size_inches(10, 8)
            fig.suptitle(f"MaleCNS skeleton · {label} · bodyId {args.body_id}")
            ax.set_title(DATASET)
            ax.set_xlabel("x")
            ax.set_ylabel("-z")
            fig.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
        finally:
            plt.close(fig)

        soma_value = skeleton.soma
        soma_count = (
            len(soma_value)
            if hasattr(soma_value, "__len__") and not isinstance(soma_value, str)
            else (1 if soma_value is not None else 0)
        )
        print("OK: skeleton rendered.")
        print(f"Neuron: {label} ({args.body_id})")
        print(f"Nodes: {len(skeleton.nodes)}")
        print(f"Branch nodes: {skeleton.n_branches}")
        print(f"Leaf nodes: {skeleton.n_leafs}")
        print(f"Soma nodes: {soma_count}")
        print(f"PNG: {output_path}")
        return 0

    except requests.exceptions.Timeout as exc:
        print(
            "ERROR: neuPrint request timed out. " f"{redact_message(str(exc), token)}",
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






