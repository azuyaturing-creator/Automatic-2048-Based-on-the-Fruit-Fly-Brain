"""Run a reproducible Brian2 LIF simulation from a MaleCNS subgraph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import brian2 as b2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

DEFAULT_NODES = Path("outputs/subgraphs/DNge104_R_12781_nodes.csv")
DEFAULT_EDGES = Path("outputs/subgraphs/DNge104_R_12781_edges.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/simulations")
DEFAULT_DURATION_MS = 500.0
DEFAULT_DT_MS = 0.1
DEFAULT_INPUT_COUNT = 20
DEFAULT_INPUT_RATE_HZ = 40.0
DEFAULT_SEED = 20260908

TAU_MS = 20.0
V_REST_MV = -70.0
V_RESET_MV = -75.0
V_THRESHOLD_MV = -55.0
REFRACTORY_MS = 2.0
BASE_SYNAPTIC_WEIGHT_MV = 1.5
STIMULUS_WEIGHT_MV = 15.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer.")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic Brian2 LIF simulation from a MaleCNS subgraph."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=DEFAULT_NODES,
        help=f"Node CSV from stage 4 (default: {DEFAULT_NODES}).",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=DEFAULT_EDGES,
        help=f"Edge CSV from stage 4 (default: {DEFAULT_EDGES}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for simulation outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--name",
        help="Output base name. Defaults to the center neuron name and bodyId.",
    )
    parser.add_argument(
        "--duration-ms",
        type=positive_float,
        default=DEFAULT_DURATION_MS,
        help=f"Simulation duration in milliseconds (default: {DEFAULT_DURATION_MS}).",
    )
    parser.add_argument(
        "--dt-ms",
        type=positive_float,
        default=DEFAULT_DT_MS,
        help=f"Simulation time step in milliseconds (default: {DEFAULT_DT_MS}).",
    )
    parser.add_argument(
        "--input-count",
        type=positive_int,
        default=DEFAULT_INPUT_COUNT,
        help=f"Number of strongest upstream neurons to stimulate (default: {DEFAULT_INPUT_COUNT}).",
    )
    parser.add_argument(
        "--input-rate-hz",
        type=positive_float,
        default=DEFAULT_INPUT_RATE_HZ,
        help=f"Poisson sensory input rate in Hz (default: {DEFAULT_INPUT_RATE_HZ}).",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=DEFAULT_SEED,
        help=f"Deterministic random seed (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "simulation"


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return int(value) if value.is_integer() else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: " + ", ".join(missing)
        )
    return frame


def infer_name(nodes: pd.DataFrame) -> str:
    center = nodes[nodes["is_center"].astype(bool)]
    if center.empty:
        raise ValueError("node table has no center neuron.")
    row = center.iloc[0]
    body_id = int(row["body_id"])
    instance = str(row.get("instance", "neuron"))
    label = instance if instance.strip() and instance.lower() != "nan" else "neuron"
    return f"{safe_name(label)}_{body_id}"


def validate_subgraph(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    node_ids = set(nodes["body_id"].astype(int).tolist())
    center_count = int(nodes["is_center"].astype(bool).sum())
    duplicate_pairs = int(edges.duplicated(["source", "target"]).sum())
    endpoints_exist = bool(
        edges["source"].astype(int).isin(node_ids).all()
        and edges["target"].astype(int).isin(node_ids).all()
    )
    weights_positive = bool((edges["weight"] > 0).all())
    return {
        "node_count_positive": int(len(nodes)) > 0,
        "edge_count_positive": int(len(edges)) > 0,
        "one_center_neuron": center_count == 1,
        "all_edge_endpoints_exist": endpoints_exist,
        "duplicate_source_target_edges": duplicate_pairs,
        "all_weights_positive": weights_positive,
    }


def normalize_weights(raw_weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_weights, dtype=float)
    log_weights = np.log1p(raw)
    mean_log = float(log_weights.mean())
    if not math.isfinite(mean_log) or mean_log <= 0:
        raise ValueError("log-transformed connection weights have no positive finite mean.")
    normalized = log_weights / mean_log
    effective_mv = normalized * BASE_SYNAPTIC_WEIGHT_MV
    details = {
        "formula": "log1p(raw_weight) / mean(log1p(raw_weight)) * base_synaptic_weight",
        "base_synaptic_weight_mv": BASE_SYNAPTIC_WEIGHT_MV,
        "raw_weight_min": float(raw.min()),
        "raw_weight_max": float(raw.max()),
        "raw_weight_mean": float(raw.mean()),
        "log_weight_min": float(log_weights.min()),
        "log_weight_max": float(log_weights.max()),
        "log_weight_mean": mean_log,
        "normalized_weight_min": float(normalized.min()),
        "normalized_weight_max": float(normalized.max()),
        "effective_synaptic_weight_min_mv": float(effective_mv.min()),
        "effective_synaptic_weight_max_mv": float(effective_mv.max()),
    }
    return normalized, details


def select_input_targets(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    center_body_id: int,
    input_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    upstream = edges[edges["target"] == center_body_id].copy()
    if upstream.empty:
        raise ValueError("edge table has no upstream connections to the center neuron.")
    upstream = upstream.sort_values(
        ["weight", "source"], ascending=[False, True]
    ).head(input_count)
    input_body_ids = upstream["source"].astype(int).tolist()
    body_id_to_index = {
        int(body_id): index for index, body_id in enumerate(nodes["body_id"].astype(int))
    }
    input_indices = np.asarray(
        [body_id_to_index[int(body_id)] for body_id in input_body_ids], dtype=int
    )
    metadata = nodes.set_index(nodes["body_id"].astype(int))
    records: list[dict[str, Any]] = []
    for row_index, body_id in zip(input_indices, input_body_ids):
        row = metadata.loc[int(body_id)]
        records.append(
            {
                "neuron_index": int(row_index),
                "body_id": int(body_id),
                "instance": str(row["instance"]),
                "type": str(row["type"]),
            }
        )
    return input_indices, records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )


def write_spikes(
    path: Path,
    indices: np.ndarray,
    times_ms: np.ndarray,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    order = np.lexsort((indices, times_ms))
    sorted_indices = np.asarray(indices, dtype=int)[order]
    sorted_times = np.asarray(times_ms, dtype=float)[order]
    metadata = nodes.iloc[sorted_indices][["body_id", "instance"]].reset_index(drop=True)
    spikes = pd.DataFrame(
        {
            "spike_id": range(len(sorted_indices)),
            "neuron_index": sorted_indices,
            "body_id": metadata["body_id"].astype(int).to_numpy(),
            "instance": metadata["instance"].astype(str).to_numpy(),
            "t_ms": sorted_times,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    spikes.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    return spikes


def plot_raster(
    path: Path,
    spikes: pd.DataFrame,
    nodes: pd.DataFrame,
    duration_ms: float,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    if spikes.empty:
        ax.text(0.5, 0.5, "No spikes", ha="center", va="center", transform=ax.transAxes)
    else:
        is_hub = nodes["is_hub"].astype(bool).to_numpy()
        center_index = int(nodes.index[nodes["is_center"].astype(bool)][0])
        hub_indices = set(nodes.index[is_hub].tolist())

        groups = [
            ("Center", "#dc2626", [center_index]),
            ("Other hubs", "#f59e0b", sorted(hub_indices - {center_index})),
            ("Other neurons", "#2563eb", sorted(set(range(len(nodes))) - hub_indices)),
        ]
        for label, color, neuron_indices in groups:
            selected = set(neuron_indices)
            mask = spikes["neuron_index"].isin(selected)
            if bool(mask.any()):
                subset = spikes[mask]
                ax.plot(
                    subset["t_ms"],
                    subset["neuron_index"],
                    linestyle="none",
                    marker="|",
                    markersize=5,
                    color=color,
                    label=label,
                )

    ax.set_xlim(0, duration_ms)
    ax.set_ylim(-1, len(nodes))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if not spikes.empty:
        ax.legend(loc="upper right", frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        nodes = load_csv(
            args.nodes,
            {"body_id", "instance", "type", "is_center", "is_hub"},
        )
        edges = load_csv(args.edges, {"source", "target", "weight"})
        validation = validate_subgraph(nodes, edges)
        if not all(
            [
                validation["node_count_positive"],
                validation["edge_count_positive"],
                validation["one_center_neuron"],
                validation["all_edge_endpoints_exist"],
                validation["duplicate_source_target_edges"] == 0,
                validation["all_weights_positive"],
            ]
        ):
            raise ValueError(f"invalid subgraph: {validation}")

        center = nodes[nodes["is_center"].astype(bool)].iloc[0]
        center_body_id = int(center["body_id"])
        name = safe_name(args.name) if args.name else infer_name(nodes)

        body_id_to_index = {
            int(body_id): index
            for index, body_id in enumerate(nodes["body_id"].astype(int))
        }
        source_indices = np.asarray(
            [body_id_to_index[int(value)] for value in edges["source"]],
            dtype=int,
        )
        target_indices = np.asarray(
            [body_id_to_index[int(value)] for value in edges["target"]],
            dtype=int,
        )
        normalized_weights, weight_details = normalize_weights(edges["weight"].to_numpy())
        input_indices, input_targets = select_input_targets(
            nodes, edges, center_body_id, args.input_count
        )

        b2.start_scope()
        b2.prefs.codegen.target = "numpy"
        b2.defaultclock.dt = args.dt_ms * b2.ms
        np.random.seed(args.seed)
        b2.seed(args.seed)

        lif_equations = "dv/dt = (v_rest - v) / tau : volt (unless refractory)"
        neurons = b2.NeuronGroup(
            len(nodes),
            lif_equations,
            threshold="v > v_threshold",
            reset="v = v_reset",
            refractory=REFRACTORY_MS * b2.ms,
            method="exact",
            namespace={
                "v_rest": V_REST_MV * b2.mV,
                "v_reset": V_RESET_MV * b2.mV,
                "v_threshold": V_THRESHOLD_MV * b2.mV,
                "tau": TAU_MS * b2.ms,
            },
        )
        neurons.v = V_RESET_MV * b2.mV

        synapses = b2.Synapses(
            neurons,
            neurons,
            model="w : volt",
            on_pre="v += w",
        )
        synapses.connect(i=source_indices, j=target_indices)
        synapses.w = normalized_weights * BASE_SYNAPTIC_WEIGHT_MV * b2.mV

        sensory_input = b2.PoissonGroup(
            len(input_indices), rates=args.input_rate_hz * b2.Hz
        )
        input_synapses = b2.Synapses(
            sensory_input,
            neurons,
            model="w : volt",
            on_pre="v += w",
        )
        input_synapses.connect(i=np.arange(len(input_indices)), j=input_indices)
        input_synapses.w = STIMULUS_WEIGHT_MV * b2.mV

        spike_monitor = b2.SpikeMonitor(neurons)
        b2.run(args.duration_ms * b2.ms)

        spike_indices = np.asarray(spike_monitor.i, dtype=int)
        spike_times_ms = np.asarray(spike_monitor.t / b2.ms, dtype=float)

        output_dir = args.output_dir
        spikes_path = output_dir / f"{name}_spikes.csv"
        raster_path = output_dir / f"{name}_raster.png"
        summary_path = output_dir / f"{name}_summary.json"
        spikes = write_spikes(spikes_path, spike_indices, spike_times_ms, nodes)
        plot_raster(
            raster_path,
            spikes,
            nodes,
            args.duration_ms,
            f"Brian2 LIF raster · {name} · seed {args.seed}",
        )

        active_neurons = int(spikes["neuron_index"].nunique()) if not spikes.empty else 0
        spike_validation = {
            "spike_count_positive": int(len(spikes)) > 0,
            "spike_times_nonnegative": bool(
                spikes.empty or (spikes["t_ms"] >= 0).all()
            ),
            "spike_times_nondecreasing": bool(
                spikes.empty or (np.diff(spikes["t_ms"].to_numpy()) >= 0).all()
            ),
            "raster_file_created": raster_path.is_file(),
            "spikes_file_created": spikes_path.is_file(),
        }
        if not all(spike_validation.values()):
            raise RuntimeError(f"simulation output validation failed: {spike_validation}")

        activity_counts = (
            spikes.groupby("neuron_index").size().reindex(range(len(nodes)), fill_value=0)
            if not spikes.empty
            else pd.Series(0, index=range(len(nodes)))
        )
        firing_rates_hz = activity_counts / (args.duration_ms / 1000.0)
        active_body_ids = spikes["body_id"].astype(int).unique().tolist() if not spikes.empty else []

        summary = {
            "status": "ok",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "nodes": str(args.nodes),
                "edges": str(args.edges),
            },
            "query": {
                "name": str(center["instance"]),
                "center_body_id": center_body_id,
            },
            "parameters": {
                "seed": int(args.seed),
                "duration_ms": float(args.duration_ms),
                "dt_ms": float(args.dt_ms),
                "input_count": int(args.input_count),
                "input_rate_hz": float(args.input_rate_hz),
                "input_selection": "strongest upstream edges to the center neuron",
                "lif": {
                    "tau_ms": TAU_MS,
                    "v_rest_mv": V_REST_MV,
                    "v_reset_mv": V_RESET_MV,
                    "v_threshold_mv": V_THRESHOLD_MV,
                    "refractory_ms": REFRACTORY_MS,
                },
                "stimulus_weight_mv": STIMULUS_WEIGHT_MV,
            },
            "mapping": {
                "neuron_group_size": int(len(nodes)),
                "connection_synapse_count": int(len(synapses)),
                "expected_connection_synapse_count": int(len(edges)),
                "sensory_poisson_neuron_count": int(len(input_indices)),
                "sensory_synapse_count": int(len(input_synapses)),
            },
            "weight_transform": weight_details,
            "input_targets": input_targets,
            "activity": {
                "spike_count": int(len(spikes)),
                "active_neuron_count": active_neurons,
                "active_body_ids": active_body_ids,
                "mean_firing_rate_hz": float(firing_rates_hz.mean()),
                "max_firing_rate_hz": float(firing_rates_hz.max()),
                "first_spike_time_ms": float(spikes["t_ms"].min()) if not spikes.empty else None,
                "last_spike_time_ms": float(spikes["t_ms"].max()) if not spikes.empty else None,
            },
            "subgraph_validation": validation,
            "simulation_validation": spike_validation,
            "outputs": {
                "spikes": str(spikes_path),
                "raster": str(raster_path),
                "summary": str(summary_path),
                "spikes_sha256": sha256_file(spikes_path),
            },
        }
        write_json(summary, summary_path)

        print("OK: Brian2 LIF simulation completed.")
        print(f"Neurons: {len(nodes)}")
        print(f"Connection synapses: {len(synapses)}")
        print(f"Sensory input neurons: {len(input_indices)}")
        print(f"Spikes: {len(spikes)}")
        print(f"Active neurons: {active_neurons}")
        print(f"Spikes CSV: {spikes_path}")
        print(f"Raster PNG: {raster_path}")
        print(f"Summary JSON: {summary_path}")
        return 0

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, IndexError) as exc:
        print(f"ERROR: invalid input or mapping: {exc}", file=sys.stderr)
        return 3
    except b2.BrianObjectException as exc:
        print(f"ERROR: Brian2 simulation failed: {exc}", file=sys.stderr)
        return 4
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"ERROR: unexpected failure: {exc}", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())



