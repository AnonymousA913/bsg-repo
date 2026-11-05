from __future__ import annotations

"""Resumable evaluation on public FJSP benchmarks for Section 5.3.2."""

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    TimingTotals,
    evaluate_batch,
    load_evaluator,
    make_environment,
    read_csv,
    seed_everything,
    validate_schedule,
    write_csv_atomic,
    write_json_atomic,
)
from CPNModelV2.experiments.sec53_generalization.public_benchmark_core import (
    PUBLIC_PROTOCOL_VERSION,
    PublicBenchmarkEvaluationTask,
    bks_makespans,
    checkpoint_audit,
    deterministic_evaluation_seed,
    initialize_output,
    or_tools_makespans,
    ordered_records,
    public_error_rows,
    sha256_file,
    validate_resumed_rows,
)
from CPNModelV2.integration.equivalence_core import CanonicalInstance


GREEDY_FIELDS = (
    "source_distribution", "source_size", "benchmark", "training_seed",
    "instance_id", "decoding", "makespan", "bks_makespan",
    "or_tools_makespan", "rpd_bks_percent", "gap_or_tools_percent",
    "decision_count", "feasible", "evaluation_seed",
)
SAMPLING_FIELDS = (
    "source_distribution", "source_size", "benchmark", "training_seed",
    "instance_id", "rollout_id", "makespan", "bks_makespan",
    "or_tools_makespan", "rpd_bks_percent", "gap_or_tools_percent",
    "decision_count", "feasible", "evaluation_seed",
)
SAMPLING_SUMMARY_FIELDS = (
    "source_distribution", "source_size", "benchmark", "training_seed",
    "instance_id", "best_makespan", "best_rollout_id", "bks_makespan",
    "or_tools_makespan", "rpd_bks_percent", "gap_or_tools_percent",
    "feasible_rollouts",
)
ERROR_FIELDS = (
    "source_distribution", "source_size", "benchmark", "training_seed",
    "instance_id", "decoder", "rollout_id", "error",
)
TIMING_FIELDS = (
    "decoder", "batch_id", "instance_id", "rollout_start", "rollout_end",
    "schedules", "end_to_end_s", "seconds_per_schedule",
    "tensorization_s", "network_inference_s", "action_selection_s",
    "environment_step_and_graph_s",
)


def _relative_gap(value: float, reference: float) -> float:
    return (value - reference) / reference * 100.0


def _timing_row(
    decoder: str,
    batch_id: int,
    timing: TimingTotals,
    *,
    instance_id: str = "",
    rollout_ids: Sequence[int] = (),
) -> Dict[str, Any]:
    values = timing.to_dict()
    schedules = int(values["schedules"])
    return {
        "decoder": decoder,
        "batch_id": batch_id,
        "instance_id": instance_id,
        "rollout_start": min(rollout_ids) if rollout_ids else "",
        "rollout_end": max(rollout_ids) if rollout_ids else "",
        "schedules": schedules,
        "end_to_end_s": values["end_to_end_s"],
        "seconds_per_schedule": values["end_to_end_s"] / schedules,
        "tensorization_s": values["tensorization_s"],
        "network_inference_s": values["network_inference_s"],
        "action_selection_s": values["action_selection_s"],
        "environment_step_and_graph_s": values[
            "environment_step_and_graph_s"
        ],
    }


def _retained_rows(path: Path, decoder: str, overwrite: bool) -> List[Dict[str, str]]:
    rows = read_csv(path)
    if overwrite:
        return [row for row in rows if row.get("decoder") != decoder]
    return rows


def _metadata_base(
    task: PublicBenchmarkEvaluationTask,
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "evaluation_protocol_version": PUBLIC_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "source_distribution": task.source_distribution,
        "source_size": task.source_size,
        "benchmark": task.benchmark,
        "training_seed": task.training_seed,
        "checkpoint_label": task.checkpoint_label,
        "checkpoint": str(task.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(task.checkpoint),
        "dataset_dir": str(task.dataset_dir.resolve()),
        "schema_refit_on_target": False,
        "normalizer_refit_on_target": False,
        "parameter_updates_on_target": False,
        **checkpoint_audit(task, checkpoint),
    }


def _quality_metrics(
    values: Sequence[float],
    bks_values: Sequence[float],
    or_values: Sequence[float],
) -> Dict[str, Any]:
    rpds = np.asarray([
        _relative_gap(value, reference)
        for value, reference in zip(values, bks_values)
    ])
    return {
        "mean_makespan": mean(values),
        "std_instance_makespan": float(np.std(values, ddof=0)),
        "mean_bks_makespan": mean(bks_values),
        "mean_or_tools_makespan": mean(or_values),
        "mean_instance_rpd_bks_percent": float(rpds.mean()),
        "median_instance_rpd_bks_percent": float(np.median(rpds)),
        "p90_instance_rpd_bks_percent": float(np.quantile(rpds, 0.9)),
        "max_instance_rpd_bks_percent": float(rpds.max()),
        "aggregate_gap_bks_percent": _relative_gap(
            mean(values), mean(bks_values)
        ),
        "mean_instance_gap_or_tools_percent": mean([
            _relative_gap(value, reference)
            for value, reference in zip(values, or_values)
        ]),
        "bks_or_better_instances": sum(
            value <= reference for value, reference in zip(values, bks_values)
        ),
    }


def _update_evaluation_metadata(
    task: PublicBenchmarkEvaluationTask,
    decoder: str,
    decoder_metadata: Mapping[str, Any],
) -> None:
    path = task.output_dir / "evaluation_metadata.json"
    existing: Dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update({
        "evaluation_protocol_version": PUBLIC_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "source_distribution": task.source_distribution,
        "source_size": task.source_size,
        "benchmark": task.benchmark,
        "training_seed": task.training_seed,
        "checkpoint_label": task.checkpoint_label,
    })
    decoders = dict(existing.get("decoders", {}))
    decoders[decoder] = dict(decoder_metadata)
    existing["decoders"] = decoders
    write_json_atomic(path, existing)


def run_greedy(
    task: PublicBenchmarkEvaluationTask,
    *,
    overwrite: bool = False,
) -> Path:
    initialize_output(task)
    seed_everything(task.training_seed)
    output_path = task.output_dir / "greedy_per_instance.csv"
    error_path = task.output_dir / "feasibility_errors.csv"
    timing_path = task.output_dir / "timing_per_schedule.csv"
    rows = [] if overwrite else read_csv(output_path)
    validate_resumed_rows(task, rows, sampling=False)
    errors = _retained_rows(error_path, "greedy", overwrite)
    timing_rows = _retained_rows(timing_path, "greedy", overwrite)

    records = ordered_records(task)
    order = {record.path.name: index for index, record in enumerate(records)}
    bks_values = bks_makespans(task)
    or_values = or_tools_makespans(task)
    bks_by_name = {
        record.path.name: float(bks_values[index])
        for index, record in enumerate(records)
    }
    or_by_name = {
        record.path.name: float(or_values[index])
        for index, record in enumerate(records)
    }
    completed = {row["instance_id"] for row in rows}
    trainer, checkpoint = load_evaluator(task.checkpoint, device=task.device)
    training = checkpoint["training_config"]
    timing_total = TimingTotals()
    pending = [record for record in records if record.path.name not in completed]
    batch_id = 1 + sum(
        row.get("decoder") == "greedy" for row in timing_rows
    )

    for start in range(0, len(pending), task.greedy_batch_size):
        chunk = pending[start : start + task.greedy_batch_size]
        environments = [make_environment(record.path, training) for record in chunk]
        outcomes, batch_timing = evaluate_batch(
            trainer, environments, deterministic=True
        )
        timing_total.add(batch_timing)
        timing_rows.append(_timing_row("greedy", batch_id, batch_timing))
        batch_id += 1
        batch_errors: List[Dict[str, Any]] = []
        for record, outcome in zip(chunk, outcomes):
            instance_id = record.path.name
            validation_errors = validate_schedule(
                CanonicalInstance.from_fjs(record.path),
                outcome.assignments,
                outcome.makespan,
            )
            batch_errors.extend(public_error_rows(
                task, instance_id, "greedy", 1, validation_errors
            ))
            bks = bks_by_name[instance_id]
            or_value = or_by_name[instance_id]
            rows.append({
                "source_distribution": task.source_distribution,
                "source_size": task.source_size,
                "benchmark": task.benchmark,
                "training_seed": task.training_seed,
                "instance_id": instance_id,
                "decoding": "greedy",
                "makespan": outcome.makespan,
                "bks_makespan": bks,
                "or_tools_makespan": or_value,
                "rpd_bks_percent": _relative_gap(outcome.makespan, bks),
                "gap_or_tools_percent": _relative_gap(
                    outcome.makespan, or_value
                ),
                "decision_count": outcome.decision_count,
                "feasible": not validation_errors,
                "evaluation_seed": deterministic_evaluation_seed(
                    task, instance_id, "greedy", 1
                ),
            })
        errors.extend(batch_errors)
        rows.sort(key=lambda row: order[str(row["instance_id"])])
        write_csv_atomic(output_path, rows, GREEDY_FIELDS)
        write_csv_atomic(error_path, errors, ERROR_FIELDS)
        write_csv_atomic(timing_path, timing_rows, TIMING_FIELDS)
        if batch_errors:
            raise RuntimeError(
                "Independent schedule validation failed; see "
                f"{error_path}"
            )
        print(
            f"  greedy {len(rows):03d}/{len(records):03d}: "
            f"mean={mean(float(row['makespan']) for row in rows):.3f}",
            flush=True,
        )

    if len(rows) != len(records):
        raise RuntimeError(
            f"Greedy output has {len(rows)} rows; expected {len(records)}"
        )
    values = [float(row["makespan"]) for row in rows]
    metadata = {
        **_metadata_base(task, checkpoint),
        "decoder": "greedy",
        "instances": len(rows),
        **_quality_metrics(values, bks_values.tolist(), or_values.tolist()),
        "feasible_schedules": sum(
            str(row["feasible"]).lower() == "true" for row in rows
        ),
        "timing_this_call": timing_total.to_dict(),
    }
    write_json_atomic(task.output_dir / "greedy_metadata.json", metadata)
    _update_evaluation_metadata(task, "greedy", metadata)
    return output_path


def run_sampling100(
    task: PublicBenchmarkEvaluationTask,
    *,
    overwrite: bool = False,
) -> Path:
    initialize_output(task)
    greedy_metadata_path = task.output_dir / "greedy_metadata.json"
    if not greedy_metadata_path.is_file():
        raise RuntimeError("Run and audit Greedy before Sampling-100")
    greedy_metadata = json.loads(
        greedy_metadata_path.read_text(encoding="utf-8")
    )
    if (
        int(greedy_metadata.get("instances", -1)) != task.test_limit
        or int(greedy_metadata.get("feasible_schedules", -1)) != task.test_limit
    ):
        raise RuntimeError("Greedy evaluation is incomplete or infeasible")

    seed_everything(task.training_seed)
    rollout_path = task.output_dir / "sampling100_rollouts.csv"
    summary_path = task.output_dir / "sampling100_per_instance.csv"
    error_path = task.output_dir / "feasibility_errors.csv"
    timing_path = task.output_dir / "timing_per_schedule.csv"
    rows = [] if overwrite else read_csv(rollout_path)
    validate_resumed_rows(task, rows, sampling=True)
    errors = _retained_rows(error_path, "sampling100", overwrite)
    timing_rows = _retained_rows(timing_path, "sampling100", overwrite)

    records = ordered_records(task)
    order = {record.path.name: index for index, record in enumerate(records)}
    bks_values = bks_makespans(task)
    or_values = or_tools_makespans(task)
    bks_by_name = {
        record.path.name: float(bks_values[index])
        for index, record in enumerate(records)
    }
    or_by_name = {
        record.path.name: float(or_values[index])
        for index, record in enumerate(records)
    }
    trainer, checkpoint = load_evaluator(task.checkpoint, device=task.device)
    training = checkpoint["training_config"]
    timing_total = TimingTotals()
    completed = {
        (row["instance_id"], int(row["rollout_id"])) for row in rows
    }
    batch_id = 1 + sum(
        row.get("decoder") == "sampling100" for row in timing_rows
    )

    for instance_index, record in enumerate(records, start=1):
        instance_id = record.path.name
        missing = [
            rollout_id
            for rollout_id in range(1, task.sampling_rollouts + 1)
            if (instance_id, rollout_id) not in completed
        ]
        instance = CanonicalInstance.from_fjs(record.path)
        for start in range(0, len(missing), task.sampling_batch_size):
            rollout_ids = missing[start : start + task.sampling_batch_size]
            seeds = [
                deterministic_evaluation_seed(
                    task, instance_id, "sampling100", rollout_id
                )
                for rollout_id in rollout_ids
            ]
            environments = [
                make_environment(record.path, training) for _ in rollout_ids
            ]
            outcomes, batch_timing = evaluate_batch(
                trainer,
                environments,
                deterministic=False,
                action_seeds=seeds,
            )
            timing_total.add(batch_timing)
            timing_rows.append(_timing_row(
                "sampling100",
                batch_id,
                batch_timing,
                instance_id=instance_id,
                rollout_ids=rollout_ids,
            ))
            batch_id += 1
            batch_errors: List[Dict[str, Any]] = []
            bks = bks_by_name[instance_id]
            or_value = or_by_name[instance_id]
            for rollout_id, evaluation_seed, outcome in zip(
                rollout_ids, seeds, outcomes
            ):
                validation_errors = validate_schedule(
                    instance, outcome.assignments, outcome.makespan
                )
                batch_errors.extend(public_error_rows(
                    task,
                    instance_id,
                    "sampling100",
                    rollout_id,
                    validation_errors,
                ))
                rows.append({
                    "source_distribution": task.source_distribution,
                    "source_size": task.source_size,
                    "benchmark": task.benchmark,
                    "training_seed": task.training_seed,
                    "instance_id": instance_id,
                    "rollout_id": rollout_id,
                    "makespan": outcome.makespan,
                    "bks_makespan": bks,
                    "or_tools_makespan": or_value,
                    "rpd_bks_percent": _relative_gap(outcome.makespan, bks),
                    "gap_or_tools_percent": _relative_gap(
                        outcome.makespan, or_value
                    ),
                    "decision_count": outcome.decision_count,
                    "feasible": not validation_errors,
                    "evaluation_seed": evaluation_seed,
                })
                completed.add((instance_id, rollout_id))
            errors.extend(batch_errors)
            rows.sort(key=lambda row: (
                order[str(row["instance_id"])], int(row["rollout_id"])
            ))
            write_csv_atomic(rollout_path, rows, SAMPLING_FIELDS)
            write_csv_atomic(error_path, errors, ERROR_FIELDS)
            write_csv_atomic(timing_path, timing_rows, TIMING_FIELDS)
            if batch_errors:
                raise RuntimeError(
                    "Independent schedule validation failed; see "
                    f"{error_path}"
                )

        instance_rows = [
            row for row in rows if row["instance_id"] == instance_id
        ]
        if len(instance_rows) == task.sampling_rollouts:
            best = min(instance_rows, key=lambda row: float(row["makespan"]))
            print(
                f"  sampling100 {instance_index:03d}/{len(records):03d} "
                f"{instance_id}: best={float(best['makespan']):.3f}",
                flush=True,
            )

    summary_rows: List[Dict[str, Any]] = []
    for record in records:
        instance_id = record.path.name
        instance_rows = [
            row for row in rows if row["instance_id"] == instance_id
        ]
        if len(instance_rows) != task.sampling_rollouts:
            continue
        best = min(instance_rows, key=lambda row: float(row["makespan"]))
        best_makespan = float(best["makespan"])
        bks = bks_by_name[instance_id]
        or_value = or_by_name[instance_id]
        summary_rows.append({
            "source_distribution": task.source_distribution,
            "source_size": task.source_size,
            "benchmark": task.benchmark,
            "training_seed": task.training_seed,
            "instance_id": instance_id,
            "best_makespan": best_makespan,
            "best_rollout_id": int(best["rollout_id"]),
            "bks_makespan": bks,
            "or_tools_makespan": or_value,
            "rpd_bks_percent": _relative_gap(best_makespan, bks),
            "gap_or_tools_percent": _relative_gap(
                best_makespan, or_value
            ),
            "feasible_rollouts": sum(
                str(row["feasible"]).lower() == "true"
                for row in instance_rows
            ),
        })
    write_csv_atomic(summary_path, summary_rows, SAMPLING_SUMMARY_FIELDS)
    if len(summary_rows) != len(records):
        raise RuntimeError(
            f"Sampling output has {len(summary_rows)} complete instances; "
            f"expected {len(records)}"
        )

    values = [float(row["best_makespan"]) for row in summary_rows]
    metadata = {
        **_metadata_base(task, checkpoint),
        "decoder": "sampling100",
        "instances": len(summary_rows),
        "rollouts_per_instance": task.sampling_rollouts,
        "total_rollouts": len(rows),
        **_quality_metrics(values, bks_values.tolist(), or_values.tolist()),
        "feasible_rollouts": sum(
            int(row["feasible_rollouts"]) for row in summary_rows
        ),
        "timing_this_call": timing_total.to_dict(),
    }
    write_json_atomic(
        task.output_dir / "sampling100_metadata.json", metadata
    )
    _update_evaluation_metadata(task, "sampling100", metadata)
    return summary_path
