from __future__ import annotations

"""Resumable Greedy and Sampling-100 evaluation for Section 5.3.1."""

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
from CPNModelV2.experiments.sec53_generalization.cross_scale_core import (
    CROSS_SCALE_PROTOCOL_VERSION,
    CrossScaleEvaluationTask,
    checkpoint_audit,
    cross_scale_error_rows,
    deterministic_evaluation_seed,
    initialize_output,
    ordered_test_paths,
    reference_makespans,
    validate_resumed_rows,
)
from CPNModelV2.integration.equivalence_core import CanonicalInstance


GREEDY_FIELDS = (
    "distribution", "source_size", "target_size", "training_seed",
    "instance_id", "decoding", "makespan", "reference_makespan",
    "rpd_percent", "decision_count", "feasible", "evaluation_seed",
)
SAMPLING_FIELDS = (
    "distribution", "source_size", "target_size", "training_seed",
    "instance_id", "rollout_id", "makespan", "reference_makespan",
    "rpd_percent", "decision_count", "feasible", "evaluation_seed",
)
SAMPLING_SUMMARY_FIELDS = (
    "distribution", "source_size", "target_size", "training_seed",
    "instance_id", "best_makespan", "best_rollout_id",
    "reference_makespan", "rpd_percent", "feasible_rollouts",
)
ERROR_FIELDS = (
    "distribution", "source_size", "target_size", "training_seed",
    "instance_id", "decoder", "rollout_id", "error",
)
TIMING_FIELDS = (
    "decoder", "batch_id", "instance_id", "rollout_start", "rollout_end",
    "schedules", "end_to_end_s", "seconds_per_schedule",
    "tensorization_s", "network_inference_s", "action_selection_s",
    "environment_step_and_graph_s",
)


def _rpd(makespan: float, reference: float) -> float:
    return (makespan - reference) / reference * 100.0


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


def _metadata_base(
    task: CrossScaleEvaluationTask,
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "evaluation_protocol_version": CROSS_SCALE_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "distribution": task.distribution,
        "source_size": task.source_size,
        "target_size": task.target_size,
        "training_seed": task.training_seed,
        "checkpoint_label": task.checkpoint_label,
        "checkpoint": str(task.checkpoint.resolve()),
        "test_dataset": str(task.test_dataset.resolve()),
        "schema_refit_on_target": False,
        "normalizer_refit_on_target": False,
        "parameter_updates_on_target": False,
        **checkpoint_audit(task, checkpoint),
    }


def _update_evaluation_metadata(
    task: CrossScaleEvaluationTask,
    decoder: str,
    decoder_metadata: Mapping[str, Any],
) -> None:
    path = task.output_dir / "evaluation_metadata.json"
    existing: Dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update({
        "evaluation_protocol_version": CROSS_SCALE_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "distribution": task.distribution,
        "source_size": task.source_size,
        "target_size": task.target_size,
        "training_seed": task.training_seed,
        "checkpoint_label": task.checkpoint_label,
    })
    decoders = dict(existing.get("decoders", {}))
    decoders[decoder] = dict(decoder_metadata)
    existing["decoders"] = decoders
    write_json_atomic(path, existing)


def _retained_errors(
    path: Path,
    decoder: str,
    overwrite: bool,
) -> List[Dict[str, str]]:
    rows = read_csv(path)
    if overwrite:
        return [row for row in rows if row.get("decoder") != decoder]
    return rows


def _retained_timing(
    path: Path,
    decoder: str,
    overwrite: bool,
) -> List[Dict[str, str]]:
    rows = read_csv(path)
    if overwrite:
        return [row for row in rows if row.get("decoder") != decoder]
    return rows


def run_greedy(
    task: CrossScaleEvaluationTask,
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
    errors = _retained_errors(error_path, "greedy", overwrite)
    timing_rows = _retained_timing(timing_path, "greedy", overwrite)

    paths = ordered_test_paths(task)
    order = {path.name: index for index, path in enumerate(paths)}
    references = reference_makespans(task)
    reference_by_name = {
        path.name: float(references[index]) for index, path in enumerate(paths)
    }
    completed = {row["instance_id"] for row in rows}
    trainer, checkpoint = load_evaluator(task.checkpoint, device=task.device)
    audit = checkpoint_audit(task, checkpoint)
    training = checkpoint["training_config"]
    timing_total = TimingTotals()

    pending = [path for path in paths if path.name not in completed]
    batch_id = 1 + sum(
        row.get("decoder") == "greedy" for row in timing_rows
    )
    for start in range(0, len(pending), task.greedy_batch_size):
        chunk = pending[start : start + task.greedy_batch_size]
        environments = [make_environment(path, training) for path in chunk]
        outcomes, batch_timing = evaluate_batch(
            trainer, environments, deterministic=True
        )
        timing_total.add(batch_timing)
        timing_rows.append(_timing_row("greedy", batch_id, batch_timing))
        batch_id += 1

        batch_errors: List[Dict[str, Any]] = []
        for path, outcome in zip(chunk, outcomes):
            validation_errors = validate_schedule(
                CanonicalInstance.from_fjs(path),
                outcome.assignments,
                outcome.makespan,
            )
            batch_errors.extend(cross_scale_error_rows(
                task, path.name, "greedy", 1, validation_errors
            ))
            reference = reference_by_name[path.name]
            rows.append({
                "distribution": task.distribution,
                "source_size": task.source_size,
                "target_size": task.target_size,
                "training_seed": task.training_seed,
                "instance_id": path.name,
                "decoding": "greedy",
                "makespan": outcome.makespan,
                "reference_makespan": reference,
                "rpd_percent": _rpd(outcome.makespan, reference),
                "decision_count": outcome.decision_count,
                "feasible": not validation_errors,
                "evaluation_seed": deterministic_evaluation_seed(
                    task, path.name, "greedy", 1
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
            f"  greedy {len(rows):03d}/{len(paths):03d}: "
            f"mean={mean(float(row['makespan']) for row in rows):.3f}",
            flush=True,
        )

    if len(rows) != len(paths):
        raise RuntimeError(
            f"Greedy output has {len(rows)} rows; expected {len(paths)}"
        )
    values = [float(row["makespan"]) for row in rows]
    reference_values = [float(row["reference_makespan"]) for row in rows]
    rpds = [float(row["rpd_percent"]) for row in rows]
    metadata = {
        **_metadata_base(task, checkpoint),
        "decoder": "greedy",
        "instances": len(rows),
        "mean_makespan": mean(values),
        "std_instance_makespan": float(np.std(values, ddof=0)),
        "mean_reference_makespan": mean(reference_values),
        "aggregate_gap_percent": _rpd(
            mean(values), mean(reference_values)
        ),
        "mean_instance_rpd_percent": mean(rpds),
        "feasible_schedules": sum(
            str(row["feasible"]).lower() == "true" for row in rows
        ),
        "timing_this_call": timing_total.to_dict(),
        **audit,
    }
    write_json_atomic(task.output_dir / "greedy_metadata.json", metadata)
    _update_evaluation_metadata(task, "greedy", metadata)
    return output_path


def run_sampling100(
    task: CrossScaleEvaluationTask,
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
    errors = _retained_errors(error_path, "sampling100", overwrite)
    timing_rows = _retained_timing(timing_path, "sampling100", overwrite)

    paths = ordered_test_paths(task)
    order = {path.name: index for index, path in enumerate(paths)}
    references = reference_makespans(task)
    reference_by_name = {
        path.name: float(references[index]) for index, path in enumerate(paths)
    }
    trainer, checkpoint = load_evaluator(task.checkpoint, device=task.device)
    audit = checkpoint_audit(task, checkpoint)
    training = checkpoint["training_config"]
    timing_total = TimingTotals()
    completed = {
        (row["instance_id"], int(row["rollout_id"])) for row in rows
    }
    batch_id = 1 + sum(
        row.get("decoder") == "sampling100" for row in timing_rows
    )

    for instance_index, path in enumerate(paths, start=1):
        missing = [
            rollout_id
            for rollout_id in range(1, task.sampling_rollouts + 1)
            if (path.name, rollout_id) not in completed
        ]
        instance = CanonicalInstance.from_fjs(path)
        for start in range(0, len(missing), task.sampling_batch_size):
            rollout_ids = missing[start : start + task.sampling_batch_size]
            seeds = [
                deterministic_evaluation_seed(
                    task, path.name, "sampling100", rollout_id
                )
                for rollout_id in rollout_ids
            ]
            environments = [
                make_environment(path, training) for _ in rollout_ids
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
                instance_id=path.name,
                rollout_ids=rollout_ids,
            ))
            batch_id += 1

            batch_errors: List[Dict[str, Any]] = []
            reference = reference_by_name[path.name]
            for rollout_id, evaluation_seed, outcome in zip(
                rollout_ids, seeds, outcomes
            ):
                validation_errors = validate_schedule(
                    instance, outcome.assignments, outcome.makespan
                )
                batch_errors.extend(cross_scale_error_rows(
                    task,
                    path.name,
                    "sampling100",
                    rollout_id,
                    validation_errors,
                ))
                rows.append({
                    "distribution": task.distribution,
                    "source_size": task.source_size,
                    "target_size": task.target_size,
                    "training_seed": task.training_seed,
                    "instance_id": path.name,
                    "rollout_id": rollout_id,
                    "makespan": outcome.makespan,
                    "reference_makespan": reference,
                    "rpd_percent": _rpd(outcome.makespan, reference),
                    "decision_count": outcome.decision_count,
                    "feasible": not validation_errors,
                    "evaluation_seed": evaluation_seed,
                })
                completed.add((path.name, rollout_id))
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
            row for row in rows if row["instance_id"] == path.name
        ]
        if len(instance_rows) == task.sampling_rollouts:
            best = min(instance_rows, key=lambda row: float(row["makespan"]))
            print(
                f"  sampling100 {instance_index:03d}/{len(paths):03d} "
                f"{path.name}: best={float(best['makespan']):.3f}",
                flush=True,
            )

    summary_rows: List[Dict[str, Any]] = []
    for path in paths:
        instance_rows = [
            row for row in rows if row["instance_id"] == path.name
        ]
        if len(instance_rows) != task.sampling_rollouts:
            continue
        best = min(instance_rows, key=lambda row: float(row["makespan"]))
        reference = reference_by_name[path.name]
        best_makespan = float(best["makespan"])
        summary_rows.append({
            "distribution": task.distribution,
            "source_size": task.source_size,
            "target_size": task.target_size,
            "training_seed": task.training_seed,
            "instance_id": path.name,
            "best_makespan": best_makespan,
            "best_rollout_id": int(best["rollout_id"]),
            "reference_makespan": reference,
            "rpd_percent": _rpd(best_makespan, reference),
            "feasible_rollouts": sum(
                str(row["feasible"]).lower() == "true"
                for row in instance_rows
            ),
        })
    write_csv_atomic(summary_path, summary_rows, SAMPLING_SUMMARY_FIELDS)
    if len(summary_rows) != len(paths):
        raise RuntimeError(
            f"Sampling output has {len(summary_rows)} complete instances; "
            f"expected {len(paths)}"
        )

    values = [float(row["best_makespan"]) for row in summary_rows]
    reference_values = [
        float(row["reference_makespan"]) for row in summary_rows
    ]
    rpds = [float(row["rpd_percent"]) for row in summary_rows]
    metadata = {
        **_metadata_base(task, checkpoint),
        "decoder": "sampling100",
        "instances": len(summary_rows),
        "rollouts_per_instance": task.sampling_rollouts,
        "total_rollouts": len(rows),
        "mean_best_makespan": mean(values),
        "std_instance_best_makespan": float(np.std(values, ddof=0)),
        "mean_reference_makespan": mean(reference_values),
        "aggregate_gap_percent": _rpd(
            mean(values), mean(reference_values)
        ),
        "mean_instance_rpd_percent": mean(rpds),
        "feasible_rollouts": sum(
            int(row["feasible_rollouts"]) for row in summary_rows
        ),
        "timing_this_call": timing_total.to_dict(),
        **audit,
    }
    write_json_atomic(
        task.output_dir / "sampling100_metadata.json", metadata
    )
    _update_evaluation_metadata(task, "sampling100", metadata)
    return summary_path
