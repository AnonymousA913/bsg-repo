from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np

from CPNModelV2.experiments.sec52_fjsp_performance.evaluation_core import (
    EVALUATION_PROTOCOL_VERSION,
    EvaluationTask,
    TimingTotals,
    checkpoint_audit,
    deterministic_evaluation_seed,
    error_rows,
    evaluate_batch,
    load_evaluator,
    make_environment,
    ordered_test_paths,
    read_csv,
    reference_makespans,
    seed_everything,
    validate_schedule,
    write_csv_atomic,
    write_json_atomic,
)
from CPNModelV2.integration.equivalence_core import CanonicalInstance


GREEDY_FIELDS = (
    "distribution", "train_size", "test_size", "training_seed",
    "instance_id", "decoding", "makespan", "reference_makespan",
    "decision_count", "feasible", "evaluation_seed",
)
SAMPLING_FIELDS = (
    "distribution", "train_size", "test_size", "training_seed",
    "instance_id", "rollout_id", "makespan", "reference_makespan",
    "decision_count", "feasible", "evaluation_seed",
)
SAMPLING_SUMMARY_FIELDS = (
    "distribution", "train_size", "test_size", "training_seed",
    "instance_id", "best_makespan", "best_rollout_id",
    "reference_makespan", "feasible_rollouts",
)
ERROR_FIELDS = (
    "distribution", "train_size", "test_size", "training_seed",
    "instance_id", "decoder", "rollout_id", "error",
)


def run_greedy(task: EvaluationTask, *, overwrite: bool = False) -> Path:
    task.validate()
    seed_everything(task.training_seed)
    output_path = task.output_dir / "greedy_per_instance.csv"
    error_path = task.output_dir / "feasibility_errors.csv"
    rows = [] if overwrite else read_csv(output_path)
    errors = [] if overwrite else read_csv(error_path)
    completed = {row["instance_id"] for row in rows}
    paths = ordered_test_paths(task)
    order = {path.name: index for index, path in enumerate(paths)}
    references = reference_makespans(task)
    reference_by_name = {
        path.name: float(references[index]) for index, path in enumerate(paths)
    }
    trainer, checkpoint = load_evaluator(task.checkpoint, device=task.device)
    audit = checkpoint_audit(task, checkpoint)
    training = checkpoint["training_config"]
    timing = TimingTotals()

    pending = [path for path in paths if path.name not in completed]
    for start in range(0, len(pending), task.greedy_batch_size):
        chunk = pending[start : start + task.greedy_batch_size]
        environments = [make_environment(path, training) for path in chunk]
        outcomes, batch_timing = evaluate_batch(
            trainer, environments, deterministic=True
        )
        timing.add(batch_timing)
        batch_errors: List[Dict[str, Any]] = []
        for path, outcome in zip(chunk, outcomes):
            validation_errors = validate_schedule(
                CanonicalInstance.from_fjs(path),
                outcome.assignments,
                outcome.makespan,
            )
            batch_errors.extend(error_rows(
                task, path.name, "greedy", 1, validation_errors
            ))
            rows.append({
                "distribution": task.distribution,
                "train_size": task.size,
                "test_size": task.size,
                "training_seed": task.training_seed,
                "instance_id": path.name,
                "decoding": "greedy",
                "makespan": outcome.makespan,
                "reference_makespan": reference_by_name[path.name],
                "decision_count": outcome.decision_count,
                "feasible": not validation_errors,
                "evaluation_seed": deterministic_evaluation_seed(
                    task, path.name, "greedy", 1
                ),
            })
        errors.extend(batch_errors)
        rows.sort(key=lambda row: order[row["instance_id"]])
        write_csv_atomic(output_path, rows, GREEDY_FIELDS)
        write_csv_atomic(error_path, errors, ERROR_FIELDS)
        if batch_errors:
            raise RuntimeError(
                "Independent schedule validation failed; see "
                f"{error_path}"
            )
        print(
            f"  greedy {len(rows):03d}/{len(paths):03d}: "
            f"mean={mean(float(row['makespan']) for row in rows):.3f}"
        )

    if len(rows) != len(paths):
        raise RuntimeError(
            f"Greedy output has {len(rows)} rows; expected {len(paths)}"
        )
    values = [float(row["makespan"]) for row in rows]
    reference_values = [float(row["reference_makespan"]) for row in rows]
    metadata = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "decoder": "greedy",
        "checkpoint": str(task.checkpoint),
        "test_dataset": str(task.test_dataset),
        "instances": len(rows),
        "mean_makespan": mean(values),
        "std_instance_makespan": float(np.std(values, ddof=0)),
        "mean_reference_makespan": mean(reference_values),
        "aggregate_gap_percent": (
            (mean(values) - mean(reference_values))
            / mean(reference_values)
            * 100.0
        ),
        "feasible_schedules": sum(
            str(row["feasible"]).lower() == "true" for row in rows
        ),
        "timing_this_call": timing.to_dict(),
        **audit,
    }
    write_json_atomic(task.output_dir / "greedy_metadata.json", metadata)
    return output_path


def run_sampling100(task: EvaluationTask, *, overwrite: bool = False) -> Path:
    task.validate()
    seed_everything(task.training_seed)
    rollout_path = task.output_dir / "sampling100_rollouts.csv"
    summary_path = task.output_dir / "sampling100_per_instance.csv"
    error_path = task.output_dir / "feasibility_errors.csv"
    rows = [] if overwrite else read_csv(rollout_path)
    errors = [] if overwrite else read_csv(error_path)
    paths = ordered_test_paths(task)
    order = {path.name: index for index, path in enumerate(paths)}
    references = reference_makespans(task)
    reference_by_name = {
        path.name: float(references[index]) for index, path in enumerate(paths)
    }
    trainer, checkpoint = load_evaluator(task.checkpoint, device=task.device)
    audit = checkpoint_audit(task, checkpoint)
    training = checkpoint["training_config"]
    timing = TimingTotals()
    completed = {
        (row["instance_id"], int(row["rollout_id"])) for row in rows
    }

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
            timing.add(batch_timing)
            batch_errors: List[Dict[str, Any]] = []
            for rollout_id, evaluation_seed, outcome in zip(
                rollout_ids, seeds, outcomes
            ):
                validation_errors = validate_schedule(
                    instance, outcome.assignments, outcome.makespan
                )
                batch_errors.extend(error_rows(
                    task,
                    path.name,
                    "sampling100",
                    rollout_id,
                    validation_errors,
                ))
                rows.append({
                    "distribution": task.distribution,
                    "train_size": task.size,
                    "test_size": task.size,
                    "training_seed": task.training_seed,
                    "instance_id": path.name,
                    "rollout_id": rollout_id,
                    "makespan": outcome.makespan,
                    "reference_makespan": reference_by_name[path.name],
                    "decision_count": outcome.decision_count,
                    "feasible": not validation_errors,
                    "evaluation_seed": evaluation_seed,
                })
                completed.add((path.name, rollout_id))
            errors.extend(batch_errors)
            rows.sort(key=lambda row: (
                order[row["instance_id"]], int(row["rollout_id"])
            ))
            write_csv_atomic(rollout_path, rows, SAMPLING_FIELDS)
            write_csv_atomic(error_path, errors, ERROR_FIELDS)
            if batch_errors:
                raise RuntimeError(
                    "Independent schedule validation failed; see "
                    f"{error_path}"
                )

        instance_rows = [row for row in rows if row["instance_id"] == path.name]
        if len(instance_rows) == task.sampling_rollouts:
            best = min(instance_rows, key=lambda row: float(row["makespan"]))
            print(
                f"  sampling100 {instance_index:03d}/{len(paths):03d} "
                f"{path.name}: best={float(best['makespan']):.3f}"
            )

    summary_rows: List[Dict[str, Any]] = []
    for path in paths:
        instance_rows = [row for row in rows if row["instance_id"] == path.name]
        if len(instance_rows) != task.sampling_rollouts:
            continue
        best = min(instance_rows, key=lambda row: float(row["makespan"]))
        summary_rows.append({
            "distribution": task.distribution,
            "train_size": task.size,
            "test_size": task.size,
            "training_seed": task.training_seed,
            "instance_id": path.name,
            "best_makespan": float(best["makespan"]),
            "best_rollout_id": int(best["rollout_id"]),
            "reference_makespan": reference_by_name[path.name],
            "feasible_rollouts": sum(
                str(row["feasible"]).lower() == "true" for row in instance_rows
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
    metadata = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "task_id": task.task_id,
        "decoder": "sampling100",
        "checkpoint": str(task.checkpoint),
        "test_dataset": str(task.test_dataset),
        "instances": len(summary_rows),
        "rollouts_per_instance": task.sampling_rollouts,
        "total_rollouts": len(rows),
        "mean_best_makespan": mean(values),
        "std_instance_best_makespan": float(np.std(values, ddof=0)),
        "mean_reference_makespan": mean(reference_values),
        "aggregate_gap_percent": (
            (mean(values) - mean(reference_values))
            / mean(reference_values)
            * 100.0
        ),
        "feasible_rollouts": sum(
            int(row["feasible_rollouts"]) for row in summary_rows
        ),
        "timing_this_call": timing.to_dict(),
        **audit,
    }
    write_json_atomic(
        task.output_dir / "sampling100_metadata.json", metadata
    )
    return summary_path
