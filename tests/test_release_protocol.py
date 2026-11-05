from __future__ import annotations

from pathlib import Path

from CPNModelV2.experiments.sec52_fjsp_performance import evaluate_worker as sec52_eval
from CPNModelV2.experiments.sec52_fjsp_performance import train_worker as sec52_train
from CPNModelV2.experiments.sec52_fjsp_performance import (
    train_generic_compact_10x5 as sec52_core,
)
from CPNModelV2.experiments.sec53_generalization import (
    evaluate_public_sd1_15x10 as sec53_public,
)
from CPNModelV2.experiments.sec54_representation import (
    evaluate_distribution_sd1_15x10_bsg as sec54_eval,
)
from CPNModelV2.experiments.sec54_representation import (
    train_worker_sd1_15x10 as sec54_train,
)
from CPNModelV2.experiments.sec55_ablation import evaluate_SD1_15x10 as sec55_eval
from CPNModelV2.experiments.sec55_ablation import train_worker_SD1_15x10 as sec55_train


FORMAL_SEEDS = (9000, 9001, 9002, 9003, 9004)


def _assert_seed_checkpoint(seed: int, checkpoint: Path) -> None:
    checkpoint = Path(checkpoint)
    assert checkpoint.name == "best_checkpoint.pt"
    assert checkpoint.parent.name == f"seed_{seed}"


def test_sections_52_and_53_use_the_1000_iteration_protocol() -> None:
    assert sec52_train.ITERATIONS == 1000
    assert sec52_core.FULL_ITERATIONS == 1000
    assert sec52_train.TRAINING_PROFILE == "full"
    assert sec52_train.RUN_GROUP == "formal_1000x20"

    for selection in sec52_eval.CHECKPOINTS_TO_EVALUATE:
        _assert_seed_checkpoint(selection[3], selection[4])
    for selection in sec53_public.CHECKPOINTS_TO_EVALUATE:
        _assert_seed_checkpoint(selection[3], selection[4])


def test_sec54_uses_five_distinct_seed_checkpoints() -> None:
    assert sec54_train.TRAINING_SEEDS == FORMAL_SEEDS
    assert tuple(seed for seed, _label, _path in sec54_eval.CHECKPOINTS) == FORMAL_SEEDS
    paths = []
    for seed, label, checkpoint in sec54_eval.CHECKPOINTS:
        assert label == "best"
        _assert_seed_checkpoint(seed, checkpoint)
        paths.append(Path(checkpoint))
    assert len(paths) == len(set(paths))


def test_sec55_uses_five_distinct_seed_checkpoints() -> None:
    assert sec55_train.TRAINING_SEEDS == FORMAL_SEEDS
    assert tuple(seed for seed, _label, _path in sec55_eval.CHECKPOINTS) == FORMAL_SEEDS
    paths = []
    for seed, label, checkpoint in sec55_eval.CHECKPOINTS:
        assert label == "best"
        _assert_seed_checkpoint(seed, checkpoint)
        paths.append(Path(checkpoint))
    assert len(paths) == len(set(paths))
