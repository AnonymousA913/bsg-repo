from __future__ import annotations

from CPNModelV2.experiments.sec52_fjsp_performance import train_worker
from CPNModelV2.experiments.sec53_generalization import (
    evaluate_cross_scale as cross_scale,
)
from CPNModelV2.experiments.sec53_generalization import (
    evaluate_public_sd1_15x10 as public_sd1_15x10,
)


def test_sec52_retained_example_is_sd1_15x10() -> None:
    assert train_worker.EXPERIMENTS == (("SD1", "15x10"),)


def test_generic_cross_scale_entrypoint_requires_explicit_checkpoint() -> None:
    assert cross_scale.CHECKPOINTS_TO_EVALUATE == ()
    assert cross_scale.TARGET_SIZES == ("30x10", "40x10")


def test_retained_public_example_is_sd1_15x10_seed9000() -> None:
    assert len(public_sd1_15x10.CHECKPOINTS_TO_EVALUATE) == 1
    checkpoint = public_sd1_15x10.CHECKPOINTS_TO_EVALUATE[0]
    assert checkpoint[1:4] == ("SD1", "15x10", 9000)
    assert public_sd1_15x10.RUN_GROUP == "seed9000_main_v1"
    assert len(public_sd1_15x10.BENCHMARKS) == 4
