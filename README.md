# BSG-RL

Official implementation of **Binding-Semantic Graph Reinforcement Learning
over Colored Petri Nets: Application to Flexible Job-Shop Scheduling**.

BSG-RL builds policy observations directly from an executable colored Petri
net (CPN). Individual tokens, typed token references, enabled bindings, and
binding participation roles are retained in a dynamic graph and encoded by a
binding-centric actor--critic policy.

## Repository structure

- `CPNModelV2/core`: executable CPN definitions and firing semantics;
- `CPNModelV2/graph`: token--binding graph construction and tensorization;
- `CPNModelV2/learning`: BSG-RL policies, PPO, and environment interfaces;
- `CPNModelV2/integration`: FJSP and GymPN adapters;
- `CPNModelV2/experiments`: paper experiment entry points;
- `gympn-main`: the GymPN dependency snapshot, retained with its MIT license;
- `examples`: small executable examples;
- `tests`: unit and protocol tests.

The `core` package is part of the release because enabled bindings and state
evolution are defined by the executable CPN rather than reconstructed by the
policy code.

## Installation

Python 3.9 is the reference environment.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install a PyTorch build compatible with the local CUDA driver when GPU
acceleration is required. The reference runs used Python 3.9.23, PyTorch
2.7.1, NumPy 2.0.2, SimPN 1.10.0, Gymnasium 1.0.0, and PyTorch Geometric
2.6.1.

## Quick check

```bash
python examples/smoke_fjsp_policy.py
python -m pytest -q
```

## Running experiments

Experiment entry points keep their important settings in a clearly marked
`USER CONFIGURATION` block and can be launched directly from an IDE. The main
starting points are:

- `CPNModelV2/experiments/sec52_fjsp_performance/train_worker.py`;
- `CPNModelV2/experiments/sec54_representation/train_worker_sd1_15x10.py`;
- `CPNModelV2/experiments/sec55_ablation/train_worker.py`;
- `CPNModelV2/experiments/sec56_applicability/gympn/train_gympn_complex_controlled.py`;
- `CPNModelV2/experiments/sec56_applicability/gympn/train_gympn_semantic_matrix.py`.

Use different output directories when launching parallel processes.

## Citation

If this implementation is useful in your work, please cite the accompanying
paper. Machine-readable citation metadata is provided in `CITATION.cff`.

## License

The original BSG-RL code is released under the Apache License 2.0. GymPN is
distributed under its original MIT license; see `THIRD_PARTY.md` and
`gympn-main/LICENSE`.
