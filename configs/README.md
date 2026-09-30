# Configuration

- [default.json](default.json): inputs, output location, objective, distance limit,
  device, CPU threads, and random seed for a single evaluation.
- [train.json](train.json): training scenarios, budgets, learning rates, and output directory.

All file paths are relative to the project root. Set `device` to `cpu` or `cuda:0`.
The `objective` weights follow `[TFR, TCR, MTE]` and must sum to one.
`reachable_distance_km` is measured in kilometers.

Set training speed in kilometers per scenario step and `distance_reference_km` for the dataset.
Grid bounds and feature scales are fitted on training scenarios unless supplied under `environment`.
`objectives` accepts a Dirichlet sampling configuration or an explicit list of weight vectors.
