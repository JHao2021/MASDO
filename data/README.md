# Data

The datasets are Foursquare-TKY, Foursquare-NYC, and Porto Taxi.
Obtain them from their providers under the applicable licenses.

Place prepared scenario JSON files in `data/prepared/` and set `scenario`
in `configs/default.json`.

Required fields:

- Scenario: `coordinate_scale_km`, `horizon`, `workers`, `tasks`.
- Worker: `worker_id`, `location`; optional `release_time` and `end_time`.
- Task: `task_id`, `location`, `release_time`, `deadline`, `value`.

Use planar coordinates and integer time steps consistent with the checkpoint.
`coordinate_scale_km` specifies kilometers per coordinate unit.
