# Decision Gates

Thresholds must be written BEFORE the run that tests them. Do not backfill a
threshold after seeing the result.

## Gates applied during this project

The formal pre-registered gate table above was not used (no entries were
written before runs). The following methodological gates were applied
retroactively as the project's process matured:

### Seeded replication at n >= 5

Any simulation-derived finding requires replication across at least 5 seeds
before it can be reported. Applied starting 2026-09-19 to test the
metastability claim: 30 seeds showed the median was 1.09x (not the 2.16x from
seed 7 of 5). The Q(N) extended sweep used 3 reps per configuration across 3
nodes.

### Baseline comparison before claiming a mechanism

A candidate finding must be compared against the simplest model that could
produce it. Applied 2026-09-20: the super-additive feedback interaction was
compared against a convexity baseline (fair-share queue with no layer pipeline)
and found to be reproduced by the baseline — the interaction was textbook
queueing, not KV-specific.

### No-episode controls

Simulation runs with degradation episodes must be compared against no-episode
controls at the same load to distinguish episode-induced effects from intrinsic
instability. Applied 2026-09-19: the 85% load metastability run was excluded
because its no-episode control trended upward (intrinsic instability, not
episode-induced).

### Provenance coupling (code reads from config)

Measurement scripts must read constants from `config/measured_constants.yaml`
and write results back with date and source provenance. Implemented 2026-09-18
after the 27x claim was found to use superseded HF coefficients that had
decoupled from the authoritative YAML values.

### Shared-resource audit before modeling concurrency

When modeling concurrency, every shared resource on the path must be divided
among concurrent requests: storage link, PCIe link, GPU compute. List each
resource and state explicitly whether it is shared or per-request before
computing anything. Applied 2026-09-24 after the loaded crossover was found
to give each concurrent fetch unshared storage bandwidth and unshared PCIe,
contradicting the project's own flat-across-streams bandwidth measurement.
