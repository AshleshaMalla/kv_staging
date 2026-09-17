1. Hardware provenance snapshot (Sep 12)

  Script: scripts/hw_snapshot.sh
  Data: data/hw/2026-09-12T10:51:09-05:00_154420_rpg-93-4/
  What: GPU topology, NUMA layout, memory, mounts — records the machine state for reproducibility.
  salloc -p h100 -N1 --time=00:30:00
  bash scripts/hw_snapshot.sh

  2. Single-node NFS bandwidth sweep — libaio (Sep 12)

  Script: scripts/bw_sweep.sh
  Data: data/raw/bw_shared_nfs_2026-09-12T11:15:22-05:00/
  What: Read bandwidth vs parallelism (1–16 fio streams), posixaio engine, 1G files, 30s each. Established the 4.4–4.8 GB/s flat ceiling.
  salloc -p h100 -N1 --time=01:00:00
  bash scripts/bw_sweep.sh /mnt/REPACSS shared_nfs

  3. Prefill latency sweep — HuggingFace (Sep 12)

  Script: scripts/prefill_sweep.py
  Data: data/raw/prefill_20260912T180723Z.json (canonical run)
  What: Prefill time vs context length (512–131072 tokens) on Llama-3.1-8B, raw HF forward passes.
  salloc -p h100 -N1 --gpus=1 --time=01:00:00
  python3 scripts/prefill_sweep.py

  4. Prefill latency sweep — vLLM (Sep 12)

  Script: scripts/prefill_sweep_vllm.py
  Data: data/raw/prefill_vllm_20260912T191052Z.json
  What: Same sweep using vLLM's offline LLM class.
  salloc -p h100 -N1 --gpus=1 --time=01:00:00
  python3 scripts/prefill_sweep_vllm.py

  5. vLLM config sweep (Sep 12)

  Script: scripts/prefill_config_sweep.py
  Data: data/raw/prefill_config_sweep_20260912T200514Z.json
  What: Sweep vLLM configs at fixed context length to find compute ceiling.
  salloc -p h100 -N1 --gpus=1 --time=01:00:00
  python3 scripts/prefill_config_sweep.py

  6. Crossover analysis (Sep 12)

  Script: analysis/crossover.py
  Data: data/crossover.png
  What: Combines prefill fit with bandwidth data to find the context length where fetching beats recomputing.
  python3 analysis/crossover.py data/raw/prefill_20260912T180723Z.json \
    data/raw/bw_shared_nfs_2026-09-12T11:15:22-05:00/summary.csv

  7. Multi-node bandwidth scaling (Sep 13)

  Script: scripts/bw_multinode.sh
  Data: data/raw/bw_multinode_20260913T195637Z/
  Analysis: analysis/multinode_scaling.py → data/multinode_scaling.png
  What: Synchronized fio across 1/2/4 nodes, 3 reps each. Answered per-node vs shared ceiling question (answer: per-node, 90% scaling efficiency at 4
  nodes).
  salloc -p h100 -N4 --time=02:00:00
  bash scripts/bw_multinode.sh /mnt/REPACSS
  python3 analysis/multinode_scaling.py \
    data/raw/bw_multinode_20260913T195637Z/summary_per_node.csv \
    data/raw/bw_multinode_20260913T195637Z/summary_aggregate.csv

  8. Engine confound check — posixaio sweep (Sep 13)

  Script: scripts/bw_engine_compare.sh
  Data: data/raw/bw_posixaio_sweep_20260913T202840Z/
  What: Single-node sweep with posixaio engine, 8G files, 3 reps per job count. Eliminated the ioengine as a confound — posixaio is if anything faster,
  but file size also differed so the exact engine delta is not isolated.
  salloc -p h100 -N1 --time=01:00:00
  bash scripts/bw_engine_compare.sh /mnt/REPACSS

  9. Bandwidth variance time series (Sep 13)

  Script: scripts/bw_timeseries.sh
  Data: data/raw/bw_timeseries_20260913T211736Z/
  Analysis: analysis/bw_variance.py → data/bw_timeseries.png
  What: 305 samples over 103 minutes, same fio config, measuring only time variation. Found CV = 0.65% — the quiescent regime is a flat line. No
  degradation episodes observed. Autocorrelation dropped below 0.5 at lag 1 (noise on a stable signal, not the contention process).
  salloc -p h100 -N1 --time=02:00:00
  bash scripts/bw_timeseries.sh /mnt/REPACSS
  python3 analysis/bw_variance.py \
    data/raw/bw_timeseries_20260913T211736Z/timeseries.csv

  10. Controlled contention experiment (Sep 13, may still be running)

  Script: scripts/contention_experiment.sh
  Data: data/raw/contention_20260913T234348Z/
  Analysis: analysis/contention_response.py
  What: Victim h100 node under continuous measurement while 1/4/16/64 zen4 aggressor nodes generate scheduled load cycles (5 min quiet → 10 min load → 10
  min recovery). Measures onset/recovery lags, dose-response, and whether p99 leads BW degradation.
  # Runs under sbatch, not salloc
  bash scripts/contention_experiment.sh 64
  # When complete:
  python3 analysis/contention_response.py \
    data/raw/contention_20260913T234348Z \
    --quiescent-csv data/raw/bw_timeseries_20260913T211736Z/timeseries.csv

  Key config & docs

  - config/measured_constants.yaml — every measured number with units, source script, and timestamp
  - docs/DECISIONS.md — decision log
  - docs/GATES.md — pre-registered decision thresholds
  - docs/memos/2026-09-12-weekend.md — weekend research plan

  Infrastructure notes

  - Cluster: h100 partition (8 nodes, H100 GPUs), zen4 partition (110 EPYC nodes, 256 cores each)
  - Storage: Hammerspace NFS at 10.102.95.220, NFSv4.2, nconnect=4, mounted at /mnt/REPACSS
  - fio: User-installed at ~/opt/bin/fio. libaio package is not installed on h100 nodes — use posixaio
  - GPU clocks: Locked at 345/1785 MHz (not base or boost) — prefill coefficients are lower bounds
