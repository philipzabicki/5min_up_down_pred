# Performance

Engineering guidance for this repository's feature pipelines, model-ready datasets, training and tuning, OOF predictions, historical experiments, and live inference. These are requirements for future changes, not claims that the current code already implements them. See [AGENTS.md](AGENTS.md) for scope, configuration, and Git rules.

## 1. Optimize the actual workload

Start with representative input sizes and an explicit objective: elapsed time, peak memory, throughput, or live latency. Identify the dominant cost before choosing a library or execution backend.

For a material change: baseline → profile → reduce complexity and repeated work → implement the simplest suitable alternative → measure the full cost → verify equivalent results.

- Estimate rows, symbols, features, folds, trials, quote ticks, policies, and scenarios. Watch for work proportional to products of these dimensions.
- Prefer a better algorithm or one shared pass over faster execution of redundant work.
- Use small cases for development and equivalence checks, then representative data for performance conclusions. Do not substitute a toy workload for the authorized full run.
- Reducing coverage, folds, trials, features, precision, or quote frequency changes the experiment. Report and obtain the relevant methodological authorization separately; do not label it a pure implementation speedup.
- Keep changes local. Do not introduce a new storage system, framework, or dependency without a measured reason.

## 2. Data layout, I/O, and reuse

- Read only required Parquet/Arrow columns and partitions; apply supported filtering early. Use bounded batches when the full decompressed dataset plus temporary arrays will not fit comfortably.
- Load, normalize, sort, and index shared immutable inputs once per suitable run or worker. Avoid rescanning raw history or rebuilding the same joins per feature, fold, trial, policy, or latency scenario.
- Prefer native column operations and compact NumPy/Arrow representations for large regular computations. Profile Python row loops, repeated DataFrame concatenation, object columns, and large intermediate copies.
- Vectorization is not automatically memory-efficient. Avoid materializing huge broadcast arrays or a full rows × features × trials/scenarios tensor.
- Batch predictions where appropriate. Reuse loaded models; do not reload an artifact for each row or market window.
- Write large outputs incrementally. Keep JSON for small configuration, manifests, and summaries rather than duplicating full tables or order books in every result.
- Preserve source records and useful replay material. Do not delete raw data or active captures as a routine performance cleanup.

## 3. Features, training, and OOF

- Reuse causal deterministic feature computations when their inputs and configuration match. Cache keys must distinguish feature definitions, order, data identity, and relevant time ranges.
- Rolling and incremental implementations must preserve warm-up, missing-data treatment, window endpoints, timezone, and timestamp alignment. Test batch/live parity on representative boundaries.
- Share immutable feature matrices and fold indices where practical. Account for actual copies made by indexing, model-library dataset construction, and process serialization.
- Fit learned transformations, calibration, and other training-dependent state within the correct fold. Reusing a fitted object across evaluation boundaries must not introduce new leakage.
- Preserve existing targets, sample weights, splits, early-stopping settings, seeds, and modeling configuration during an implementation optimization. Changes to modeling methodology require a separate stated scope.
- Schedule independent folds or trials with a combined worker/thread budget. Parallel trials each using all LightGBM/OpenMP/BLAS threads can make the whole run slower.
- Record the runtime and actual backend. Do not silently replace a custom LightGBM/CUDA installation or change its backend to make an optimization work.

## 4. Historical replay and portfolios

- Build shared quote lookups, settlement mappings, and window joins once where inputs match. Do not reread and resort the same history for every policy.
- Parallelize independent portfolios or scenarios with isolated state. Within a portfolio, preserve chronological cash locking, order handling, fees, settlement, and redemption.
- A sequential stateful loop is legitimate. Improve its data access, representation, or measured hot kernel instead of forcing invalid vectorization.
- Preserve token/side mapping, decision-time information availability, quote selection, liquidity constraints, fee arithmetic, and rounding.
- Do not treat a future snapshot as available earlier, fill missing liquidity, or reduce latency merely to simplify or speed up replay.
- Compare ledgers and eligibility decisions as well as final PnL. Similar final totals can hide different trades and cash flows.

## 5. Live inference and network work

- Keep required state and the model loaded. Update only the history/state needed for the next decision when parity with batch computation is established.
- Measure input arrival, feature construction, inference, quote acquisition, decision, order submission, acknowledgement, and fill separately where observable. Do not infer fill latency from a synchronous call returning.
- Use a monotonic clock for durations. Wall-clock comparisons across services require clock-offset awareness; label unavailable timestamps and feed-age information as unknown.
- Report median and tail latency with sample counts, model identity, and cold/warm conditions. Quote freshness and model-compute latency are different quantities.
- Avoid bulk history reads, training, report serialization, and unnecessary synchronous I/O in the decision path. Protect live work from competing offline jobs.
- Cache and coalesce identical network requests when validity permits. Use supported batching, bounded concurrency, backpressure, timeouts, and retry limits.
- Local parallelism does not increase API allowances or paid-service authorization. Respect service rate limits and partial failures.
- A cached live quote must retain source/receive times and validity checks. Faster access does not make stale data executable.

## 6. CPU, memory, GPU, and Windows

Detect the actual interpreter, installed libraries, CPU availability, RAM, free disk, GPU/CUDA support, and competing jobs before expensive work. Use the reference workstation in AGENTS.md for planning only.

- Budget total memory: decompressed inputs + intermediate copies + per-worker model/data state + output buffers. Compressed file size is not a RAM estimate.
- Choose bounded worker counts and chunk sizes together with native library thread counts. Avoid nested oversubscription and unbounded task submission.
- On Windows, account for process spawning, pickling, startup cost, and duplicated memory; use the appropriate main-entry guard. Do not assume Unix fork-based sharing.
- Initialize expensive worker state once. Consider shared read-only storage or memory mapping only when measured savings justify the complexity.
- Compare CPU vectorization, compiled/JIT kernels, processes, and GPU against the actual workload. GPU availability alone does not justify migration.
- GPU measurements include allocation, conversion, host/device transfers, warm-up, synchronization, and result consumption. Bound batches by available VRAM, including model and temporary buffers.
- Verify library support for the actual Python/CUDA environment before adding dependencies. Preserve working environments and existing user processes.
- Do not use lower precision or fast-math without measured error bounds, including cases near model and trading decision thresholds.

## 7. Cache, checkpoints, and reproducibility

- Fingerprint relevant source data, schema, configuration, feature order, model artifact, fold/time boundaries, and computation version. Cache validity follows semantics, not just file existence.
- Hash immutable large inputs once per run or validated manifest, not inside every fold or candidate evaluation. Do not trust file names alone as identity.
- Distinguish reusable immutable preprocessing from fold-specific learned state and time-sensitive live data.
- Write checkpoints atomically with completed partitions/tasks and sufficient state to resume. Validate identity and completeness before reuse; incomplete outputs must not appear final.
- Long runs should expose progress, throughput, failures, and resource pressure. Support orderly interruption without losing completed work or corrupting source files.
- Record seeds and task identities so scheduling changes do not silently change the experiment. State any unavoidable backend nondeterminism.

## 8. Evidence and acceptance

Use an existing benchmark or representative runner where possible. Add a small reproducible benchmark only when a material hot path lacks one; avoid creating a general framework. Configure it through editable constants or existing configuration, respecting the no-console-arguments rule.

A useful before/after record includes:

- Commit/runtime/library/backend identity, hardware, worker/thread counts, and competing load.
- Input identity, date coverage, rows/windows/features/folds/trials, and relevant configuration.
- Cache state, cold start, JIT/warm-up, I/O, and steady-state computation.
- End-to-end wall time and dominant stages; throughput of completed work.
- Peak memory including workers, and peak VRAM when measured. Final RSS is not peak memory; Python allocation tracking does not include all native memory.
- Output sizes, network requests/retries, and cache hits where relevant.
- Correctness comparisons with tolerances specified before accepting the change.

Compare the same workload and resource budget. Consume lazy results and synchronize asynchronous GPU work before stopping timers. Repeat short measurements to show typical time and variability; do not rerun hours of work without a concrete validation need.

Check timestamps, row coverage/order, feature values, prediction differences, and downstream decisions as appropriate. Numerical tolerance alone is insufficient if small differences change trades. Separate implementation speedup from model-quality changes.

Accept the simplest variant meeting the objective without unacceptable memory or correctness regressions. Set regression thresholds from observed variability, not arbitrary speedup targets. Stop optimizing once the task's relevant bottleneck is resolved.

Keep this document stable. Put measured results and reproduction details in the relevant project report; do not copy benchmark claims from other projects or accumulate a run-by-run journal here.
