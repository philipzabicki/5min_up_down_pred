# ESN expansion downstream budget amendment — 2026-10-21

This operational amendment applies only to the continuation of the frozen development search. It was prepared before any downstream LightGBM candidate scores were available.

## Observed execution cost

At 18:15 (Europe/Warsaw), the runner had used 27.1 minutes of the preregistered 50-minute exploratory downstream stage. It had generated complete feature matrices for 34 of the 60 selected candidates; `downstream_exploratory_progress.json` still contained no baseline metrics or candidate scores. The selected pool and its 30% probe-independent stratified sample were already frozen. This measured full-prefix feature-generation cost shows the original 50-minute stage allocation leaves insufficient room for scoring all selected configurations.

The screening stage finished after 14,531 seconds, versus its frozen 27,900-second cap. Its unused 13,369 seconds provide room to extend downstream without reducing the reserved finalist and portfolio stage.

## Amended execution limits

| Stage | Frozen cap | Continuation cap | Reason |
|---|---:|---:|---|
| Screening | 27,900 s | 27,900 s | Unchanged; the 300-valid-configuration target was reached in 14,531 s |
| Exploratory downstream | 3,000 s | 14,000 s total | Allow full-prefix features and matched LightGBM scoring for the frozen 60-candidate pool |
| Finalists and portfolios | 12,300 s | 12,300 s | Unchanged; 3 h 25 min remains reserved |
| Whole runner | 43,200 s | 43,200 s | Unchanged 12-hour hard cap |

At the stage caps, screening (14,531 s actual), exploratory downstream (14,000 s maximum), and finalists/portfolios (12,300 s maximum) total 40,831 seconds, leaving 2,369 seconds below the whole-runner cap for setup and reporting. The finalist reserve remains 28.5% of the 12-hour cap and exceeds the required 25%.

This is a time-allocation amendment based only on observed compute cost. It does not change the data cutoff, splits, causal transforms, labels, purge, architectures, selected candidate IDs, seeds, metrics, selection rules, holdout/confirmation exclusions, or model/live/replay behavior. The original frozen configuration and hashes remain intact; the runtime continuation must record the amended stage cap and preserve the existing identity-checked checkpoints. This development search remains exploratory and is not independent confirmation.

The permanent checkpoint-resume and transient Windows file-replacement fixes were recorded after the run in commit `d78189f`. They repair resume behavior and do not change the saved candidate records, scores, or selection.

## Checkpoint selection recovery

The first continuation attempt re-ran the stratified random pool selector, whose RNG state was not persisted. The downstream checkpoint's identity check rejected the newly sampled list before any additional feature generation or LightGBM fit. The checkpoint's original 60 candidate IDs and order are the authoritative frozen pool. The resumed runner reconstructs the pool from those IDs and verifies the same candidate set and ordering before continuing; this restores the original selection and does not promote or exclude any candidate based on downstream results.

The next continuation restored the exact pool, completed all 60 candidate fits for the first exploratory LightGBM seed, then exposed a resume-only cleanup error (`baseline` was deleted even when its matching baseline metrics were already cached). All 60 candidate records and the baseline checkpoint were saved before the exception. The resumed run retained those records, conditionally cleaned up a baseline model only when that seed had required a new fit, and continued the remaining two LightGBM seeds. This affects runtime recovery only; it does not change any score or candidate decision.
