# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. No Console Arguments

**Never add command-line argument parsing in this project.**

- Do not use argparse, click, typer, sys.argv, or similar console-argument APIs.
- Project scripts should be configured through constants near the top of the file, JSON config files, or existing project settings.
- If a script needs user-adjustable inputs, make them explicit editable variables in the file or read them from the project's configuration conventions.
- Do not suggest console flags as the normal way to run project scripts.

## 6. Performance and Local Compute

**Performance is part of correctness for substantial data and compute paths.** Read [performance.md](performance.md) before changing dataset construction, features, training/tuning, OOF generation, historical replay, or live inference.

- Complete the authorized workload. Do not silently reduce history, features, folds, trials, scenarios, or precision to make an implementation appear faster.
- Use available local CPU, RAM, GPU, and disk proactively when useful. Routine local computation within the task does not require repeated approval. This does not authorize paid services, cloud provisioning, or unrelated experiments.
- Detect actual resources and active workloads before a heavy run. The user's reference workstation is an i7-13650HX (14 cores / 20 threads), 64 GB RAM, and an RTX 4060 Laptop GPU (8 GB VRAM); other environments may differ.
- Profile representative work, remove repeated computation and unnecessary copies, then choose native operations, batching, JIT, parallelism, or GPU only where justified.
- Bound workers, queues, memory, and library threads together. Preserve headroom for the OS and active live processes; do not terminate the user's jobs or overwrite their environment to improve a benchmark.
- Preserve timestamps, feature order, causal availability, fold boundaries, labels, and trading/accounting semantics. A faster implementation that changes the experiment is a separate methodological change.
- For material optimizations, report end-to-end time, memory, workload, and output equivalence. Documentation-only and unrelated small changes do not need performance benchmarks.

## 7. Git Workflow

**Finish authorized work with focused commits and a normal push to the configured upstream, unless the user says otherwise.**

- Inspect the current branch, upstream, working tree, and existing staged changes before editing or committing. Preserve unrelated and concurrent work.
- Stage only your own intended paths or hunks. Do not use blanket staging, include another contributor's staged changes, or automatically stash their work.
- Keep commits coherent: implementation, relevant tests, and necessary documentation belong together. Avoid mixing unrelated fixes or splitting a single change into unusable intermediate commits.
- Use concise, accurate English commit messages: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, or `chore`, with an optional component scope. Avoid messages such as "update", "final", or "WIP".
- Before committing, review the final and staged diffs, run `git diff --check` and appropriate validation, and inspect additions for secrets and unintended large files. For documentation-only changes, verify content, links, and whitespace; do not launch training or a full backtest.
- Do not commit credentials, private keys, secret URLs, local environment files, raw datasets, caches, model binaries, checkpoints, or bulky generated logs/results by default. Small intentional fixtures, manifests, configurations, and useful reports may be versioned. Adding an ignore rule does not untrack existing files.
- After each completed commit, push the intended branch normally. Verify which commits would be pushed; do not publish unrelated local commits, all branches, or tags as a side effect.
- Never use destructive cleanup, discard user changes, rewrite shared history, or force-push without explicit authorization. Do not bypass protected branches.
- If upstream, access, conflicts, divergence, or branch protection prevent publishing, preserve the work, investigate safely, and report the specific blocker. Do not guess a remote or hide a failed push.
- A push does not authorize merging a PR, releasing, deploying, changing live settings, or placing orders.
- In the completion message, state what changed, relevant validation, commit SHA, and push outcome. Mention remaining changes or blockers accurately.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
