# interp-research

Autonomous interpretability research for CrossExamine.AI — optimizing attribution quality
(text span extraction from SAE activations) for privilege review.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12-attr`). The branch `interp-research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b interp-research/<tag>` from current HEAD.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `interp_prepare.py` — fixed constants, data prep, evaluation harness. Do not modify.
   - `interp_experiment.py` — the file you modify. Span extraction, feature selection, thresholds.
4. **Verify data exists**: Check that `~/.cache/interp-research/` contains data and activations. If not, tell the human to run `uv run interp_prepare.py --synth`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates attribution quality on a held-out test set. Launch it as: `uv run interp_experiment.py`.

**What you CAN do:**
- Modify `interp_experiment.py` — this is the only file you edit. Everything is fair game:
  - Span extraction thresholds (`ACTIVATION_THRESHOLD`, `RELATIVE_THRESHOLD`, `MIN_SPAN_TOKENS`, `MAX_SPAN_GAP`)
  - Feature selection (`N_TOP_FEATURES`, `FEATURE_SCORE_METHOD`, `PRIVILEGE_FEATURE_WEIGHT`)
  - Normalization (`NORMALIZATION_METHOD`)
  - Span scoring (`SPAN_SCORE_AGG`, `MIN_SPAN_SCORE`, `SPAN_EXPANSION_CHARS`)
  - The span extraction algorithm itself (`extract_spans_from_scores`, `extract_attribution_spans`)
  - Token scoring strategy (`score_tokens_for_feature`)
  - Feature combination strategies (how top features are weighted and combined)
  - Probe hyperparameters (secondary metric)
  - Entirely new approaches to span extraction — attention-based, gradient-based, contrastive

**What you CANNOT do:**
- Modify `interp_prepare.py`. It is read-only. It contains the evaluation harness, data loading, and constants.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_attribution` function in `interp_prepare.py` is ground truth.

**The goal: get the highest span_f1.** This measures how well your extracted spans match the gold-standard privilege indicator annotations. Higher is better.

**Secondary metric**: `probe_f1` — classification accuracy. If span_f1 is tied between two experiments, prefer higher probe_f1.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a simplification win.

## Output format

Once the script finishes it prints a summary:

```
---
span_f1:          0.450000
span_precision:   0.500000
span_recall:      0.410000
mean_iou:         0.380000
probe_f1:         0.820000
total_seconds:    12.3
docs_evaluated:   40
```

Extract the key metric: `grep "^span_f1:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Header and 6 columns:

```
commit	span_f1	probe_f1	mean_iou	status	description
```

1. git commit hash (short, 7 chars)
2. span_f1 achieved — use 0.000000 for crashes
3. probe_f1 achieved — use 0.000000 for crashes
4. mean_iou achieved — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	span_f1	probe_f1	mean_iou	status	description
a1b2c3d	0.450000	0.820000	0.380000	keep	baseline
b2c3d4e	0.480000	0.830000	0.400000	keep	lower activation threshold to 0.2
c3d4e5f	0.440000	0.810000	0.370000	discard	increase max_span_gap to 3
d4e5f6g	0.000000	0.000000	0.000000	crash	removed min_span_tokens filter (error)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `interp_experiment.py` with an experimental idea by directly hacking the code
3. git commit
4. Run the experiment: `uv run interp_experiment.py > run.log 2>&1`
5. Read out the results: `grep "^span_f1:\|^probe_f1:\|^mean_iou:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` and attempt a fix.
7. Record the results in the tsv (do NOT commit the results.tsv file)
8. If span_f1 improved (higher), keep the commit
9. If span_f1 is equal or worse, git reset back

## Ideas to try

Here are concrete experimental directions, roughly ordered by expected impact:

### Threshold tuning
- Sweep `ACTIVATION_THRESHOLD` from 0.1 to 0.5
- Try `RELATIVE_THRESHOLD` at 0.1, 0.15, 0.25, 0.3
- Adjust `MIN_SPAN_TOKENS` — try 1, 2, 3
- Increase `MAX_SPAN_GAP` to bridge more gaps (1, 2, 3, 5)

### Feature selection
- Try different `N_TOP_FEATURES` values (5, 10, 20, 50)
- Change `FEATURE_SCORE_METHOD` to "sum" or "mean"
- Weight features by their contribution_direction if available

### Span scoring
- Try `SPAN_SCORE_AGG` = "max" instead of "mean"
- Adjust `MIN_SPAN_SCORE` threshold
- Add context expansion (`SPAN_EXPANSION_CHARS`) for broader matching

### Algorithm changes
- Attention-weighted token scoring: weight tokens by their attention to privilege-signal features
- Contrastive scoring: subtract mean non-privileged activation pattern from each doc
- Multi-scale spans: extract at multiple threshold levels and merge
- Per-feature span extraction: extract spans per feature, then merge across features
- Adaptive thresholding: use per-document statistics instead of global thresholds
- Token scoring with exponential decay from peak activation positions

### Normalization
- Compare "l2" vs "zscore" vs "none" normalization
- Try per-feature normalization (across documents) instead of per-document

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, re-read the code, try combining approaches, try more radical changes. The loop runs until interrupted.
