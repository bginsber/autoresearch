"""
Interp-research experiment script — Attribution Quality Track.

This is the file the agent edits. Contains span extraction algorithms,
feature selection, thresholds, and attribution pipeline configuration.

Usage: uv run interp_experiment.py > run.log 2>&1
"""

import time
import sys

import numpy as np

from interp_prepare import (
    TIME_BUDGET,
    SAE_DIM,
    load_corpus,
    load_cached_activations,
    split_corpus,
    evaluate_attribution,
    evaluate_probe_f1,
    normalize_activations,
    get_top_activations,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Span extraction thresholds
ACTIVATION_THRESHOLD = 0.3       # minimum activation score for a token to be "active"
RELATIVE_THRESHOLD = 0.2         # fraction of max activation — tokens below this are filtered
MIN_SPAN_TOKENS = 2              # minimum consecutive active tokens to form a span
MAX_SPAN_GAP = 1                 # merge spans separated by <= this many inactive tokens
MAX_SPANS_PER_DOC = 10           # cap on number of spans returned per document

# Feature selection
N_TOP_FEATURES = 20              # number of top SAE features to consider per document
FEATURE_SCORE_METHOD = "max"     # "max", "mean", "sum" — how to score a feature across tokens
PRIVILEGE_FEATURE_WEIGHT = 1.5   # upweight features known to indicate privilege direction

# Normalization
NORMALIZATION_METHOD = "l2"      # "l2", "zscore", "none"

# Span scoring
SPAN_SCORE_AGG = "mean"          # "mean", "max", "sum" — how to aggregate token scores in a span
MIN_SPAN_SCORE = 0.1             # minimum aggregated score for a span to be kept
SPAN_EXPANSION_CHARS = 0         # expand spans by N chars on each side (context padding)

# Probe (secondary metric — lightweight linear probe for classification)
PROBE_THRESHOLD = 0.5            # decision threshold for privilege classification
PROBE_REGULARIZATION = 1.0       # L2 regularization strength for logistic regression

# ---------------------------------------------------------------------------
# Span Extraction Algorithm
# ---------------------------------------------------------------------------


def score_tokens_for_feature(
    token_activations: np.ndarray,
    feature_idx: int,
    method: str = FEATURE_SCORE_METHOD,
) -> np.ndarray:
    """Score each token's activation for a specific SAE feature.

    Args:
        token_activations: (seq_len, sae_dim) array
        feature_idx: SAE feature index
        method: Scoring method

    Returns:
        (seq_len,) score array
    """
    return token_activations[:, feature_idx]


def extract_spans_from_scores(
    token_scores: np.ndarray,
    token_offsets: list[tuple[int, int]],
    document_text: str,
    abs_threshold: float = ACTIVATION_THRESHOLD,
    rel_threshold: float = RELATIVE_THRESHOLD,
    min_tokens: int = MIN_SPAN_TOKENS,
    max_gap: int = MAX_SPAN_GAP,
    score_agg: str = SPAN_SCORE_AGG,
    min_score: float = MIN_SPAN_SCORE,
    expansion: int = SPAN_EXPANSION_CHARS,
) -> list[dict]:
    """Extract character-level spans from token-level activation scores.

    Algorithm:
    1. Threshold: mark tokens as active if score >= max(abs_threshold, rel_threshold * max_score)
    2. Merge: group consecutive active tokens, bridging gaps <= max_gap
    3. Filter: remove spans shorter than min_tokens
    4. Score: aggregate token scores within each span
    5. Expand: optionally pad spans by N characters for context

    Args:
        token_scores: (seq_len,) activation scores per token
        token_offsets: character offset mapping [(start, end), ...]
        document_text: original document text
        abs_threshold: absolute minimum activation score
        rel_threshold: fraction of max score for relative threshold
        min_tokens: minimum span length in tokens
        max_gap: maximum gap (inactive tokens) to bridge when merging
        score_agg: aggregation method for span scores
        min_score: minimum aggregated score for a span
        expansion: character expansion on each side

    Returns:
        List of {"start": int, "end": int, "score": float, "text": str}
    """
    if len(token_scores) == 0:
        return []

    max_score = np.max(token_scores)
    if max_score <= 0:
        return []

    # Step 1: Threshold
    threshold = max(abs_threshold, rel_threshold * max_score)
    active = token_scores >= threshold

    # Step 2: Find runs of active tokens, merging gaps
    spans_raw = []
    current_start = None
    gap_count = 0

    for t in range(len(active)):
        if active[t]:
            if current_start is None:
                current_start = t
            gap_count = 0
        else:
            if current_start is not None:
                gap_count += 1
                if gap_count > max_gap:
                    end_t = t - gap_count
                    spans_raw.append((current_start, end_t + 1))
                    current_start = None
                    gap_count = 0

    # Close final span
    if current_start is not None:
        end_t = len(active) - 1
        while end_t >= current_start and not active[end_t]:
            end_t -= 1
        if end_t >= current_start:
            spans_raw.append((current_start, end_t + 1))

    # Step 3: Filter by minimum token length
    spans_raw = [(s, e) for s, e in spans_raw if (e - s) >= min_tokens]

    # Step 4: Score and convert to character offsets
    spans = []
    for tok_start, tok_end in spans_raw:
        # Aggregate score
        span_scores = token_scores[tok_start:tok_end]
        if score_agg == "mean":
            agg_score = float(np.mean(span_scores))
        elif score_agg == "max":
            agg_score = float(np.max(span_scores))
        elif score_agg == "sum":
            agg_score = float(np.sum(span_scores))
        else:
            agg_score = float(np.mean(span_scores))

        if agg_score < min_score:
            continue

        # Convert to character offsets
        if tok_start >= len(token_offsets) or tok_end - 1 >= len(token_offsets):
            continue

        char_start = token_offsets[tok_start][0]
        char_end = token_offsets[tok_end - 1][1]

        # Step 5: Expand
        if expansion > 0:
            char_start = max(0, char_start - expansion)
            char_end = min(len(document_text), char_end + expansion)

        text = document_text[char_start:char_end]

        spans.append({
            "start": char_start,
            "end": char_end,
            "score": agg_score,
            "text": text,
        })

    # Sort by score descending
    spans.sort(key=lambda s: s["score"], reverse=True)
    return spans[:MAX_SPANS_PER_DOC]


def extract_attribution_spans(
    token_activations: np.ndarray,
    sae_features: np.ndarray,
    token_offsets: list[tuple[int, int]],
    document_text: str,
    n_top_features: int = N_TOP_FEATURES,
) -> list[dict]:
    """Extract attributed text spans for a single document.

    Pipeline:
    1. Identify top-N activated SAE features from document-level features
    2. For each top feature, score all tokens
    3. Combine token scores across features (weighted by feature importance)
    4. Extract spans from combined score map

    Args:
        token_activations: (seq_len, sae_dim) per-token SAE activations
        sae_features: (sae_dim,) document-level SAE feature vector
        token_offsets: [(start, end), ...] character offsets
        document_text: original text
        n_top_features: number of top features to use

    Returns:
        List of span dicts
    """
    # Get top features
    top_indices, top_scores = get_top_activations(sae_features, n=n_top_features)

    if len(top_indices) == 0:
        return []

    # Normalize top scores to use as weights
    max_top = np.max(top_scores)
    if max_top > 0:
        feature_weights = top_scores / max_top
    else:
        feature_weights = np.ones_like(top_scores)

    # Combine token scores across top features
    seq_len = token_activations.shape[0]
    combined_scores = np.zeros(seq_len, dtype=np.float32)

    for feat_idx, weight in zip(top_indices, feature_weights, strict=False):
        token_scores = score_tokens_for_feature(token_activations, int(feat_idx))
        combined_scores += token_scores * weight

    # Normalize combined scores
    max_combined = np.max(combined_scores)
    if max_combined > 0:
        combined_scores = combined_scores / max_combined

    # Extract spans
    return extract_spans_from_scores(
        token_scores=combined_scores,
        token_offsets=token_offsets,
        document_text=document_text,
    )


# ---------------------------------------------------------------------------
# Lightweight Probe (secondary metric)
# ---------------------------------------------------------------------------


def train_and_evaluate_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train a simple logistic regression probe and evaluate.

    Args:
        train_features: (n_train, sae_dim) feature matrix
        train_labels: (n_train,) binary labels
        test_features: (n_test, sae_dim) feature matrix
        test_labels: (n_test,) binary labels

    Returns:
        Probe evaluation dict
    """
    # Simple logistic regression via gradient descent (no sklearn dependency)
    n_features = train_features.shape[1]
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    lr = 0.01
    reg = PROBE_REGULARIZATION

    for epoch in range(100):
        logits = train_features @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        # Gradient
        error = probs - train_labels.astype(np.float64)
        grad_w = (train_features.T @ error) / len(train_labels) + reg * weights
        grad_b = np.mean(error)

        weights -= lr * grad_w
        bias -= lr * grad_b

    # Evaluate on test set
    test_logits = test_features @ weights + bias
    test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -500, 500)))

    return evaluate_probe_f1(test_probs, test_labels, threshold=PROBE_THRESHOLD)


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------


def run_experiment():
    t_start = time.time()

    print("Loading data...")
    docs = load_corpus()
    cache = load_cached_activations()
    train_docs, test_docs = split_corpus(docs)

    # Map doc_ids to indices in cache
    doc_id_to_idx = {did: i for i, did in enumerate(cache.doc_ids)}

    # Normalize SAE features
    print(f"Normalizing activations (method={NORMALIZATION_METHOD})...")
    normalized_features = normalize_activations(cache.sae_features, method=NORMALIZATION_METHOD)

    # ---------------------------------------------------------------------------
    # Attribution evaluation
    # ---------------------------------------------------------------------------
    print("\n=== Attribution Quality Evaluation ===")

    predicted_spans_all = []
    gold_spans_all = []
    n_evaluated = 0

    for doc in test_docs:
        idx = doc_id_to_idx.get(doc.doc_id)
        if idx is None:
            predicted_spans_all.append([])
            gold_spans_all.append(doc.spans or [])
            continue

        # Check time budget
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            print(f"Time budget exceeded ({elapsed:.0f}s > {TIME_BUDGET}s), stopping early")
            break

        # Get token-level activations
        if cache.token_activations is None or cache.token_offsets is None:
            predicted_spans_all.append([])
            gold_spans_all.append(doc.spans or [])
            continue

        token_acts = cache.token_activations[idx]
        token_offs = cache.token_offsets[idx]
        doc_features = normalized_features[idx]

        # Run span extraction
        pred_spans = extract_attribution_spans(
            token_activations=token_acts,
            sae_features=doc_features,
            token_offsets=token_offs,
            document_text=doc.text,
            n_top_features=N_TOP_FEATURES,
        )

        predicted_spans_all.append(pred_spans)
        gold_spans_all.append(doc.spans or [])
        n_evaluated += 1

    # Evaluate attribution
    attr_metrics = evaluate_attribution(predicted_spans_all, gold_spans_all)

    print(f"\nAttribution results ({n_evaluated} docs evaluated):")
    print(f"  span_precision: {attr_metrics['span_precision']:.6f}")
    print(f"  span_recall:    {attr_metrics['span_recall']:.6f}")
    print(f"  span_f1:        {attr_metrics['span_f1']:.6f}")
    print(f"  mean_iou:       {attr_metrics['mean_iou']:.6f}")
    print(f"  exact_match:    {attr_metrics['exact_match_rate']:.6f}")

    # ---------------------------------------------------------------------------
    # Probe evaluation (secondary)
    # ---------------------------------------------------------------------------
    print("\n=== Probe Evaluation (secondary) ===")

    train_indices = [doc_id_to_idx[d.doc_id] for d in train_docs if d.doc_id in doc_id_to_idx]
    test_indices = [doc_id_to_idx[d.doc_id] for d in test_docs if d.doc_id in doc_id_to_idx]

    train_features = normalized_features[train_indices]
    test_features = normalized_features[test_indices]
    train_labels = cache.labels[train_indices]
    test_labels = cache.labels[test_indices]

    probe_metrics = train_and_evaluate_probe(
        train_features, train_labels, test_features, test_labels
    )

    print(f"  probe_accuracy:  {probe_metrics['accuracy']:.6f}")
    print(f"  probe_f1:        {probe_metrics['f1']:.6f}")
    print(f"  probe_precision: {probe_metrics['precision']:.6f}")
    print(f"  probe_recall:    {probe_metrics['recall']:.6f}")

    # ---------------------------------------------------------------------------
    # Summary (parseable output)
    # ---------------------------------------------------------------------------
    t_total = time.time() - t_start
    print(f"\n---")
    print(f"span_f1:          {attr_metrics['span_f1']:.6f}")
    print(f"span_precision:   {attr_metrics['span_precision']:.6f}")
    print(f"span_recall:      {attr_metrics['span_recall']:.6f}")
    print(f"mean_iou:         {attr_metrics['mean_iou']:.6f}")
    print(f"probe_f1:         {probe_metrics['f1']:.6f}")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"docs_evaluated:   {n_evaluated}")


if __name__ == "__main__":
    run_experiment()
