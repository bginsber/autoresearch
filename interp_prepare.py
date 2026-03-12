"""
Interp-research data preparation and evaluation infrastructure.

Fixed constants, data prep (privilege corpus + SAE activations), and runtime
evaluation utilities for CrossExamine.AI interpretability research.

This file is NOT modified by the agent — it is the read-only infrastructure.

Usage: uv run interp_prepare.py          # verify real data exists
       uv run interp_prepare.py --synth  # generate synthetic data for development
"""

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (do not modify)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("INTERP_CACHE", "~/.cache/interp-research")).expanduser()
DATA_DIR = CACHE_DIR / "data"
ACTIVATIONS_DIR = CACHE_DIR / "activations"

TIME_BUDGET = 300  # 5 minutes wall clock per experiment
MAX_SEQ_LEN = 2048
HIDDEN_DIM = 4096  # base model hidden dimension (e.g., Llama-8B)
SAE_EXPANSION = 8  # SAE expansion factor
SAE_DIM = HIDDEN_DIM * SAE_EXPANSION  # 32768 SAE features

# Evaluation constants
EVAL_SPLIT_RATIO = 0.2  # held-out test fraction
MIN_SPAN_OVERLAP_IOU = 0.5  # IoU threshold for span matching


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A privilege-labeled document with optional span annotations."""
    doc_id: str
    text: str
    label: str  # "privileged", "not_privileged"
    spans: list[dict] | None = None  # [{"start": int, "end": int, "label": str}]

    @property
    def char_length(self) -> int:
        return len(self.text)


@dataclass
class ActivationCache:
    """Cached hidden states and SAE features for a document set."""
    doc_ids: list[str]
    hidden_states: np.ndarray  # (n_docs, hidden_dim) pooled
    sae_features: np.ndarray  # (n_docs, sae_dim) after pooling
    token_activations: np.ndarray | None = None  # (n_docs, seq_len, sae_dim)
    token_offsets: list[list[tuple[int, int]]] | None = None  # char offsets per doc
    labels: np.ndarray | None = None  # (n_docs,) binary labels


# ---------------------------------------------------------------------------
# Corpus Loading
# ---------------------------------------------------------------------------

def load_corpus(path: Path | None = None) -> list[Document]:
    """Load privilege-labeled document corpus from JSONL.

    Expected format per line:
    {"doc_id": "...", "text": "...", "label": "privileged"|"not_privileged",
     "spans": [{"start": 0, "end": 50, "label": "attorney_communication"}, ...]}
    """
    if path is None:
        path = DATA_DIR / "privilege_corpus.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run `uv run interp_prepare.py --synth` to generate synthetic data."
        )

    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Document(
                doc_id=obj["doc_id"],
                text=obj["text"],
                label=obj["label"],
                spans=obj.get("spans"),
            ))

    logger.info(f"Loaded {len(docs)} documents from {path}")
    return docs


def split_corpus(
    docs: list[Document],
    test_ratio: float = EVAL_SPLIT_RATIO,
    seed: int = 42,
) -> tuple[list[Document], list[Document]]:
    """Split corpus into train/test with stratified sampling."""
    rng = np.random.default_rng(seed)

    privileged = [d for d in docs if d.label == "privileged"]
    not_privileged = [d for d in docs if d.label == "not_privileged"]

    def _split(subset):
        indices = rng.permutation(len(subset))
        n_test = max(1, int(len(subset) * test_ratio))
        test_idx = set(indices[:n_test].tolist())
        train = [subset[i] for i in range(len(subset)) if i not in test_idx]
        test = [subset[i] for i in range(len(subset)) if i in test_idx]
        return train, test

    p_train, p_test = _split(privileged)
    np_train, np_test = _split(not_privileged)

    train = p_train + np_train
    test = p_test + np_test
    rng.shuffle(train)
    rng.shuffle(test)

    logger.info(f"Split: {len(train)} train, {len(test)} test")
    return train, test


# ---------------------------------------------------------------------------
# Activation Caching
# ---------------------------------------------------------------------------

def load_cached_activations(path: Path | None = None) -> ActivationCache:
    """Load pre-extracted activations from disk."""
    if path is None:
        path = ACTIVATIONS_DIR / "activations.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Cached activations not found at {path}. "
            "Run `uv run interp_prepare.py --synth` to generate synthetic data."
        )

    data = np.load(path, allow_pickle=True)

    doc_ids = data["doc_ids"].tolist()
    hidden_states = data["hidden_states"]
    sae_features = data["sae_features"]

    token_activations = data.get("token_activations")
    if token_activations is not None and token_activations.shape == ():
        token_activations = None

    labels = data.get("labels")
    if labels is not None and labels.shape == ():
        labels = None

    token_offsets = None
    raw_offsets = data.get("token_offsets_json")
    if raw_offsets is not None and raw_offsets.shape != ():
        token_offsets = [json.loads(s) for s in raw_offsets]

    return ActivationCache(
        doc_ids=doc_ids,
        hidden_states=hidden_states,
        sae_features=sae_features,
        token_activations=token_activations,
        token_offsets=token_offsets,
        labels=labels,
    )


def save_cached_activations(cache: ActivationCache, path: Path | None = None) -> None:
    """Save activation cache to disk."""
    if path is None:
        path = ACTIVATIONS_DIR / "activations.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "doc_ids": np.array(cache.doc_ids, dtype=object),
        "hidden_states": cache.hidden_states,
        "sae_features": cache.sae_features,
    }
    if cache.token_activations is not None:
        save_dict["token_activations"] = cache.token_activations
    if cache.labels is not None:
        save_dict["labels"] = cache.labels
    if cache.token_offsets is not None:
        save_dict["token_offsets_json"] = np.array(
            [json.dumps(offsets) for offsets in cache.token_offsets], dtype=object
        )

    np.savez_compressed(path, **save_dict)
    logger.info(f"Saved activation cache to {path} ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Evaluation: Attribution Quality (primary metric)
# ---------------------------------------------------------------------------

def compute_span_iou(
    pred: tuple[int, int],
    gold: tuple[int, int],
) -> float:
    """Compute Intersection-over-Union for two character spans."""
    inter_start = max(pred[0], gold[0])
    inter_end = min(pred[1], gold[1])
    intersection = max(0, inter_end - inter_start)
    union = (pred[1] - pred[0]) + (gold[1] - gold[0]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate_attribution(
    predicted_spans: list[list[dict]],
    gold_spans: list[list[dict]],
    iou_threshold: float = MIN_SPAN_OVERLAP_IOU,
) -> dict:
    """Evaluate attribution quality: span precision, recall, F1.

    For each document, matches predicted spans to gold spans using IoU.
    A predicted span is a true positive if it overlaps any gold span
    with IoU >= threshold.

    Args:
        predicted_spans: Per-document list of {"start": int, "end": int, ...}
        gold_spans: Per-document list of {"start": int, "end": int, ...}
        iou_threshold: Minimum IoU for a match

    Returns:
        Dict with span_precision, span_recall, span_f1, mean_iou,
        exact_match_rate, n_docs_evaluated
    """
    total_pred = 0
    total_gold = 0
    total_tp = 0
    total_iou_sum = 0.0
    total_iou_count = 0
    exact_matches = 0
    n_docs = 0

    for pred_doc, gold_doc in zip(predicted_spans, gold_spans, strict=False):
        if not gold_doc:
            continue

        n_docs += 1
        gold_matched = set()

        for p_span in pred_doc:
            total_pred += 1
            best_iou = 0.0
            best_gold_idx = -1

            for g_idx, g_span in enumerate(gold_doc):
                iou = compute_span_iou(
                    (p_span["start"], p_span["end"]),
                    (g_span["start"], g_span["end"]),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gold_idx = g_idx

            total_iou_sum += best_iou
            total_iou_count += 1

            if best_iou >= iou_threshold and best_gold_idx not in gold_matched:
                total_tp += 1
                gold_matched.add(best_gold_idx)

        total_gold += len(gold_doc)

        if len(gold_matched) == len(gold_doc) and len(pred_doc) == len(gold_doc):
            exact_matches += 1

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou_sum / total_iou_count if total_iou_count > 0 else 0.0

    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "mean_iou": mean_iou,
        "exact_match_rate": exact_matches / n_docs if n_docs > 0 else 0.0,
        "n_docs_evaluated": n_docs,
        "true_positives": total_tp,
        "total_predicted": total_pred,
        "total_gold": total_gold,
    }


def evaluate_probe_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Evaluate probe classification quality (secondary metric)."""
    binary_preds = (predictions >= threshold).astype(int)
    tp = int(np.sum((binary_preds == 1) & (labels == 1)))
    fp = int(np.sum((binary_preds == 1) & (labels == 0)))
    fn = int(np.sum((binary_preds == 0) & (labels == 1)))
    tn = int(np.sum((binary_preds == 0) & (labels == 0)))

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Activation Utilities (from CrossExamine activations module)
# ---------------------------------------------------------------------------

def validate_activations(
    activations: np.ndarray,
    expected_hidden_dim: int | None = None,
) -> tuple[bool, str | None]:
    """Validate raw activation array from transformer."""
    if activations.ndim not in (2, 3):
        return False, f"Activations must be 2D or 3D, got shape {activations.shape}"
    hidden_dim = activations.shape[-1]
    if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
        return False, f"Hidden dim mismatch: expected {expected_hidden_dim}, got {hidden_dim}"
    if np.any(np.isnan(activations)):
        return False, "Activations contain NaN values"
    if np.any(np.isinf(activations)):
        return False, "Activations contain infinite values"
    return True, None


def normalize_activations(
    activations: np.ndarray,
    method: str = "l2",
) -> np.ndarray:
    """Normalize activations for probe input."""
    if method == "none":
        return activations
    if method == "l2":
        norms = np.linalg.norm(activations, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return activations / norms
    if method == "zscore":
        mean = np.mean(activations, axis=-1, keepdims=True)
        std = np.std(activations, axis=-1, keepdims=True)
        std = np.maximum(std, 1e-10)
        return (activations - mean) / std
    raise ValueError(f"Unknown normalization method: {method}")


def get_top_activations(
    features: np.ndarray,
    n: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract top-N activated features from SAE feature vector."""
    if features.ndim != 1:
        raise ValueError(f"Expected 1D features, got shape {features.shape}")
    n = min(n, len(features))
    top_indices = np.argsort(features)[-n:][::-1]
    top_scores = features[top_indices]
    return top_indices.astype(np.int32), top_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

PRIVILEGE_TEMPLATES = [
    "Dear {attorney}, I am writing to seek your legal advice regarding {topic}. "
    "As discussed in our privileged consultation on {date}, the matter involves {detail}. "
    "Please advise on the best course of action. Regards, {client}",

    "PRIVILEGED AND CONFIDENTIAL — ATTORNEY-CLIENT COMMUNICATION\n"
    "From: {attorney}\nTo: {client}\nRe: {topic}\n\n"
    "Based on my legal analysis, I recommend the following course of action: {detail}. "
    "This advice is protected by attorney-client privilege.",

    "MEMORANDUM — ATTORNEY WORK PRODUCT\n"
    "Prepared by: {attorney}\nDate: {date}\nRe: {topic}\n\n"
    "Legal Analysis: After reviewing the relevant case law and statutes, "
    "it is my professional opinion that {detail}. "
    "This memorandum reflects attorney mental impressions and legal theories.",

    "CONFIDENTIAL LEGAL MEMORANDUM\n"
    "From: {attorney}\nDate: {date}\nSubject: {topic}\n\n"
    "I have completed my legal analysis of the issues raised in our last privileged consultation. "
    "My professional opinion is that {detail}. "
    "I recommend we proceed cautiously given the regulatory landscape. "
    "This communication is protected by attorney-client privilege and work product doctrine.",

    "Dear {client},\n\nThank you for consulting with me regarding {topic}. "
    "After careful legal analysis, here is my advice: {detail}. "
    "Please note that this legal advice is confidential and privileged. "
    "Do not forward this communication outside of our attorney-client relationship.\n\n"
    "Best regards,\n{attorney}",
]

NON_PRIVILEGE_TEMPLATES = [
    "Meeting minutes — {date}\nAttendees: {names}\nTopic: {topic}\n\n"
    "The team discussed {detail}. Action items were assigned. "
    "Next meeting scheduled for next week.",

    "Invoice #{invoice_num}\nFrom: {vendor}\nTo: {company}\nDate: {date}\n\n"
    "Services rendered: {detail}\nAmount due: ${amount}\nPayment terms: Net 30",

    "Email from {sender} to {recipient}\nDate: {date}\nSubject: {topic}\n\n"
    "Hi, just following up on {detail}. Let me know if you have any questions. Thanks!",

    "QUARTERLY BUSINESS REVIEW — {date}\n"
    "Department: Operations\nPrepared by: {sender}\n\n"
    "Performance Summary: {detail}. Revenue targets were met. "
    "Headcount remains stable. No significant issues to report.",

    "VENDOR AGREEMENT\nBetween: {vendor} and {company}\nEffective: {date}\n\n"
    "Scope of services: {detail}. Standard commercial terms apply. "
    "This agreement supersedes all prior agreements between the parties.",
]


def _fill_template(template: str, rng: np.random.Generator) -> str:
    """Fill a template with random but realistic-ish values."""
    attorneys = ["Sarah Chen, Esq.", "James Morrison, Esq.", "Patricia Williams, J.D.",
                 "Robert Kim, Esq.", "Maria Santos, J.D."]
    clients = ["John Smith", "Acme Corp", "TechStart LLC", "Global Industries",
               "Meridian Holdings", "Pacific Ventures"]
    topics = [
        "patent infringement claim", "merger compliance review",
        "employment dispute", "regulatory investigation",
        "contract breach litigation", "IP licensing agreement",
        "data privacy audit", "antitrust inquiry",
        "securities disclosure obligations", "trade secret misappropriation",
    ]
    details = [
        "the potential exposure under Section 10(b) of the Securities Exchange Act",
        "our defensive strategy for the upcoming deposition schedule",
        "the implications of the recent Supreme Court ruling on our position",
        "document retention obligations under the litigation hold",
        "the risk assessment for proceeding to trial versus settlement",
        "compliance requirements under the new regulatory framework",
        "potential liability from the whistleblower allegations",
        "the enforceability of the non-compete provisions",
    ]
    names = ["Alice, Bob, Carol", "Marketing Team", "Engineering leads", "Board members"]
    vendors = ["Consulting Group LLC", "Office Solutions Inc.", "DataTech Services"]

    result = template
    result = result.replace("{attorney}", str(rng.choice(attorneys)))
    result = result.replace("{client}", str(rng.choice(clients)))
    result = result.replace("{topic}", str(rng.choice(topics)))
    result = result.replace("{detail}", str(rng.choice(details)))
    result = result.replace("{date}", f"202{rng.integers(3,6)}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}")
    result = result.replace("{names}", str(rng.choice(names)))
    result = result.replace("{vendor}", str(rng.choice(vendors)))
    result = result.replace("{company}", str(rng.choice(clients)))
    result = result.replace("{sender}", str(rng.choice(["Mike", "Lisa", "HR Dept"])))
    result = result.replace("{recipient}", str(rng.choice(["Team", "All Staff", "Management"])))
    result = result.replace("{invoice_num}", str(rng.integers(10000, 99999)))
    result = result.replace("{amount}", f"{rng.integers(500, 50000):,}")
    return result


def _generate_span_annotations(text: str, label: str, rng: np.random.Generator) -> list[dict]:
    """Generate plausible span annotations for a document."""
    if label == "not_privileged":
        return []

    spans = []
    privilege_phrases = [
        "legal advice", "privileged consultation", "attorney-client privilege",
        "PRIVILEGED AND CONFIDENTIAL", "ATTORNEY-CLIENT COMMUNICATION",
        "ATTORNEY WORK PRODUCT", "legal analysis", "professional opinion",
        "attorney mental impressions", "legal theories", "I recommend",
        "Based on my legal analysis", "CONFIDENTIAL LEGAL MEMORANDUM",
        "work product doctrine", "confidential and privileged",
    ]

    for phrase in privilege_phrases:
        idx = text.find(phrase)
        if idx >= 0:
            spans.append({
                "start": idx,
                "end": idx + len(phrase),
                "label": "privilege_indicator",
            })

    if not spans and len(text) > 100:
        start = int(rng.integers(0, len(text) // 2))
        end = min(start + int(rng.integers(30, 100)), len(text))
        spans.append({"start": start, "end": end, "label": "privilege_indicator"})

    return spans


def generate_synthetic_corpus(n_docs: int = 200, seed: int = 42) -> list[Document]:
    """Generate synthetic privilege-labeled documents for development."""
    rng = np.random.default_rng(seed)
    docs = []

    n_privileged = n_docs // 2
    n_not_privileged = n_docs - n_privileged

    for i in range(n_privileged):
        template = str(rng.choice(PRIVILEGE_TEMPLATES))
        text = _fill_template(template, rng)
        spans = _generate_span_annotations(text, "privileged", rng)
        doc_id = hashlib.md5(f"priv_{i}_{seed}".encode()).hexdigest()[:12]
        docs.append(Document(doc_id=doc_id, text=text, label="privileged", spans=spans))

    for i in range(n_not_privileged):
        template = str(rng.choice(NON_PRIVILEGE_TEMPLATES))
        text = _fill_template(template, rng)
        spans = _generate_span_annotations(text, "not_privileged", rng)
        doc_id = hashlib.md5(f"nopriv_{i}_{seed}".encode()).hexdigest()[:12]
        docs.append(Document(doc_id=doc_id, text=text, label="not_privileged", spans=spans))

    rng.shuffle(docs)
    logger.info(f"Generated {len(docs)} synthetic documents ({n_privileged} privileged)")
    return docs


def generate_synthetic_activations(
    docs: list[Document],
    hidden_dim: int = HIDDEN_DIM,
    sae_dim: int = SAE_DIM,
    seq_len: int = 64,
    seed: int = 42,
) -> ActivationCache:
    """Generate synthetic activations for development/testing.

    Creates structurally correct activations with class-dependent signal:
    privileged docs have different feature patterns than non-privileged,
    and privilege-signal features concentrate near annotated spans.
    """
    rng = np.random.default_rng(seed)
    n = len(docs)

    hidden_states = rng.standard_normal((n, hidden_dim)).astype(np.float32) * 0.1

    sae_features = np.zeros((n, sae_dim), dtype=np.float32)

    # Define signal features
    priv_features = list(range(100, 150))
    nopriv_features = list(range(200, 250))

    for i, doc in enumerate(docs):
        # Sparse random activations
        active = rng.choice(sae_dim, size=int(rng.integers(20, 80)), replace=False)
        sae_features[i, active] = rng.exponential(0.3, size=len(active)).astype(np.float32)

        # Class-dependent signal
        if doc.label == "privileged":
            signal_feats = rng.choice(priv_features, size=int(rng.integers(5, 15)), replace=False)
            sae_features[i, signal_feats] += rng.uniform(0.5, 2.0, size=len(signal_feats)).astype(np.float32)
        else:
            signal_feats = rng.choice(nopriv_features, size=int(rng.integers(5, 15)), replace=False)
            sae_features[i, signal_feats] += rng.uniform(0.5, 2.0, size=len(signal_feats)).astype(np.float32)

    # Token-level activations for attribution
    token_activations = np.zeros((n, seq_len, sae_dim), dtype=np.float32)
    token_offsets = []

    for i, doc in enumerate(docs):
        text_len = len(doc.text)
        chars_per_token = max(1, text_len // seq_len)

        offsets = []
        for t in range(seq_len):
            start = t * chars_per_token
            end = min((t + 1) * chars_per_token, text_len)
            offsets.append((start, end))
        token_offsets.append(offsets)

        # Spread document features across tokens
        active_feats = np.flatnonzero(sae_features[i] > 0)
        for feat_idx in active_feats:
            n_active_tokens = int(rng.integers(1, max(2, seq_len // 4)))
            active_tokens = rng.choice(seq_len, size=n_active_tokens, replace=False)
            base_score = sae_features[i, feat_idx]
            token_activations[i, active_tokens, feat_idx] = (
                base_score * rng.uniform(0.3, 1.0, size=n_active_tokens).astype(np.float32)
            )

        # Concentrate privilege-signal features near annotated spans
        if doc.label == "privileged" and doc.spans:
            for span in doc.spans:
                span_start_token = span["start"] // chars_per_token
                span_end_token = min(span["end"] // chars_per_token + 1, seq_len)
                span_tokens = list(range(span_start_token, span_end_token))
                if span_tokens:
                    for feat_idx in rng.choice(priv_features, size=min(5, len(priv_features)), replace=False):
                        token_activations[i, span_tokens, feat_idx] += rng.uniform(
                            0.5, 1.5, size=len(span_tokens)
                        ).astype(np.float32)

    labels = np.array([1 if d.label == "privileged" else 0 for d in docs], dtype=np.int32)

    return ActivationCache(
        doc_ids=[d.doc_id for d in docs],
        hidden_states=hidden_states,
        sae_features=sae_features,
        token_activations=token_activations,
        token_offsets=token_offsets,
        labels=labels,
    )


def save_corpus(docs: list[Document], path: Path | None = None) -> None:
    """Save corpus to JSONL."""
    if path is None:
        path = DATA_DIR / "privilege_corpus.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for doc in docs:
            obj = {"doc_id": doc.doc_id, "text": doc.text, "label": doc.label}
            if doc.spans is not None:
                obj["spans"] = doc.spans
            f.write(json.dumps(obj) + "\n")
    logger.info(f"Saved {len(docs)} documents to {path}")


# ---------------------------------------------------------------------------
# Main: Data Preparation
# ---------------------------------------------------------------------------

def prepare_synthetic(n_docs: int = 200, seed: int = 42):
    """Generate synthetic data for development."""
    print(f"Generating {n_docs} synthetic documents...")
    docs = generate_synthetic_corpus(n_docs=n_docs, seed=seed)

    print("Saving corpus...")
    save_corpus(docs)

    print("Generating synthetic activations...")
    cache = generate_synthetic_activations(docs, seed=seed)

    print("Saving activation cache...")
    save_cached_activations(cache)

    train, test = split_corpus(docs)
    priv_train = sum(1 for d in train if d.label == "privileged")
    priv_test = sum(1 for d in test if d.label == "privileged")
    annotated = sum(1 for d in docs if d.spans)

    print(f"\n--- Data Summary ---")
    print(f"Total documents:     {len(docs)}")
    print(f"Train/test split:    {len(train)} / {len(test)}")
    print(f"Privileged (train):  {priv_train}/{len(train)}")
    print(f"Privileged (test):   {priv_test}/{len(test)}")
    print(f"With span annots:    {annotated}")
    print(f"Hidden dim:          {HIDDEN_DIM}")
    print(f"SAE dim:             {SAE_DIM}")
    print(f"Cache location:      {CACHE_DIR}")


def prepare_real():
    """Verify real data exists and print stats."""
    print("Checking for real data...")

    corpus_path = DATA_DIR / "privilege_corpus.jsonl"
    activations_path = ACTIVATIONS_DIR / "activations.npz"

    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Place your privilege_corpus.jsonl there, or run with --synth for synthetic data.")
        sys.exit(1)

    if not activations_path.exists():
        print(f"ERROR: Activations not found at {activations_path}")
        print("Extract activations from your base model and SAE first.")
        sys.exit(1)

    docs = load_corpus()
    cache = load_cached_activations()
    train, test = split_corpus(docs)

    print(f"\n--- Data Summary ---")
    print(f"Total documents:     {len(docs)}")
    print(f"Train/test:          {len(train)} / {len(test)}")
    print(f"SAE features shape:  {cache.sae_features.shape}")
    print(f"Token activations:   {'yes' if cache.token_activations is not None else 'no'}")
    print(f"Token offsets:       {'yes' if cache.token_offsets is not None else 'no'}")
    print("Data OK.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "--synth" in sys.argv:
        n = 200
        for i, arg in enumerate(sys.argv):
            if arg == "--n" and i + 1 < len(sys.argv):
                n = int(sys.argv[i + 1])
        prepare_synthetic(n_docs=n)
    else:
        prepare_real()
