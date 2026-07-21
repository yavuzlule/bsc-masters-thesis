import re
from collections import Counter
from difflib import SequenceMatcher

def normalize_text(text):
    """Basic normalization for fair comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text):
    """Simple whitespace + punctuation-aware tokenizer."""
    return re.findall(r"\w+|[^\w\s]", normalize_text(text))


def levenshtein_distance(a, b):
    """Classic edit distance."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def precision_recall_f1(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1


def sequence_iou(pred_tokens, gold_tokens):
    matcher = SequenceMatcher(None, pred_tokens, gold_tokens)
    match_blocks = matcher.get_matching_blocks()

    intersection = sum(block.size for block in match_blocks)
    union = len(pred_tokens) + len(gold_tokens) - intersection

    return intersection / union if union else 0.0


def exact_match(pred, gold):
    return normalize_text(pred) == normalize_text(gold)


def boundary_accuracy(pred, gold):
    """Approximate boundary correctness via SequenceMatcher alignment."""
    matcher = SequenceMatcher(None, pred, gold)
    blocks = matcher.get_matching_blocks()

    if not blocks:
        return {"start_match": 0, "end_match": 0}

    best = max(blocks, key=lambda x: x.size)

    pred_start_ok = best.a == 0
    gold_start_ok = best.b == 0

    pred_end_ok = (best.a + best.size == len(pred))
    gold_end_ok = (best.b + best.size == len(gold))

    return {
        "pred_start_match": int(pred_start_ok),
        "pred_end_match": int(pred_end_ok),
        "gold_start_match": int(gold_start_ok),
        "gold_end_match": int(gold_end_ok),
    }


def evaluate_extraction(reference_text, extracted_text, iou_threshold=0.5):
    """
    Full evaluation report for text extraction systems.
    """

    ref_norm = normalize_text(reference_text)
    ext_norm = normalize_text(extracted_text)

    ref_tokens = tokenize(reference_text)
    ext_tokens = tokenize(extracted_text)

    # Exact match
    em = exact_match(extracted_text, reference_text)

    # Token metrics
    p, r, f1 = precision_recall_f1(ext_tokens, ref_tokens)

    # Character metrics
    char_p, char_r, char_f1 = precision_recall_f1(list(ext_norm), list(ref_norm))

    # IoU
    iou = sequence_iou(ext_tokens, ref_tokens)

    # Levenshtein
    lev = levenshtein_distance(list(ext_norm), list(ref_norm))

    # Boundary
    boundary = boundary_accuracy(ext_norm, ref_norm)

    # Threshold correctness
    iou_correct = iou >= iou_threshold

    return {
        "exact_match": em,

        "token_level": {
            "precision": p,
            "recall": r,
            "f1": f1,
        },

        "char_level": {
            "precision": char_p,
            "recall": char_r,
            "f1": char_f1,
        },

        "iou": iou,
        "iou_threshold": iou_threshold,
        "iou_pass": iou_correct,

        "levenshtein_distance": lev,

        "boundary_accuracy": boundary,
    }



"""
LIST-LEVEL COMPARISON
"""
import re
from collections import Counter


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(a, b):
    a_tokens = Counter(normalize(a).split())
    b_tokens = Counter(normalize(b).split())

    overlap = sum((a_tokens & b_tokens).values())

    if sum(a_tokens.values()) == 0:
        return 0.0

    if sum(b_tokens.values()) == 0:
        return 0.0

    precision = overlap / sum(b_tokens.values())
    recall = overlap / sum(a_tokens.values())

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def match_lists(reference, generated):
    """
    Returns one-to-one matches.

    Output:
        [
            (ref_idx, gen_idx, similarity),
            ...
        ]
    """
    candidates = []

    for r_idx, r in enumerate(reference):
        for g_idx, g in enumerate(generated):
            sim = token_f1(r, g)
            candidates.append((sim, r_idx, g_idx))

    candidates.sort(reverse=True)

    used_ref = set()
    used_gen = set()

    matches = []

    for sim, r_idx, g_idx in candidates:
        if r_idx in used_ref:
            continue

        if g_idx in used_gen:
            continue

        used_ref.add(r_idx)
        used_gen.add(g_idx)

        matches.append((r_idx, g_idx, sim))

    return matches

def coverage(reference, generated):
    matches = match_lists(reference, generated)

    if not reference:
        return 1.0

    return sum(sim for _, _, sim in matches) / len(reference)

def precision(reference, generated):
    matches = match_lists(reference, generated)

    if not generated:
        return 0.0

    return sum(sim for _, _, sim in matches) / len(generated)

def recall(reference, generated):
    return coverage(reference, generated)

def list_f1(reference, generated):
    p = precision(reference, generated)
    r = recall(reference, generated)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)

def hallucination_rate(reference, generated, threshold=0.5):
    matches = match_lists(reference, generated)

    matched_generated = {
        g_idx
        for _, g_idx, sim in matches
        if sim >= threshold
    }

    hallucinated = len(generated) - len(matched_generated)

    if not generated:
        return 0.0

    return hallucinated / len(generated)

def missing_rate(reference, generated, threshold=0.5):
    matches = match_lists(reference, generated)

    matched_reference = {
        r_idx
        for r_idx, _, sim in matches
        if sim >= threshold
    }

    missing = len(reference) - len(matched_reference)

    if not reference:
        return 0.0

    return missing / len(reference)

def exact_match_ratio(reference, generated):
    ref_norm = {normalize(x) for x in reference}
    gen_norm = {normalize(x) for x in generated}

    if not reference:
        return 1.0

    return len(ref_norm & gen_norm) / len(ref_norm)

def distinct_items_ratio(generated):
    if not generated:
        return 0.0

    norm = [normalize(x) for x in generated]

    return len(set(norm)) / len(norm)

def evaluate_list(reference, generated):
    p = precision(reference, generated)
    r = recall(reference, generated)
    f1 = list_f1(reference, generated)

    halluc = hallucination_rate(reference, generated)
    missing = missing_rate(reference, generated)

    exact = exact_match_ratio(reference, generated)
    diversity = distinct_items_ratio(generated)

    final_score = (
        0.35 * f1 +
        0.25 * r +
        0.15 * p +
        0.15 * exact +
        0.10 * diversity -
        0.20 * halluc
    )

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "coverage": r,
        "hallucination_rate": halluc,
        "missing_rate": missing,
        "exact_match_ratio": exact,
        "diversity": diversity,
        "final_score": max(0.0, final_score),
    }