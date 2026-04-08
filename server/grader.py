"""
meeting_env/server/grader.py
Deterministic grader for action item extraction tasks.

Scoring strategy:
  - For each ground truth item, find the best-matching agent item.
  - Match is determined by: owner match + keyword overlap in task description.
  - Score per item:
      task keywords  40%  (fraction of GT keywords present in agent's task text)
      owner correct  30%  (binary — wrong owner immediately scores 0)
      deadline match 15%  (substring match, or both null)
      priority match 15%  (exact match after normalisation)
  - Overall score: 0.7 × avg_item_score + 0.3 × F1
  - Partial progress rewarded at every step — not just at episode end.
"""
from typing import List, Dict, Any, Tuple
import re


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _owner_matches(agent_owner: str, gt_owner: str) -> bool:
    """True if the agent identified the correct owner."""
    a = _normalize(agent_owner)
    g = _normalize(gt_owner)
    # Accept if either contains the other (handles "Maya" vs "Maya (Eng)")
    return g in a or a in g or g.split()[0] in a or a.split()[0] in g


def _keyword_score(agent_task: str, keywords: List[str]) -> float:
    """Fraction of ground-truth keywords present in agent's task description."""
    if not keywords:
        return 1.0
    task_norm = _normalize(agent_task)
    hits = sum(1 for kw in keywords if _normalize(kw) in task_norm)
    return hits / len(keywords)


def _deadline_matches(agent_deadline: str | None, gt_deadline: str | None) -> bool:
    """True if deadline is correct (or both are absent)."""
    if gt_deadline is None:
        return True  # No deadline expected — any answer is fine
    if agent_deadline is None:
        return False
    return _normalize(gt_deadline) in _normalize(agent_deadline) or \
           _normalize(agent_deadline) in _normalize(gt_deadline)


def _priority_matches(agent_priority: str, gt_priority: str) -> bool:
    return _normalize(agent_priority) == _normalize(gt_priority)


def _score_single_item(
    agent_item: Dict[str, Any],
    gt_item: Dict[str, Any]
) -> float:
    """
    Score a single (agent_item, gt_item) pair.
    Returns float in [0.0, 1.0].
    """
    owner_ok = _owner_matches(agent_item.get("owner", ""), gt_item["owner"])
    if not owner_ok:
        return 0.0  # Wrong owner = no match for this GT item

    kw_score = _keyword_score(
        agent_item.get("task", ""),
        gt_item.get("keywords", [])
    )
    if kw_score < 0.3:
        return 0.0  # Not describing the right task at all

    deadline_ok = _deadline_matches(
        agent_item.get("deadline"),
        gt_item.get("deadline")
    )
    priority_ok = _priority_matches(
        agent_item.get("priority", "medium"),
        gt_item.get("priority", "medium")
    )

    # Weighted sum:  task keywords 40%, owner confirmed 30%, deadline 15%, priority 15%
    score = (kw_score * 0.40) + (0.30) + (0.15 if deadline_ok else 0.0) + (0.15 if priority_ok else 0.0)
    return round(score, 4)


def grade(
    agent_items: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a full submission against ground truth.

    Returns:
        (score, details)
        score  — float in [0.0, 1.0], the primary reward signal
        details — dict with precision, recall, f1, and per-item breakdown
    """
    n_gt = len(ground_truth)
    n_agent = len(agent_items)

    if n_gt == 0:
        return 1.0, {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": []}

    if n_agent == 0:
        return 0.0, {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, 
            "avg_item_score": 0.0, "final_score": 0.0, 
            "items_found": 0, "items_total": n_gt, "matched": []
        }

    # For each GT item, find the best matching agent item
    matched_pairs = []
    used_agent_indices = set()

    for gt_item in ground_truth:
        best_score = 0.0
        best_idx = -1

        for idx, agent_item in enumerate(agent_items):
            if idx in used_agent_indices:
                continue
            s = _score_single_item(
                {
                    "owner": agent_item.get("owner", ""),
                    "task": agent_item.get("task", ""),
                    "deadline": agent_item.get("deadline"),
                    "priority": agent_item.get("priority", "medium"),
                },
                gt_item
            )
            if s > best_score:
                best_score = s
                best_idx = idx

        matched_pairs.append({
            "gt_id": gt_item["id"],
            "gt_owner": gt_item["owner"],
            "matched_agent_idx": best_idx if best_score > 0.0 else None,
            "item_score": best_score,
        })
        if best_idx >= 0 and best_score > 0.0:
            used_agent_indices.add(best_idx)

    # Precision: of agent items, how many map to a GT item
    matched_count = sum(1 for p in matched_pairs if p["matched_agent_idx"] is not None)
    precision = matched_count / n_agent if n_agent > 0 else 0.0

    # Recall: of GT items, how many were found (score > 0.5 = "found")
    found_count = sum(1 for p in matched_pairs if p["item_score"] >= 0.5)
    recall = found_count / n_gt

    # F1
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Final score = average item score (richer signal than binary F1)
    avg_item_score = sum(p["item_score"] for p in matched_pairs) / n_gt

    # Blend: 70% avg_item_score + 30% F1 for smooth gradient
    final_score = round(0.7 * avg_item_score + 0.3 * f1, 4)

    details = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_item_score": round(avg_item_score, 4),
        "final_score": final_score,
        "items_found": found_count,
        "items_total": n_gt,
        "matched": matched_pairs,
    }

    return final_score, details


def generate_feedback(details: Dict[str, Any], max_steps: int, step: int) -> str:
    """
    Generate non-answer feedback for the agent based on grading details.
    Helpful signal without revealing the ground truth directly.
    """
    found = details["items_found"]
    total = details["items_total"]
    f1 = details["f1"]
    avg_quality = details.get("avg_item_score", 0.0)
    missing = total - found
    remaining_steps = max_steps - step

    # ── Stats line ──────────────────────────────────────────────────────────
    stats = (
        f"Score: {details['final_score']:.2f} | "
        f"Found {found}/{total} items ({missing} still missing) | "
        f"F1: {f1:.2f} | Avg item quality: {avg_quality:.2f} | "
        f"Steps remaining: {remaining_steps}"
    )

    # ── Hint based on F1 band ────────────────────────────────────────────────
    if f1 < 0.3:
        hint = (
            "Hint: You may be missing several action items. "
            "Re-read the transcript carefully for implicit commitments, "
            "agreements that need follow-up, and any tasks where someone "
            "volunteered without being explicitly asked."
        )
    elif f1 < 0.6:
        hint = (
            "Hint: Some items found, but check ownership carefully — "
            "especially where responsibility changed hands mid-conversation. "
            "Also look for tasks that were agreed upon without a direct 'I will...' statement."
        )
    elif f1 < 0.85:
        hint = (
            "Hint: Good progress. Verify deadlines and priorities for the items "
            "you have, and check whether any implicit commitments were overlooked."
        )
    else:
        hint = (
            "Hint: Strong extraction. Double-check you haven't over-extracted — "
            "remove any items without a clear owner or a genuine commitment to act."
        )

    # ── Step urgency ─────────────────────────────────────────────────────────
    if remaining_steps <= 0:
        urgency = "No steps remaining — this was your final submission."
    elif remaining_steps == 1:
        urgency = "Last step — set is_final to true in your response."
    elif remaining_steps == 2:
        urgency = f"Only {remaining_steps} steps remaining — consider finalising soon."
    else:
        urgency = ""

    parts = [stats, hint]
    if urgency:
        parts.append(urgency)
    return "\n".join(parts)

