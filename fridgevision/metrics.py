"""Detection metrics (mAP@IoU) in pure Python, so they are testable anywhere.

Boxes are COCO-style [x, y, width, height] in pixels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    label: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    image_id: str
    label: str
    score: float
    box: tuple[float, float, float, float]


def iou(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    """COCO-style 101-point interpolated AP."""
    if not recalls:
        return 0.0
    # Make precision monotonically non-increasing from the right.
    prec = precisions[:]
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    total = 0.0
    j = 0
    for k in range(101):
        r = k / 100
        while j < len(recalls) and recalls[j] < r:
            j += 1
        total += prec[j] if j < len(recalls) else 0.0
    return total / 101


def evaluate_detections(
    ground_truth: list[GroundTruth],
    detections: list[Detection],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict:
    """Per-class AP plus precision/recall at `score_threshold`.

    mAP averages over classes that have at least one ground-truth box.
    """
    gt_by_key: dict[tuple[str, str], list[GroundTruth]] = defaultdict(list)
    gt_count: dict[str, int] = defaultdict(int)
    for g in ground_truth:
        gt_by_key[(g.image_id, g.label)].append(g)
        gt_count[g.label] += 1

    dets_by_label: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        dets_by_label[d.label].append(d)

    per_class: dict[str, dict] = {}
    for label in sorted(set(gt_count) | set(dets_by_label)):
        dets = sorted(dets_by_label.get(label, []), key=lambda d: -d.score)
        matched: dict[tuple[str, str], list[bool]] = {
            k: [False] * len(v) for k, v in gt_by_key.items() if k[1] == label
        }
        tp_flags: list[bool] = []
        for d in dets:
            candidates = gt_by_key.get((d.image_id, label), [])
            best, best_iou = -1, iou_threshold
            for idx, g in enumerate(candidates):
                if matched[(d.image_id, label)][idx]:
                    continue
                o = iou(d.box, g.box)
                if o >= best_iou:
                    best, best_iou = idx, o
            if best >= 0:
                matched[(d.image_id, label)][best] = True
            tp_flags.append(best >= 0)

        n_gt = gt_count.get(label, 0)
        tp = fp = 0
        recalls, precisions = [], []
        tp_at_thr = fp_at_thr = 0
        for d, is_tp in zip(dets, tp_flags):
            tp += is_tp
            fp += not is_tp
            recalls.append(tp / n_gt if n_gt else 0.0)
            precisions.append(tp / (tp + fp))
            if d.score >= score_threshold:
                tp_at_thr += is_tp
                fp_at_thr += not is_tp

        per_class[label] = {
            "gt": n_gt,
            "detections": len(dets),
            "ap": _average_precision(recalls, precisions) if n_gt else None,
            "precision": tp_at_thr / (tp_at_thr + fp_at_thr) if (tp_at_thr + fp_at_thr) else None,
            "recall": tp_at_thr / n_gt if n_gt else None,
        }

    aps = [v["ap"] for v in per_class.values() if v["ap"] is not None]
    return {
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "map": sum(aps) / len(aps) if aps else 0.0,
        "classes_evaluated": len(aps),
        "per_class": per_class,
    }
