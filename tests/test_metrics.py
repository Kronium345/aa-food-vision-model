import pytest

from fridgevision.metrics import Detection, GroundTruth, evaluate_detections, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_perfect_detections_give_map_1():
    gt = [GroundTruth("1", "apple", (0, 0, 10, 10)), GroundTruth("1", "lime", (50, 50, 10, 10))]
    dets = [Detection("1", "apple", 0.9, (0, 0, 10, 10)), Detection("1", "lime", 0.8, (50, 50, 10, 10))]
    r = evaluate_detections(gt, dets)
    assert r["map"] == pytest.approx(1.0)
    assert r["per_class"]["apple"]["precision"] == 1.0


def test_duplicate_detection_is_false_positive_and_wrong_label_misses():
    gt = [GroundTruth("1", "apple", (0, 0, 10, 10)), GroundTruth("2", "apple", (0, 0, 10, 10))]
    dets = [
        Detection("1", "apple", 0.9, (0, 0, 10, 10)),
        Detection("1", "apple", 0.8, (1, 1, 10, 10)),  # duplicate -> FP
        Detection("2", "lemon", 0.9, (0, 0, 10, 10)),  # wrong class -> apple missed
    ]
    r = evaluate_detections(gt, dets)
    apple = r["per_class"]["apple"]
    assert apple["recall"] == 0.5
    assert apple["precision"] == 0.5
    # precision 1.0 up to recall 0.5, nothing after -> 51 of 101 recall points
    assert apple["ap"] == pytest.approx(51 / 101)
    assert r["per_class"]["lemon"]["ap"] is None  # no ground truth: not part of mAP
    assert r["classes_evaluated"] == 1


def test_low_iou_is_a_miss():
    gt = [GroundTruth("1", "egg", (0, 0, 10, 10))]
    dets = [Detection("1", "egg", 0.9, (8, 8, 10, 10))]
    assert evaluate_detections(gt, dets)["map"] == 0.0
