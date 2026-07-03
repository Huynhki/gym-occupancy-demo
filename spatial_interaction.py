"""
Spatial Interaction Score module (section V.4)
S(t) = alpha * O(t) + beta * I(t) + gamma * D(t)

O(t): Overlap score   = area(person_bbox ∩ ROI) / area(person_bbox)
I(t): Inside score    = does the active body keypoint group fall in the ROI
D(t): Distance score  = normalized distance from active keypoint to ROI center

Weights switch dynamically based on keypoint confidence (occlusion-aware),
exactly as described in V.4.d.
"""
import math


def bbox_overlap_over_person(person_bbox, roi_bbox):
    """O(t): area(person ∩ roi) / area(person)."""
    px0, py0, px1, py1 = person_bbox
    rx0, ry0, rx1, ry1 = roi_bbox
    ix0, iy0 = max(px0, rx0), max(py0, ry0)
    ix1, iy1 = min(px1, rx1), min(py1, ry1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    person_area = max(1e-6, (px1 - px0) * (py1 - py0))
    return float(min(1.0, inter / person_area))


def point_in_bbox(x, y, bbox):
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def inside_score(keypoints, keypoint_conf, roi_bbox, keypoint_rules, conf_thresh=0.5):
    """
    I(t): walk the priority-ordered keypoint groups (Primary / Fallback1 /
    Fallback2) defined for the equipment category, use the first group with
    sufficient average confidence, check whether its (averaged) location
    falls inside the ROI bbox, and return the group's associated score.
    Also returns the (x, y) of the chosen "active keypoint" for D(t), and
    the confidence used (for the dynamic-weight switch in V.4.d).
    """
    best = (0.0, None, 0.0)  # score, active_point, used_confidence
    for rule in keypoint_rules:
        ids = rule["ids"]
        pts = [keypoints[i] for i in ids if i < len(keypoints)]
        confs = [keypoint_conf[i] for i in ids if i < len(keypoint_conf)]
        if not pts or not confs:
            continue
        avg_conf = sum(confs) / len(confs)
        if avg_conf < 0.05:  # essentially undetected
            continue
        ax = sum(p[0] for p in pts) / len(pts)
        ay = sum(p[1] for p in pts) / len(pts)
        inside = point_in_bbox(ax, ay, roi_bbox)
        score = rule["score"] if inside else 0.0
        # Prefer the first rule group (by priority) that has decent confidence,
        # matching the proposal's "primary, else fallback 1, else fallback 2" logic.
        if avg_conf >= conf_thresh:
            return score, (ax, ay), avg_conf
        # keep best-so-far low-confidence candidate in case nothing clears thresh
        if score >= best[0]:
            best = (score, (ax, ay), avg_conf)
    return best


def distance_score(active_point, roi_center, roi_diag):
    """D(t) = max(0, 1 - d / d_max), d_max = ROI diagonal."""
    if active_point is None or roi_diag <= 1e-6:
        return 0.0
    dx = active_point[0] - roi_center[0]
    dy = active_point[1] - roi_center[1]
    d = math.hypot(dx, dy)
    return max(0.0, 1.0 - d / roi_diag)


def dynamic_weights(used_confidence, conf_thresh=0.5):
    """
    Section V.4.d - dynamic weight switching:
      normal (conf >= 0.5):    alpha=0.3, beta=0.5, gamma=0.2
      occluded (conf < 0.5):   alpha=0.2, beta=0.3, gamma=0.5
    NOTE: the proposal text names O/I/D with weights alpha/beta/gamma in the
    general formula, and in the occlusion case raises trust in Overlap+Distance
    over Inside (since keypoints are unreliable). We mirror that intent here.
    """
    if used_confidence >= conf_thresh:
        return dict(alpha=0.3, beta=0.5, gamma=0.2)
    return dict(alpha=0.2, beta=0.3, gamma=0.5)


def interaction_score(person_bbox, keypoints, keypoint_conf, roi_entry, conf_thresh=0.5):
    """
    Full S(t) for one (person, equipment) pair at one frame.
    roi_entry: one value from roi.build_roi(...) dict (has bbox/center/diag/keypoint_rules)
    Returns (S, components_dict) for debuggability.
    """
    roi_bbox = roi_entry["bbox"]
    roi_center = roi_entry["center"]
    roi_diag = roi_entry["diag"]
    rules = roi_entry["keypoint_rules"]

    O = bbox_overlap_over_person(person_bbox, roi_bbox)
    I, active_pt, used_conf = inside_score(keypoints, keypoint_conf, roi_bbox, rules, conf_thresh)
    D = distance_score(active_pt, roi_center, roi_diag)

    w = dynamic_weights(used_conf, conf_thresh)
    S = w["alpha"] * O + w["beta"] * I + w["gamma"] * D
    return S, {"O": O, "I": I, "D": D, "weights": w, "active_point": active_pt}
