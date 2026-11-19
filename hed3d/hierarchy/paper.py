"""Reference implementation of the front end described in manuscript Section 2.2."""

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import Ellipse
from .axes import major_first
from .contour import ANGLE_TOL_DEG, thinning
from .fit import STRICT, compute_fit_score


MIN_ARC_OVERLAP_DEG = 10.0
MIN_ABSOLUTE_IMPROVEMENT = 1e-4
# Interpret the unitless value of 1 in Table 1 as one percent.
MIN_RELATIVE_IMPROVEMENT = 0.01
RADIUS_DISCREPANCY_TOLERANCE = 0.15
TANGENT_AGREEMENT_DEG = 10.0
TURN_TOLERANCE_DEG = 10.0
GRADIENT_EPS = 1e-6
# Duplicate detections must agree to within one source-image pixel at their boundary.
DUPLICATE_BOUNDARY_TOLERANCE_PX = 1.0


def color_gradient(image):
    smooth = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    a, b, c = (dx * dx).sum(2), (dx * dy).sum(2), (dy * dy).sum(2)
    delta = np.hypot(a - c, 2 * b)
    return np.sqrt((a + c + delta) / 2), 0.5 * np.arctan2(2 * b, a - c)


def edge_maps(magnitude):
    positive = magnitude[magnitude > GRADIENT_EPS]
    if not positive.size:
        return []
    strong = np.quantile(positive, 0.90)
    weak = magnitude > 0.40 * strong
    _, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    global_edges = np.isin(labels, np.unique(labels[magnitude >= strong])) & weak

    local_mean = cv2.boxFilter(magnitude, -1, (31, 31))
    normalized = np.clip(128 * magnitude / np.maximum(local_mean, max(strong * 0.1, 1e-6)), 0, 255)
    mean = cv2.boxFilter(normalized, -1, (31, 31))
    mean_sq = cv2.boxFilter(normalized**2, -1, (31, 31))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    local_edges = (normalized > mean * (1 + 0.34 * (std / 128 - 1))) & weak
    return [global_edges, global_edges | local_edges]


def radial_center(points, direction, strength, shape, box_center, box_size):
    """Accumulate radial symmetry responses with unsigned Di Zenzo normals."""
    normals = np.column_stack((np.cos(direction), np.sin(direction)))
    votes = np.zeros(shape, dtype=np.float32)
    strength = strength / max(float(strength.max()), 1e-6)
    start = max(4, int(0.2 * min(box_size)))
    stop = int(0.65 * max(box_size))
    for radius in range(start, stop + 1, 3):
        orientation_votes = np.zeros_like(votes)
        magnitude_votes = np.zeros_like(votes)
        for sign in (-1, 1):
            x, y = np.rint(points + sign * radius * normals).astype(int).T
            valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
            locations = (y[valid], x[valid])
            np.add.at(orientation_votes, locations, 1)
            np.add.at(magnitude_votes, locations, strength[valid])
        symmetry = (magnitude_votes / 9) * (np.minimum(orientation_votes, 9) / 9) ** 2
        votes += cv2.GaussianBlur(symmetry, (0, 0), max(1, radius * 0.04))
    allowed = np.zeros(shape, dtype=np.uint8)
    center_px = tuple(np.rint(box_center).astype(int))
    cv2.circle(allowed, center_px, max(3, int(min(box_size) * 0.25)), 1, -1)
    bounded = votes * allowed
    if not np.any(bounded):
        return box_center
    y, x = np.unravel_index(np.argmax(bounded), shape)
    return np.array([x, y], dtype=float)


def endpoint_tangent(point, labels, component):
    """Fit an outward principal direction within the stated 5 x 5 window."""
    x, y = point
    left, top = max(0, x - 2), max(0, y - 2)
    iy, ix = np.nonzero(labels[top:y + 3, left:x + 3] == component)
    neighbors = np.column_stack((ix + left, iy + top))
    if len(neighbors) < 2:
        return None
    centered = neighbors - neighbors.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    tangent = vectors[:, -1]
    return tangent if np.dot(tangent, point - neighbors.mean(axis=0)) >= 0 else -tangent


def direct_fit(points, weights):
    """Solve the weighted algebraic fit with the ellipse constraint 4ac-b²=1."""
    x, y = points.T
    q = np.column_stack((x * x, x * y, y * y))
    l = np.column_stack((x, y, np.ones_like(x)))
    constraint = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    w = weights[:, None]
    try:
        linear = np.linalg.solve(l.T @ (w * l), -l.T @ (w * q))
        reduced = q.T @ (w * (q + l @ linear))
        _, vectors = np.linalg.eig(np.linalg.solve(constraint, reduced))
        for a in vectors.T:
            if np.max(np.abs(a.imag)) > 1e-8:
                continue
            a = a.real
            if 4 * a[0] * a[2] - a[1] ** 2 <= 0:
                continue
            b = linear @ a
            matrix = np.array([[a[0], a[1] / 2], [a[1] / 2, a[2]]])
            center = np.linalg.solve(matrix, -b[:2] / 2)
            eigenvalues, axes = np.linalg.eigh(matrix)
            if not np.all(np.abs(eigenvalues) > 1e-12):
                continue
            lengths_sq = (center @ matrix @ center - b[2]) / eigenvalues
            if not np.all(lengths_sq > 0):
                continue
            angle = np.degrees(np.arctan2(axes[1, 0], axes[0, 0]))
            return major_first((center, 2 * np.sqrt(lengths_sq), angle))
    except np.linalg.LinAlgError:
        pass
    return None


def ellipse_coordinates(points, ellipse):
    center, diameters, angle = ellipse
    a = np.deg2rad(angle)
    x, y = ((points - center) @ [[np.cos(a), -np.sin(a)],
                                 [np.sin(a), np.cos(a)]] / (np.asarray(diameters) / 2)).T
    return np.hypot(x, y) - 1, np.mod(np.arctan2(y, x), 2 * np.pi)


def angular_stats(angles):
    if not len(angles):
        return 0, 360
    angles = np.sort(angles)
    bins = np.floor(angles * (36 / (2 * np.pi))).astype(int).clip(0, 35)
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return len(np.unique(bins)) / 36, float(np.rad2deg(gaps.max()))


def angular_span(angles):
    """Use the circular interval remaining after cutting at the largest gap."""
    ordered = np.sort(np.rad2deg(angles))
    gaps = np.diff(np.r_[ordered, ordered[0] + 360])
    cut = int(gaps.argmax())
    start, stop = ordered[(cut + 1) % len(ordered)], ordered[cut]
    return [(start, stop)] if start <= stop else [(start, 360), (0, stop)]


def angular_overlap(angles_a, angles_b):
    return sum(max(0, min(end_a, end_b) - max(start_a, start_b))
               for start_a, end_a in angular_span(angles_a)
               for start_b, end_b in angular_span(angles_b))


def improves_arc_geometry(angles_a, angles_b):
    coverage, gap = angular_stats(np.concatenate((angles_a, angles_b)))
    before = [angular_stats(angles) for angles in (angles_a, angles_b)]
    if coverage <= max(values[0] for values in before) or gap >= min(values[1] for values in before):
        return False
    # Compare normalized coverage × supported angular span on the common ellipse.
    previous = max(cover * (1 - missing / 360) for cover, missing in before)
    current = coverage * (1 - gap / 360)
    improvement = current - previous
    return (improvement >= MIN_ABSOLUTE_IMPROVEMENT
            and improvement >= MIN_RELATIVE_IMPROVEMENT * previous)


def robust_fit(points, iterations=8):
    shift = points.mean(0)
    scale = np.sqrt(np.mean(np.sum((points - shift) ** 2, axis=1)))
    if not np.isfinite(scale) or scale < 1e-8:
        return None
    points = (points - shift) / scale
    weights = np.ones(len(points))
    ellipse = direct_fit(points, weights)
    if ellipse is None:
        return None
    for _ in range(iterations):
        residual, angles = ellipse_coordinates(points, ellipse)
        scale_mad = max(1e-4, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        u = np.clip(residual / (4.685 * scale_mad), -1, 1)
        weights = (1 - u * u) ** 2
        bins = np.floor(angles * (36 / (2 * np.pi))).astype(int).clip(0, 35)
        for number in np.unique(bins):
            group = np.flatnonzero(bins == number)
            order = np.argsort(np.abs(residual[group]), kind="stable")
            weights[group[order[12:]]] = 0
        if np.count_nonzero(weights) < 6:
            break
        updated = direct_fit(points, weights)
        if updated is None:
            break
        new_residual, _ = ellipse_coordinates(points, updated)
        ellipse = updated
        if np.max(np.abs(residual - new_residual)) <= 1e-5:
            break
    center, diameters, angle = ellipse
    return shift + center * scale, np.asarray(diameters) * scale, angle


def grow_arcs(skeleton, radial_seed):
    mask = skeleton.astype(np.uint8)
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    points = np.column_stack(np.nonzero(mask)[::-1])
    ids = labels[points[:, 1], points[:, 0]]
    by_arc = {i: points[ids == i] for i in np.unique(ids)}
    degree = cv2.filter2D(mask, -1, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT) - mask
    endpoints = np.column_stack(np.nonzero((mask != 0) & (degree == 1))[::-1])
    if len(endpoints) < 2:
        return by_arc.values()
    arc_ids = labels[endpoints[:, 1], endpoints[:, 0]]
    tangents = [endpoint_tangent(p, labels, i) for p, i in zip(endpoints, arc_ids)]
    radii = np.linalg.norm(endpoints - radial_seed, axis=1)
    pairs = cKDTree(endpoints).query_pairs(3.0)
    pairs = sorted(pairs, key=lambda ij: (np.linalg.norm(endpoints[ij[1]] - endpoints[ij[0]]), ij))
    parents = {i: i for i in by_arc}
    used = set()

    def root(i):
        while i != parents[i]:
            i = parents[i]
        return i

    for i, j in pairs:
        p, q = root(arc_ids[i]), root(arc_ids[j])
        if p == q or i in used or j in used or tangents[i] is None or tangents[j] is None:
            continue
        if abs(radii[i] - radii[j]) > RADIUS_DISCREPANCY_TOLERANCE * (radii[i] + radii[j]) / 2:
            continue
        link = endpoints[j] - endpoints[i]
        link = link / np.linalg.norm(link)
        if min(link @ tangents[i], -link @ tangents[j]) < np.cos(np.deg2rad(15)):
            continue
        if tangents[i] @ -tangents[j] < np.cos(np.deg2rad(TANGENT_AGREEMENT_DEG)):
            continue
        turning = np.sin(np.deg2rad(TURN_TOLERANCE_DEG))
        a, b = tangents[i], -tangents[j]
        left_turn = a[0] * link[1] - a[1] * link[0]
        right_turn = link[0] * b[1] - link[1] * b[0]
        if left_turn * right_turn < 0 and min(abs(left_turn), abs(right_turn)) > turning:
            continue
        joined = np.concatenate((by_arc[p], by_arc[q]))
        if len(joined) < 8:
            continue
        trial = robust_fit(joined, iterations=0)
        if trial is None:
            continue
        angles_p = ellipse_coordinates(by_arc[p], trial)[1]
        angles_q = ellipse_coordinates(by_arc[q], trial)[1]
        if improves_arc_geometry(angles_p, angles_q):
            by_arc[p] = joined
            del by_arc[q]
            parents[q] = p
            used.update((i, j))
    return by_arc.values()


def merge_overlapping_arcs(arcs):
    """Join compatible overlapping arcs separately from short endpoint bridging."""
    pending = [arc for arc in arcs if len(arc) >= 8]
    merged = []
    while pending:
        arc = pending.pop(0)
        index = 0
        while index < len(pending):
            other = pending[index]
            index += 1
            # On overlapping angular support, the pixel bounds should meet.
            if (np.any(arc.min(0) > other.max(0) + 3)
                    or np.any(other.min(0) > arc.max(0) + 3)):
                continue
            joined = np.concatenate((arc, other))
            ellipse = robust_fit(joined, iterations=0)
            if ellipse is None:
                continue
            residual_a, angles_a = ellipse_coordinates(arc, ellipse)
            residual_b, angles_b = ellipse_coordinates(other, ellipse)
            if max(np.median(np.abs(residual_a)), np.median(np.abs(residual_b))) > 0.02:
                continue
            if angular_overlap(angles_a, angles_b) <= MIN_ARC_OVERLAP_DEG:
                continue
            if improves_arc_geometry(angles_a, angles_b):
                arc = joined
                pending.pop(index - 1)
                index = 0
        merged.append(arc)
    return merged


def dedup_paper_candidates(candidates):
    """Suppress duplicate outlines using directional support of each fitted ellipse."""
    directions = np.arange(360) * np.pi / 180
    normals = np.column_stack((np.cos(directions), np.sin(directions)))
    ranked = sorted(candidates, key=lambda candidate: compute_fit_score({
        "A": candidate.major_px / 2, "support": candidate.support,
        "ang_cover": candidate.coverage * 360, "max_gap": candidate.max_gap_deg,
    }), reverse=True)
    retained = []
    outlines = []
    for candidate in ranked:
        angle = np.deg2rad(candidate.angle_deg)
        major_projection = np.cos(directions - angle) * candidate.major_px / 2
        minor_projection = np.sin(directions - angle) * candidate.minor_px / 2
        outline = normals @ [candidate.cx, candidate.cy] + np.hypot(
            major_projection, minor_projection)
        if any(np.max(np.abs(outline - kept)) <= DUPLICATE_BOUNDARY_TOLERANCE_PX
               for kept in outlines):
            continue
        retained.append(candidate)
        outlines.append(outline)
    return retained


def concentric_cluster_selection(candidates):
    """Group co-oriented, nested rims and select the outer rim of the best group."""
    remaining = sorted(dedup_paper_candidates(candidates), key=lambda e: e.major_px, reverse=True)
    best_rims, best_score = (None, None), -1.0
    while remaining:
        outer = remaining[0]
        members = [outer]
        unmatched = []
        for candidate in remaining[1:]:
            center_distance = np.hypot(candidate.cx - outer.cx, candidate.cy - outer.cy)
            angle_distance = abs((candidate.angle_deg - outer.angle_deg + 90) % 180 - 90)
            nested = candidate.major_px < outer.major_px and candidate.minor_px < outer.minor_px
            if (center_distance <= STRICT["center_eps"] and angle_distance <= ANGLE_TOL_DEG
                    and nested):
                members.append(candidate)
            else:
                unmatched.append(candidate)
        # Match the archive's cluster score: sum of semi-major axes × adjusted mean support.
        total_semimajor = sum(member.major_px / 2 for member in members)
        mean_support = np.mean([member.support for member in members])
        score = total_semimajor * (0.5 + 0.5 * mean_support)
        if score > best_score:
            best_rims = (members[-1] if len(members) > 1 else None, outer)
            best_score = score
        remaining = unmatched
    return best_rims


def detect(frame, box, pad=20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_size = np.array([x2 - x1, y2 - y1])
    if min(box_size) < 20:
        return None
    left, top = max(0, int(np.floor(x1)) - pad), max(0, int(np.floor(y1)) - pad)
    right, bottom = min(w, int(np.ceil(x2)) + pad), min(h, int(np.ceil(y2)) + pad)
    crop = frame[top:bottom, left:right]
    if min(crop.shape[:2]) < 20:
        return None
    box_center = np.array([(x1 + x2) / 2 - left, (y1 + y2) / 2 - top])
    magnitude, orientation = color_gradient(crop)
    candidates = []
    for edges in edge_maps(magnitude):
        skeleton = thinning(np.uint8(edges) * 255) > 0
        y, x = np.nonzero(skeleton)
        if len(x) < 24:
            continue
        radial_seed = radial_center(np.column_stack((x, y)), orientation[y, x],
                                    magnitude[y, x], edges.shape, box_center, box_size)
        arcs = merge_overlapping_arcs(grow_arcs(skeleton, radial_seed))
        for arc in arcs:
            if len(arc) < max(24, 0.35 * max(box_size)):
                continue
            ellipse = robust_fit(arc)
            if ellipse is None:
                continue
            center, (major, minor), angle = ellipse
            offset = np.linalg.norm((center - box_center) / box_size)
            if not (0.45 <= major / max(box_size) <= 1.15
                    and minor / major >= 0.28 and offset <= 0.25):
                continue
            residual, angles = ellipse_coordinates(arc, ellipse)
            inliers = np.abs(residual) <= max(0.025, 5 / minor)
            coverage, gap = angular_stats(angles[inliers])
            if coverage < 0.80 or gap > 60:
                continue
            error = float(np.mean(np.abs(residual[inliers])))
            if error > 0.02:
                continue
            candidates.append(Ellipse(float(center[0] + left), float(center[1] + top),
                                      float(major), float(minor), float(angle),
                                      float(inliers.mean()), float(coverage), gap, 4, error))
    _, outer = concentric_cluster_selection(candidates)
    return outer
