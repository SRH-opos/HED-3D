import cv2


def major_first(ellipse):
    """Use the major diameter first and report the angle of its axis."""
    center, (first, second), angle = ellipse
    if first < second:
        first, second = second, first
        angle += 90.0
    return center, (first, second), angle % 180.0


def fit_major_first(points):
    return major_first(cv2.fitEllipse(points))
