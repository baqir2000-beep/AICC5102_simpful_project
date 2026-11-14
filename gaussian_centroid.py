"""Centroid of the maximum of clipped, unit-height Gaussian membership functions."""

import math

from scipy.special import ndtr


def _breakpoints(terms, lower, upper, include_crossings):
    points = {lower, upper}

    def add(point):
        if lower < point < upper:
            points.add(point)

    for center, width, log_height in terms:
        # A Gaussian may meet its own clipping plateau or that of another term.
        for _, _, level in terms:
            if level <= log_height:
                offset = width * math.sqrt(-2.0 * level)
                add(center - offset)
                add(center + offset)

    if include_crossings:
        for i, (left_center, left_width, _) in enumerate(terms):
            for right_center, right_width, _ in terms[i + 1:]:
                # Equal Gaussian heights imply equal or opposite standardized distances.
                delta = right_center - left_center
                add(left_center + delta * (left_width / (left_width + right_width)))
                if left_width != right_width:
                    add(left_center + delta * (left_width / (left_width - right_width)))
    return sorted(points)


def gaussian_centroid(sets, lower, upper, *, include_crossings=True):
    """Return the centroid of (firing strength, center, sigma) triples.

    Positive firing strengths are clipped to 1. The analytic method assumes
    finite centers and bounds, positive widths, and maximum aggregation.
    Disabling Gaussian crossings is intended only for the ablation experiment.
    """
    terms = [(center, width, math.log(min(strength, 1.0)))
             for strength, center, width in sets if strength > 0]
    if not terms or upper <= lower:
        return 0.0

    # Shift the first moment to reduce cancellation for translated universes.
    origin = lower + (upper - lower) / 2.0
    points = _breakpoints(terms, lower, upper, include_crossings)
    areas, moments = [], []
    for left, right in zip(points, points[1:]):
        span = right - left
        midpoint = left + span / 2.0
        distances = [(midpoint - center) / width for center, width, _ in terms]
        log_values = [-0.5 * z * z for z in distances]
        winner = max(range(len(terms)), key=lambda i: min(log_values[i], terms[i][2]))
        center, width, log_height = terms[winner]
        if log_values[winner] >= log_height:
            area = math.exp(log_height) * span
            moment = (midpoint - origin) * area
        else:
            z_left = (left - center) / width
            z_right = (right - center) / width
            # Use survival probabilities on the right tail to avoid subtracting two 1s.
            if z_left >= 0:
                probability = ndtr(-z_left) - ndtr(-z_right)
            else:
                probability = ndtr(z_right) - ndtr(z_left)
            area = width * math.sqrt(2.0 * math.pi) * probability
            exponent_left = -0.5 * z_left * z_left
            exponent_right = -0.5 * z_right * z_right
            if exponent_left == exponent_right:
                tail_difference = 0.0
            elif exponent_left > exponent_right:
                tail_difference = -math.exp(exponent_left) * math.expm1(exponent_right - exponent_left)
            else:
                tail_difference = math.exp(exponent_right) * math.expm1(exponent_left - exponent_right)
            moment = (center - origin) * area + width * width * tail_difference
        areas.append(area)
        moments.append(moment)
    total_area = math.fsum(areas)
    return origin + math.fsum(moments) / total_area if total_area else 0.0
