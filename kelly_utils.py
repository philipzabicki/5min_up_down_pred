import numpy as np


def adjust_probability_for_kelly(
    prob_raw,
    *,
    prob_shrink,
    min_clip,
):
    p = np.asarray(prob_raw, dtype=np.float64)
    p = 0.5 + float(prob_shrink) * (p - 0.5)
    return np.clip(p, float(min_clip), 1.0 - float(min_clip))
