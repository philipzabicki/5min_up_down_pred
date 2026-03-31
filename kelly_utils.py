import numpy as np


def adjust_probability_for_kelly(
    prob_raw,
    *,
    min_clip,
):
    p = np.asarray(prob_raw, dtype=np.float64)
    return np.clip(p, float(min_clip), 1.0 - float(min_clip))
