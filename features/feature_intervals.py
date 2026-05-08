FEATURE_INTERVAL_TO_RULE = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def _supported_intervals_text():
    return ", ".join(FEATURE_INTERVAL_TO_RULE.keys())


def resolve_feature_intervals(settings):
    raw_intervals = (settings.get("feature_intervals") or {}).get("enabled")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError(
            "settings['feature_intervals']['enabled'] must be a non-empty list. "
            f"Supported intervals: {_supported_intervals_text()}"
        )

    intervals = []
    seen = set()
    unsupported = []
    for raw_interval in raw_intervals:
        interval = str(raw_interval).strip()
        if not interval:
            unsupported.append(interval)
            continue
        if interval in seen:
            continue
        if interval not in FEATURE_INTERVAL_TO_RULE:
            unsupported.append(interval)
            continue
        intervals.append(interval)
        seen.add(interval)

    if unsupported:
        raise ValueError(
            "Unsupported feature engineering intervals: "
            f"{unsupported}. Supported intervals: {_supported_intervals_text()}"
        )
    if not intervals:
        raise ValueError(
            "settings['feature_intervals']['enabled'] cannot be empty after "
            "deduplication. "
            f"Supported intervals: {_supported_intervals_text()}"
        )
    return tuple(intervals)
