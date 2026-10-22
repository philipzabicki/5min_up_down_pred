"""Fail-closed guard for the separately authorized BTC candidate holdout evaluation."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_candidate_holdout_gate_20261022.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not config.get("allow_protected_data", False):
        raise SystemExit(
            "BLOCKED: protected BTC data access is disabled. No protected files or data pipelines were opened. "
            "The one-time scorer is not implemented in this development-only change."
        )
    if not config.get("scorer_implemented", False):
        raise SystemExit(
            "BLOCKED: the access flag is enabled, but this gate has no protected-data scorer. "
            "A separately authorized evaluation must add and review the scorer before any protected read."
        )
    raise SystemExit(
        "BLOCKED: no protected-data scorer is shipped by this gate. Replacing this guard requires a separately "
        "authorized one-time evaluation implementation."
    )


if __name__ == "__main__":
    main()
