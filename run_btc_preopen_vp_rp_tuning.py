"""Run or resume the single corrected BTC pre-open VP+RP tuning pipeline."""
from pathlib import Path

import run_btc_preopen_model_update as model_update


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/research/btc_preopen_vp_rp_tuning_20261016.json"


def run():
    model_update.configure_model_update(CONFIG_PATH)
    return model_update.run_model_update()


if __name__ == "__main__":
    run()
