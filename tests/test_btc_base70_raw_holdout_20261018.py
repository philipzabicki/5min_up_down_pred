import tempfile
import unittest
import hashlib
from datetime import date
from pathlib import Path
import zipfile

import run_btc_base70_raw_holdout_20261018 as experiment
import run_btc_preopen_taker_flow_20261018 as taker


class FrozenRawHoldoutTests(unittest.TestCase):
    def test_model_specs_preserve_frozen_feature_sets_and_iterations(self):
        base64_features = [f"base_{index}" for index in range(64)]
        base70_features = base64_features + [
            "taker_Z_1m", "taker_V_1m", "taker_Z_5m", "taker_V_5m", "taker_Z_15m", "taker_V_15m",
        ]
        params = experiment.CONFIG["models"]["BASE64"]["effective_lightgbm_params"]
        specs = experiment._model_specs(
            {"feature_order": base64_features, "params": params, "best_iteration": 25},
            {"feature_order": base70_features, "params": params, "best_iteration": 31},
        )
        self.assertEqual(specs["BASE64"]["feature_order"], base64_features)
        self.assertEqual(specs["BASE64"]["iterations"], 25)
        self.assertEqual(specs["BASE70"]["feature_order"], base70_features)
        self.assertEqual(specs["BASE70"]["iterations"], 31)

    def test_missing_daily_klines_are_reported_for_each_holdout_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            days_root = Path(temporary)
            for day in ("2026-10-02", "2026-10-04"):
                directory = days_root / day
                directory.mkdir()
                (directory / f"BTCUSDT-1m-{day}.csv").touch()

            missing = experiment._missing_holdout_flow_files(
                days_root,
                start=date(2026, 10, 2),
                end_inclusive=date(2026, 10, 6),
            )

        self.assertEqual(
            missing,
            [
                (days_root / "2026-10-03" / "BTCUSDT-1m-2026-10-03.csv").as_posix(),
                (days_root / "2026-10-05" / "BTCUSDT-1m-2026-10-05.csv").as_posix(),
                (days_root / "2026-10-06" / "BTCUSDT-1m-2026-10-06.csv").as_posix(),
            ],
        )

    def test_daily_archive_parser_checks_full_minute_grid_and_volume_tolerance(self):
        day = "2026-10-02"
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / f"BTCUSDT-1m-{day}.zip"
            day_start_ms = int(experiment.pd.Timestamp(day, tz="UTC").timestamp() * 1000)
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                with archive.open(f"BTCUSDT-1m-{day}.csv", "w") as binary:
                    binary.write(
                        b"open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                        b"taker_buy_volume,taker_buy_quote_volume,ignore\n"
                    )
                    for minute in range(1440):
                        opened = day_start_ms + minute * 60_000
                        quote_volume = 1.0
                        taker_quote_volume = 1.0 + 5e-13 if minute == 0 else 0.5
                        row = (
                            f"{opened},100,101,99,100,1,{opened + 59_999},"
                            f"{quote_volume},{1},0.5,{taker_quote_volume},0\n"
                        )
                        binary.write(row.encode("ascii"))
            record = {
                "day": day,
                "zip_path": str(archive_path),
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }

            bars, audit = taker._load_and_validate_daily_archives([record])

        self.assertEqual(audit["instrument_identity"], {
            "market_type": "USD-M Futures", "symbol": "BTCUSDT", "interval": "1m",
        })
        self.assertEqual(audit["expected_minute_grid_rows"], 1440)
        self.assertEqual(audit["observed_valid_close_rows"], 1440)
        self.assertEqual(audit["missing_or_invalid_bar_minutes"], 0)
        self.assertTrue(audit["day_audit"][0]["header_row_present"])
        self.assertEqual(audit["day_audit"][0]["taker_buy_above_quote_within_tolerance_rows"], 1)
        self.assertEqual(bars.iloc[0]["quote_volume"], 1.0)
        self.assertAlmostEqual(bars.iloc[0]["taker_buy_quote_volume"], 1.0 + 5e-13)
        self.assertFalse(audit["values_clipped_or_imputed"])

    def test_daily_archive_parser_keeps_invalid_volume_missing(self):
        day = "2026-10-03"
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / f"BTCUSDT-1m-{day}.zip"
            opened = int(experiment.pd.Timestamp(day, tz="UTC").timestamp() * 1000)
            row = f"{opened},100,101,99,100,1,{opened + 59_999},-1,1,0.5,0.5,0\n"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    f"BTCUSDT-1m-{day}.csv",
                    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                    "taker_buy_volume,taker_buy_quote_volume,ignore\n" + row,
                )
            record = {
                "day": day,
                "zip_path": str(archive_path),
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }

            bars, audit = taker._load_and_validate_daily_archives([record])

        self.assertEqual(audit["day_audit"][0]["invalid_quote_or_taker_volume_rows_beyond_tolerance"], 1)
        self.assertTrue(bars.iloc[0][["quote_volume", "taker_buy_quote_volume"]].isna().all())


if __name__ == "__main__":
    unittest.main()
