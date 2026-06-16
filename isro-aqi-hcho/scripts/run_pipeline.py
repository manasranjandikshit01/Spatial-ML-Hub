#!/usr/bin/env python3
"""
run_pipeline.py
===============
Top-level CLI orchestrator for the ISRO AQI & HCHO pipeline.

Provides single-command access to every pipeline stage:

    python scripts/run_pipeline.py download_all  --start 2019-01-01 --end 2022-12-31
    python scripts/run_pipeline.py build_datasets --synthetic
    python scripts/run_pipeline.py train_baseline
    python scripts/run_pipeline.py train_deep --synthetic
    python scripts/run_pipeline.py export_for_dashboard
    python scripts/run_pipeline.py run_all --synthetic   # full demo pipeline

Run with --help on any subcommand for available options:

    python scripts/run_pipeline.py train_baseline --help

Must be run from the isro-aqi-hcho project root:

    cd isro-aqi-hcho
    python scripts/run_pipeline.py <command> [options]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("run_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], desc: str = "") -> int:
    """
    Run a subprocess command and stream its output.

    Returns
    -------
    int  Return code (0 = success).
    """
    label = desc or " ".join(cmd)
    logger.info("▶ %s", label)
    t0 = time.time()
    result = subprocess.run(cmd, check=False, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    rc = result.returncode
    if rc == 0:
        logger.info("  ✓ Done (%.1fs)", elapsed)
    else:
        logger.error("  ✗ Failed (rc=%d, %.1fs): %s", rc, elapsed, label)
    return rc


def _python(module_or_script: str, *args: str) -> list[str]:
    """Build a `python -m module ...` or `python script.py ...` command."""
    if module_or_script.endswith(".py"):
        return [sys.executable, module_or_script] + list(args)
    return [sys.executable, "-m", module_or_script] + list(args)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stage functions
# ──────────────────────────────────────────────────────────────────────────────

def cmd_download_all(args: argparse.Namespace) -> int:
    """Orchestrate all data downloads."""
    logger.info("=== Stage: Download All Data ===")
    start, end = args.start, args.end
    rc_total = 0

    steps = [
        (_python("src.data.download_cpcb",
                 "--start_date", start, "--end_date", end,
                 "--output_dir", "data/raw/cpcb"),
         "Download CPCB ground-station data"),

        (_python("src.data.download_tropomi",
                 "--start_date", start, "--end_date", end,
                 "--output_dir", "data/raw/tropomi"),
         "Download TROPOMI satellite columns"),

        (_python("src.data.download_insat_aod",
                 "--start_date", start, "--end_date", end,
                 "--output_dir", "data/raw/insat_aod"),
         "Download INSAT-3D AOD"),

        (_python("src.data.download_reanalysis",
                 "--start_date", start, "--end_date", end,
                 "--output_dir", "data/raw/reanalysis"),
         "Download ERA5 reanalysis"),

        (_python("src.data.download_firms_fire",
                 "--start_date", start, "--end_date", end,
                 "--output_dir", "data/raw/firms"),
         "Download FIRMS fire data"),
    ]

    if args.static:
        steps.append((
            _python("src.data.download_static_layers",
                    "--layers", "land_cover", "population"),
            "Download static layers (land cover, population)",
        ))

    for cmd, desc in steps:
        rc = _run(cmd, desc)
        if rc != 0 and not args.skip_errors:
            logger.error("Download step failed; stopping. Use --skip_errors to continue.")
            return rc
        rc_total += rc

    return 0 if rc_total == 0 else 1


def cmd_build_datasets(args: argparse.Namespace) -> int:
    """Build AQI and HCHO training datasets."""
    logger.info("=== Stage: Build Datasets ===")
    rc_total = 0

    synthetic_flag = ["--synthetic"] if args.synthetic else []

    # AQI dataset
    rc = _run(
        _python("src.data.build_dataset_aqi", *synthetic_flag),
        "Build AQI training dataset",
    )
    rc_total += rc
    if rc != 0 and not args.skip_errors:
        return rc

    # HCHO dataset
    rc = _run(
        _python("src.data.build_dataset_hcho", *synthetic_flag),
        "Build HCHO hotspot dataset",
    )
    rc_total += rc

    return 0 if rc_total == 0 else 1


def cmd_train_baseline(args: argparse.Namespace) -> int:
    """Train Random Forest and Gradient Boosting baseline models."""
    logger.info("=== Stage: Train Baseline Models ===")

    extra: list[str] = []
    if args.hparam_search:
        extra.append("--hparam_search")
    if args.igp_only:
        extra.append("--igp_only")

    rc = _run(
        _python("src.models.baseline_ml",
                "--input", args.input,
                "--output_dir", args.output_dir,
                "--train_end", args.train_end,
                "--test_start", args.test_start,
                *extra),
        "Train baseline models (RF + GBM)",
    )
    return rc


def cmd_train_deep(args: argparse.Namespace) -> int:
    """Train the CNN-LSTM / ConvLSTM deep model."""
    logger.info("=== Stage: Train Deep Model ===")

    extra: list[str] = []
    if args.synthetic:
        extra.append("--synthetic")
    if args.hparam_sweep:
        extra.append("--hparam_sweep")

    rc = _run(
        _python("src.models.train_aqi",
                "--config", args.config,
                "--output_dir", args.output_dir,
                *extra),
        "Train CNN-LSTM model",
    )
    return rc


def cmd_export_for_dashboard(args: argparse.Namespace) -> int:
    """Run feature engineering pipelines to prepare dashboard-ready files."""
    logger.info("=== Stage: Export for Dashboard ===")
    rc_total = 0

    # AQI features
    rc = _run(
        _python("src.features.make_features_aqi",
                "--input",  "data/processed/aqi_training_dataset.csv",
                "--output", "data/processed/aqi_features.csv"),
        "AQI feature engineering",
    )
    rc_total += rc

    # HCHO features
    rc = _run(
        _python("src.features.make_features_hcho",
                "--input",  "data/processed/hcho_fire_daily_grid.csv",
                "--output", "data/processed/hcho_hotspot_features.csv",
                "--config", "config/hcho_hotspot.yaml"),
        "HCHO hotspot feature engineering",
    )
    rc_total += rc

    if rc_total == 0:
        logger.info("Dashboard files ready. Launch with:")
        logger.info("  streamlit run src/webapp/app.py")

    return 0 if rc_total == 0 else 1


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run the complete pipeline end-to-end (demo or real data)."""
    logger.info("=== Full Pipeline ===")
    stages: list[tuple[str, argparse.Namespace]] = []

    if not args.synthetic:
        stages.append(("download_all", args))

    stages += [
        ("build_datasets", args),
        ("train_baseline", args),
        ("train_deep", args),
        ("export_for_dashboard", args),
    ]

    dispatch = {
        "download_all": cmd_download_all,
        "build_datasets": cmd_build_datasets,
        "train_baseline": cmd_train_baseline,
        "train_deep": cmd_train_deep,
        "export_for_dashboard": cmd_export_for_dashboard,
    }

    t0 = time.time()
    for stage_name, stage_args in stages:
        rc = dispatch[stage_name](stage_args)
        if rc != 0 and not args.skip_errors:
            logger.error("Pipeline aborted at stage '%s'.", stage_name)
            return rc

    logger.info("Full pipeline completed in %.1f min", (time.time() - t0) / 60)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "ISRO AQI & HCHO Pipeline Orchestrator\n\n"
            "Run from the isro-aqi-hcho directory:\n"
            "  cd isro-aqi-hcho && python scripts/run_pipeline.py <command>"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--log_file", default="logs/pipeline.log",
                        help="Path to log file (default: logs/pipeline.log)")
    parser.add_argument("--skip_errors", action="store_true",
                        help="Continue even if a stage fails (not recommended)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── download_all ─────────────────────────────────────────────────────────
    p_dl = sub.add_parser("download_all", help="Download all raw data sources")
    p_dl.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    p_dl.add_argument("--end", default="2022-12-31", help="End date (YYYY-MM-DD)")
    p_dl.add_argument("--static", action="store_true",
                      help="Also download static layers (land cover, population)")

    # ── build_datasets ───────────────────────────────────────────────────────
    p_bd = sub.add_parser("build_datasets", help="Build AQI and HCHO training datasets")
    p_bd.add_argument("--synthetic", action="store_true",
                      help="Generate synthetic data (no API keys required)")

    # ── train_baseline ───────────────────────────────────────────────────────
    p_tb = sub.add_parser("train_baseline", help="Train RF + GBM baseline models")
    p_tb.add_argument("--input", default="data/processed/aqi_training_dataset.csv")
    p_tb.add_argument("--output_dir", default="models/baseline")
    p_tb.add_argument("--train_end", default="2021-12-31")
    p_tb.add_argument("--test_start", default="2022-01-01")
    p_tb.add_argument("--hparam_search", action="store_true")
    p_tb.add_argument("--igp_only", action="store_true",
                      help="Restrict training to the Indo-Gangetic Plain")

    # ── train_deep ───────────────────────────────────────────────────────────
    p_td = sub.add_parser("train_deep", help="Train CNN-LSTM / ConvLSTM deep model")
    p_td.add_argument("--config", default="config/aqi_training.yaml")
    p_td.add_argument("--output_dir", default="models/cnn_lstm")
    p_td.add_argument("--synthetic", action="store_true")
    p_td.add_argument("--hparam_sweep", action="store_true")

    # ── export_for_dashboard ─────────────────────────────────────────────────
    sub.add_parser("export_for_dashboard",
                   help="Run feature engineering to prepare Streamlit-ready files")

    # ── run_all ──────────────────────────────────────────────────────────────
    p_all = sub.add_parser("run_all", help="Run the full pipeline end-to-end")
    p_all.add_argument("--start", default="2019-01-01")
    p_all.add_argument("--end", default="2022-12-31")
    p_all.add_argument("--synthetic", action="store_true",
                       help="Skip downloads; use synthetic data throughout")
    p_all.add_argument("--config", default="config/aqi_training.yaml")
    p_all.add_argument("--input", default="data/processed/aqi_training_dataset.csv")
    p_all.add_argument("--output_dir", default="models/baseline")
    p_all.add_argument("--train_end", default="2021-12-31")
    p_all.add_argument("--test_start", default="2022-01-01")
    p_all.add_argument("--hparam_search", action="store_true")
    p_all.add_argument("--igp_only", action="store_true")
    p_all.add_argument("--hparam_sweep", action="store_true")
    p_all.add_argument("--static", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)
    logger.info("ISRO AQI & HCHO Pipeline  |  command: %s", args.command)

    dispatch = {
        "download_all": cmd_download_all,
        "build_datasets": cmd_build_datasets,
        "train_baseline": cmd_train_baseline,
        "train_deep": cmd_train_deep,
        "export_for_dashboard": cmd_export_for_dashboard,
        "run_all": cmd_run_all,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    rc = fn(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
