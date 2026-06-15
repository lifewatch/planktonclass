"""
Metadata
--------
- Authors: Jonas Mortelmans <jonas.mortelmans@vliz.be>, Wout Decrop <wout.decrop@vliz.be>
- Created: 2025-10-03
- Updated: 2026-06-06
- Version: 2.2.0-option-a-tar-stream-remote
- Documentation: Mortelmans J., Decrop W., Heynderickx H., Cattrijsse A., Depaepe M., Van Walraeven L., Scott J., Van Oevelen D., Deneudt K., Muniz C. (2025, submitted). High-throughput image classification and morphometry though the Pi-10 imaging pipeline
- Source: https://github.com/lifewatch/planktonclass/tree/PI10
"""



# === LIBRARIES ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import argparse


def _env_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_float(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def _parse_startup_args():
    parser = argparse.ArgumentParser(description="PI10 GPU prediction worker")
    parser.add_argument(
        "--gpu",
        default=os.getenv("PI10_GPU", ""),
        help="Physical GPU id to expose to TensorFlow, for example 0 or 1.",
    )
    parser.add_argument(
        "--worker-id",
        default=os.getenv("PI10_WORKER_ID", ""),
        help="Worker name used for per-worker scratch/log paths.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=_env_int("PI10_SLEEP_SECONDS", 3600),
        help="Seconds to sleep between source directory scans.",
    )
    parser.add_argument(
        "--stale-lock-hours",
        type=float,
        default=_env_float("PI10_STALE_LOCK_HOURS", 72.0),
        help="Remove .lock files older than this many hours; set 0 to disable.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan/process the current queue once, then exit.",
    )
    parser.add_argument(
        "--disable-email",
        action="store_true",
        help="Disable this worker's daily email scheduler.",
    )
    return parser.parse_args()


STARTUP_ARGS = _parse_startup_args()
if STARTUP_ARGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(STARTUP_ARGS.gpu)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import shutil
from pathlib import Path
import tarfile
import pandas as pd
import subprocess
import time
import random
import json
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import measure, morphology
from tensorflow.keras.models import load_model
from planktonclass import paths as plk_paths, utils
from planktonclass.test_utils import predict
from planktonclass.data_utils import load_class_names
import datetime
import threading
import time
import csv
import re
import io
import uuid
import posixpath
from concurrent.futures import ThreadPoolExecutor
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

last_summary_date = None  # will track the last date email was sent
last_afternoon_summary_sent_day = None  # for 15:00 status update

# === CONFIG ===
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ENV_VAR = "PI10_PREDICT_CONFIG"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "predict_gpu_config.json"


def _load_predict_config():
    config_path = Path(os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)).expanduser()
    if not config_path.is_absolute():
        config_path = (SCRIPT_DIR / config_path).resolve()

    if not config_path.exists():
        example_path = SCRIPT_DIR / "predict_gpu_config.example.json"
        raise FileNotFoundError(
            f"Missing PI10 predict config file: {config_path}\n"
            f"Create a private copy from {example_path}, or set {CONFIG_ENV_VAR}."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file), config_path
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in PI10 predict config: {config_path}") from exc


def _config_section(config, section_name):
    section = config.get(section_name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{section_name}' must be an object.")
    return section


def _require_config_value(config_section, key):
    value = config_section.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {key}")
    return value


def _resolve_path(value, base_dir):
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _path_from_config(config_section, key, default, base_dir):
    value = config_section.get(key)
    if value in (None, ""):
        return Path(default)
    return _resolve_path(value, base_dir)


def _path_from_pi10_root(config_section, key, *default_parts):
    return _path_from_config(
        config_section,
        key,
        PI10_ROOT.joinpath(*default_parts),
        PI10_ROOT,
    )


def _executable_from_config(config_section, key, default, base_dir):
    value = config_section.get(key)
    if value in (None, ""):
        value = default

    value = os.path.expandvars(str(value)).strip()
    if not value:
        return shutil.which("exiftool") or "exiftool"

    # A bare command such as "exiftool" should be resolved from PATH.
    if not Path(value).is_absolute() and "/" not in value and "\\" not in value:
        return shutil.which(value) or value

    path = _resolve_path(value, base_dir)
    if path.is_dir():
        path = path / ("exiftool.exe" if os.name == "nt" else "exiftool")

    # Avoid trying to execute the Windows binary on Linux.
    if os.name != "nt" and path.suffix.lower() == ".exe":
        return shutil.which("exiftool") or "exiftool"

    if not path.exists() and path.name.startswith("exiftool"):
        return shutil.which("exiftool") or path

    return path


PREDICT_CONFIG, PREDICT_CONFIG_PATH = _load_predict_config()
PATH_CONFIG = _config_section(PREDICT_CONFIG, "paths")
MODEL_CONFIG = _config_section(PREDICT_CONFIG, "model")
MAIL_CONFIG = _config_section(PREDICT_CONFIG, "mail")

PI10_ROOT = _resolve_path(
    _require_config_value(PATH_CONFIG, "pi10_root"),
    PREDICT_CONFIG_PATH.parent,
)


def _sanitize_worker_id(value):
    value = str(value or "").strip()
    if not value:
        return ""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-._") or "worker"


def _default_worker_id():
    if STARTUP_ARGS.worker_id:
        return STARTUP_ARGS.worker_id
    if STARTUP_ARGS.gpu:
        return f"gpu{str(STARTUP_ARGS.gpu).replace(',', '-')}"
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return f"gpu{cuda_visible.replace(',', '-')}"
    return ""


WORKER_ID = _sanitize_worker_id(_default_worker_id())
WORKER_LABEL = WORKER_ID or "default"

# === LOGGING ===
log_dir = _path_from_pi10_root(PATH_CONFIG, "log_dir", "not_processed", "GPU_ENVIRONMENT", "logs")
log_dir.mkdir(parents=True, exist_ok=True)
mail_metrics_dir = _path_from_config(
    PATH_CONFIG,
    "mail_metrics_dir",
    log_dir / "mail_metrics",
    PI10_ROOT,
)
mail_metrics_dir.mkdir(parents=True, exist_ok=True)

now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_worker_suffix = f"_{WORKER_ID}" if WORKER_ID else ""
log_file_path = log_dir / f"processing_times_{now_time}{log_worker_suffix}.csv"
daily_tar_count_json = mail_metrics_dir / "daily_tar_count.json"
daily_mail_metrics_csv = mail_metrics_dir / "daily_mail_metrics_history.csv"
daily_tar_progress_png = mail_metrics_dir / "daily_tar_progress.png"

# === PATHS ===
preview_path = _path_from_pi10_root(PATH_CONFIG, "preview_path", "not_processed", "previews")

# ExifTool executable path.
exiftool_path = _executable_from_config(
    PATH_CONFIG,
    "exiftool_path",
    "exiftool",
    PI10_ROOT,
)

# === DIRECTORIES ===
source_dir = _path_from_pi10_root(PATH_CONFIG, "source_dir", "processed", "2025")
base_work_dir = _path_from_pi10_root(
    PATH_CONFIG,
    "work_dir",
    "not_processed",
    "GPU_ENVIRONMENT",
    "PI10_tempUntarred",
)
work_dir = base_work_dir / f"worker_{WORKER_ID}" if WORKER_ID else base_work_dir
gpu_env = _path_from_pi10_root(PATH_CONFIG, "gpu_env", "not_processed", "GPU_ENVIRONMENT")


quarantine_bubbles_dir = _path_from_config(
    PATH_CONFIG,
    "quarantine_bubbles_dir",
    source_dir / "quarantine-bubbles",
    PI10_ROOT,
)
quarantine_bubbles_dir.mkdir(parents=True, exist_ok=True)

quarantine_hitsmiss_dir = _path_from_config(
    PATH_CONFIG,
    "quarantine_hitsmiss_dir",
    source_dir / "quarantine-hitsmisses",
    PI10_ROOT,
)
quarantine_hitsmiss_dir.mkdir(parents=True, exist_ok=True)

quarantine_gray_edge_dir = _path_from_config(
    PATH_CONFIG,
    "quarantine_gray_edge_dir",
    source_dir / "quarantine-gray-edge",
    PI10_ROOT,
)
quarantine_gray_edge_dir.mkdir(parents=True, exist_ok=True)

quarantine_raisingfactor_dir = _path_from_config(
    PATH_CONFIG,
    "quarantine_raisingfactor_dir",
    source_dir / "quarantine-raisingfactor",
    PI10_ROOT,
)
quarantine_raisingfactor_dir.mkdir(parents=True, exist_ok=True)

quarantine_near_point_dir = _path_from_config(
    PATH_CONFIG,
    "quarantine_near_point_dir",
    source_dir / "quarantine-location-50m",
    PI10_ROOT,
)
quarantine_near_point_dir.mkdir(parents=True, exist_ok=True)

quarantine_nogps_dir = _path_from_config(
    PATH_CONFIG,
    "quarantine_nogps_dir",
    source_dir / "quarantine-nogps",
    PI10_ROOT,
)
quarantine_nogps_dir.mkdir(parents=True, exist_ok=True)

VALIDATION_CONFIG = _config_section(PREDICT_CONFIG, "validation")
GPS_QUARANTINE_CONFIG = _config_section(PREDICT_CONFIG, "gps_quarantine")
MAX_MISS_HIT_RATIO = float(VALIDATION_CONFIG.get("max_miss_hit_ratio", 50))
REQUIRED_HITSMISSES_ROWS = int(VALIDATION_CONFIG.get("required_hitsmisses_rows", 10))
GRAY_EDGE_SAMPLE_SIZE = int(VALIDATION_CONFIG.get("gray_edge_sample_size", 20))
GRAY_EDGE_FRACTION = float(VALIDATION_CONFIG.get("gray_edge_fraction", 0.01))
_gray_edge_mean_min = VALIDATION_CONFIG.get("gray_edge_mean_min", 150.0)
GRAY_EDGE_MEAN_MIN = (
    None if _gray_edge_mean_min in (None, "") else float(_gray_edge_mean_min)
)
GRAY_EDGE_SAMPLE_N_IMAGES = GRAY_EDGE_SAMPLE_SIZE
QUARANTINE_LAT = float(GPS_QUARANTINE_CONFIG.get("latitude", 51.235293843807796))
QUARANTINE_LON = float(GPS_QUARANTINE_CONFIG.get("longitude", 2.9310864728604327))
QUARANTINE_RADIUS_M = float(GPS_QUARANTINE_CONFIG.get("radius_m", 50))
PREVIEW_BUBBLE_THRESHOLD = float(VALIDATION_CONFIG.get("preview_bubble_threshold", 0.4))
DAILY_SUMMARY_HOUR = int(MAIL_CONFIG.get("daily_summary_hour", 11))


# === OPTION A / PERFORMANCE CONFIG ===
PERFORMANCE_CONFIG = _config_section(PREDICT_CONFIG, "performance")


def _env_key_from_config_key(key):
    return "PI10_" + str(key).upper()


def _config_bool(config_section, key, default=False):
    value = os.getenv(_env_key_from_config_key(key), config_section.get(key, default))
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _config_int(config_section, key, default):
    value = os.getenv(_env_key_from_config_key(key), config_section.get(key, default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _config_str(config_section, key, default):
    value = os.getenv(_env_key_from_config_key(key), config_section.get(key, default))
    if value in (None, ""):
        return str(default)
    return str(value)


# Native tar + async cleanup are the safest speedups. They are used only when
# a full extraction is still needed as a fallback.
FAST_NATIVE_TAR = _config_bool(PERFORMANCE_CONFIG, "fast_native_tar", True)
TRUSTED_TAR_INPUTS = _config_bool(PERFORMANCE_CONFIG, "trusted_tar_inputs", True)
ASYNC_CLEANUP = _config_bool(PERFORMANCE_CONFIG, "async_cleanup", True)
EXPORT_TAXON_PREVIEWS = _config_bool(PERFORMANCE_CONFIG, "export_taxon_previews", True)
COPY_TAR_TO_LOCAL = _config_str(PERFORMANCE_CONFIG, "copy_tar_to_local", "auto").strip().lower()
MORPHOLOGY_WORKERS = max(1, _config_int(PERFORMANCE_CONFIG, "morphology_workers", 4))
PREVIEW_SAMPLE_N = max(1, _config_int(PERFORMANCE_CONFIG, "preview_sample_n", 200))

# Option A switches. TAR_STREAM_PREDICT keeps using planktonclass.predict(),
# but patches planktonclass.data_utils.load_image to accept filemode="tar".
TAR_STREAM_PREDICT = _config_bool(PERFORMANCE_CONFIG, "tar_stream_predict", True)
TAR_STREAM_MORPHOLOGY = _config_bool(PERFORMANCE_CONFIG, "tar_stream_morphology", True)
TAR_STREAM_EXPORT_TAXON_PREVIEWS = _config_bool(PERFORMANCE_CONFIG, "tar_stream_export_taxon_previews", True)
TAR_STREAM_EXIF_PYTHON = _config_bool(PERFORMANCE_CONFIG, "tar_stream_exif_python", True)
TAR_STREAM_EXIF_FALLBACK_FULL_EXTRACT = _config_bool(PERFORMANCE_CONFIG, "tar_stream_exif_fallback_full_extract", True)

#=== MAILING ===
from dotenv import load_dotenv
dotenv_path = _path_from_config(PATH_CONFIG, "dotenv_path", gpu_env / ".env", PI10_ROOT)
print(f"Loading environment variables from: {dotenv_path}")
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"⚠️ Email .env not found at {dotenv_path}; email summaries disabled.")

smtp_port = os.getenv('SMTP_PORT')
email_recipients = os.getenv('EMAIL_RECIPIENTS', '')
EMAIL_SETTINGS = {
    'smtp_server': os.getenv('SMTP_SERVER'),
    'smtp_port': int(smtp_port) if smtp_port else None,
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),
    'recipients': [email.strip() for email in email_recipients.split(',') if email.strip()]
}
EMAIL_ENABLED = (not STARTUP_ARGS.disable_email) and all([
    EMAIL_SETTINGS['smtp_server'],
    EMAIL_SETTINGS['smtp_port'],
    EMAIL_SETTINGS['sender_email'],
    EMAIL_SETTINGS['sender_password'],
    EMAIL_SETTINGS['recipients'],
])
if STARTUP_ARGS.disable_email:
    print("Email summaries disabled for this worker by --disable-email.")
elif not EMAIL_ENABLED:
    print("⚠️ Email summaries disabled because SMTP settings are incomplete.")
daily_tar_reports = []  # Stores dicts with tar_name, quarantined, quarantine_reason, quarantine_path, status_log

import smtplib
from email.mime.text import MIMEText
#test

def email_scheduler():
    global last_afternoon_summary_sent_day
    if not EMAIL_ENABLED:
        return

    while True:
        now = datetime.datetime.now()
        current_date = now.date()

        # Send summary at exactly 15:00 once per day
        if (now.hour == DAILY_SUMMARY_HOUR and now.minute == 0
                and last_afternoon_summary_sent_day != current_date):
            send_daily_summary_email(now.strftime('%Y-%m-%d'), daily_tar_reports)
            last_afternoon_summary_sent_day = current_date

        time.sleep(60)  # check every minute

def collect_mail_metrics(summary_date):
    all_tar_files = list(source_dir.glob("*.tar"))
    tar_stems = {tar.stem for tar in all_tar_files}
    done_marker_stems = {p.stem for p in source_dir.glob("*.done")}
    fully_done_stems = {
        stem for stem in tar_stems
        if all((source_dir / f"{stem}{suffix}").exists() for suffix in REQUIRED_SUFFIXES)
    }
    done_stems_combined = done_marker_stems.union(fully_done_stems)

    total_tars = len(tar_stems)
    done_tars = len(done_stems_combined.intersection(tar_stems))
    remaining_tars = max(0, total_tars - done_tars)

    metrics = {
        "date": summary_date,
        "logged_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tars": total_tars,
        "done_tars": done_tars,
        "remaining_tars": remaining_tars,
        "tar_files": len(all_tar_files),
        "done_markers": len(done_marker_stems),
        "gpstag_files": len(list(source_dir.glob("*_gpstag.csv"))),
        "prediction_json_files": len(list(source_dir.glob("*_predictions_relative.json"))),
        "image_property_files": len(list(source_dir.glob("*_image_properties.csv"))),
        "topspecies_files": len(list(source_dir.glob("*_topspecies.csv"))),
        "hitsmisses_files": len(list(source_dir.glob("*_hitsmisses.txt"))),
        "background_files": len(list(source_dir.glob("*_Background.tif"))),
        "bio_metrics_files": len(list(source_dir.glob("*_bio-metrics.csv"))),
    }
    return metrics, tar_stems

def update_daily_mail_history(metrics):
    new_row = pd.DataFrame([metrics])

    if daily_mail_metrics_csv.exists():
        history_df = pd.read_csv(daily_mail_metrics_csv)
    else:
        history_df = pd.DataFrame()

    if not history_df.empty and "date" in history_df.columns:
        history_df = history_df[history_df["date"] != metrics["date"]]

    history_df = pd.concat([history_df, new_row], ignore_index=True)
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df = history_df.dropna(subset=["date"])
    history_df = history_df.sort_values("date")
    history_df["date"] = history_df["date"].dt.strftime("%Y-%m-%d")
    history_df.to_csv(daily_mail_metrics_csv, index=False)

    return history_df

def make_daily_tar_progress_plot(history_df):
    if not MATPLOTLIB_AVAILABLE or history_df is None or history_df.empty:
        return None

    plot_df = history_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"])
    if plot_df.empty:
        return None

    for col in ["total_tars", "done_tars", "remaining_tars"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(plot_df["date"], plot_df["total_tars"], marker="o", linewidth=2, label="Total TARs")
    ax.plot(plot_df["date"], plot_df["done_tars"], marker="o", linewidth=2, label="Done")
    ax.plot(plot_df["date"], plot_df["remaining_tars"], marker="o", linewidth=2, label="Remaining")

    latest = plot_df.iloc[-1]
    latest_date = latest["date"]
    for col, label in [("total_tars", "Total"), ("done_tars", "Done"), ("remaining_tars", "Remaining")]:
        value = latest[col]
        if pd.notna(value):
            ax.annotate(
                f"{label}: {int(value)}",
                xy=(latest_date, value),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )

    done = latest["done_tars"]
    total = latest["total_tars"]
    remaining = latest["remaining_tars"]
    pct_done = 100 * done / total if pd.notna(done) and pd.notna(total) and total > 0 else 0
    summary_text = (
        f"Latest: {latest_date.strftime('%Y-%m-%d')}\n"
        f"Total TARs: {int(total) if pd.notna(total) else 0}\n"
        f"Done: {int(done) if pd.notna(done) else 0} ({pct_done:.1f}%)\n"
        f"Remaining: {int(remaining) if pd.notna(remaining) else 0}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.15),
    )
    ax.set_title("PI10 TAR processing progress")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of TARs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(daily_tar_progress_png, dpi=150)
    plt.close(fig)

    return daily_tar_progress_png

def send_daily_summary_email(summary_date, report_data):
    global source_dir

    subject = f"[PI10] Daily Summary - {summary_date}"
    metrics, tar_stems = collect_mail_metrics(summary_date)
    history_df = update_daily_mail_history(metrics)
    plot_path = make_daily_tar_progress_plot(history_df)

    current_counts = {
        "tar": metrics["tar_files"],
        "done": metrics["done_tars"],
        "remaining": metrics["remaining_tars"],
        "gpstag": metrics["gpstag_files"],
        "predictions": metrics["prediction_json_files"],
        "image_props": metrics["image_property_files"],
        "topspecies": metrics["topspecies_files"],
        "hitsmisses": metrics["hitsmisses_files"],
        "backgrounds": metrics["background_files"],
        "bio_metrics": metrics["bio_metrics_files"],
    }

    yesterday_counts = {k: 0 for k in current_counts}
    if daily_tar_count_json.exists():
        try:
            with open(daily_tar_count_json, "r") as f:
                yesterday_counts.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Could not read previous daily count: {e}")

    deltas = {k: current_counts[k] - yesterday_counts.get(k, 0) for k in current_counts}

    try:
        with open(daily_tar_count_json, "w") as f:
            json.dump(current_counts, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write current daily count: {e}")

    required_outputs = {
        "_gpstag.csv": ("GPS data", 0.5),
        "_hitsmisses.txt": ("Hits/Misses", 10 / 3600),
        "_Background.tif": ("Background.tif", 10 / 3600),
        "_predictions_relative.json": ("Predictions (JSON)", 3),
        "_image_properties.csv": ("Image Properties (CSV)", 0.5),
        "_topspecies.csv": ("Top Species CSV", 2 / 60),
        "_bio-metrics.csv": ("Bio metrics", 2 / 60),
    }

    todo_counts = {}
    raw_time_estimations = {}
    formatted_time_estimations = {}

    for suffix, (label, per_file_hours) in required_outputs.items():
        count = sum(not (source_dir / f"{stem}{suffix}").exists() for stem in tar_stems)
        todo_counts[label] = count
        total_hours = count * per_file_hours
        raw_time_estimations[label] = total_hours

        if total_hours >= 24:
            formatted_time_estimations[label] = f"{round(total_hours / 24, 2)} day(s)"
        else:
            h = int(total_hours)
            m = round((total_hours - h) * 60)
            formatted_time_estimations[label] = f"{h}h {m}min"

    total_time = sum(raw_time_estimations.values())
    if total_time >= 24:
        total_time_str = f"{round(total_time / 24, 2)} day(s)"
    else:
        th = int(total_time)
        tm = round((total_time - th) * 60)
        total_time_str = f"{th}h {tm}min"

    body_lines = []
    body_lines.append(f"Summary for {summary_date}")
    body_lines.append("=" * 60)
    body_lines.append("")
    body_lines.append("TAR processing progress:")
    body_lines.append(f"- Total TARs in processing folder: {metrics['total_tars']}")
    body_lines.append(f"- Done TARs: {metrics['done_tars']}")
    body_lines.append(f"- Remaining TARs: {metrics['remaining_tars']}")
    if metrics["total_tars"] > 0:
        pct_done = 100 * metrics["done_tars"] / metrics["total_tars"]
        body_lines.append(f"- Completion: {pct_done:.1f}%")
    body_lines.append("")
    body_lines.append(f"TARs entirely processed since script start: {len(report_data)}")
    body_lines.append("")
    body_lines.append("Folder totals vs previous mail:")
    body_lines.append(f"- TAR files: {current_counts['tar']} ({deltas['tar']:+})")
    body_lines.append(f"- Done TARs: {current_counts['done']} ({deltas['done']:+})")
    body_lines.append(f"- Remaining TARs: {current_counts['remaining']} ({deltas['remaining']:+})")
    body_lines.append(f"- GPS data: {current_counts['gpstag']} ({deltas['gpstag']:+})")
    body_lines.append(f"- Predictions JSON: {current_counts['predictions']} ({deltas['predictions']:+})")
    body_lines.append(f"- Image properties CSV: {current_counts['image_props']} ({deltas['image_props']:+})")
    body_lines.append(f"- Top Species CSV: {current_counts['topspecies']} ({deltas['topspecies']:+})")
    body_lines.append(f"- Hits/Misses TXT: {current_counts['hitsmisses']} ({deltas['hitsmisses']:+})")
    body_lines.append(f"- Background.tif: {current_counts['backgrounds']} ({deltas['backgrounds']:+})")
    body_lines.append(f"- Bio metrics CSV: {current_counts['bio_metrics']} ({deltas['bio_metrics']:+})")
    body_lines.append("")
    body_lines.append("To-do by output module:")
    for label in todo_counts:
        count = todo_counts[label]
        formatted_time = formatted_time_estimations[label]
        body_lines.append(f"- {label}: {count} files missing ({formatted_time})")
    body_lines.append("")
    body_lines.append(f"Total estimated processing time left: {total_time_str}")
    body_lines.append("")
    body_lines.append(f"Mail metrics history: {daily_mail_metrics_csv}")
    if plot_path is not None:
        body_lines.append(f"Progress plot: {daily_tar_progress_png}")

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SETTINGS["sender_email"]
    msg["To"] = ", ".join(EMAIL_SETTINGS["recipients"])
    image_section = ""
    if plot_path is not None:
        image_section = (
            "<h3 style=\"font-family: Arial, sans-serif;\">PI10 TAR processing progress</h3>"
            "<img src=\"cid:daily_tar_progress\" style=\"max-width: 900px; width: 100%; height: auto;\">"
        )
    html = (
        "<html><body>"
        "<div style=\"font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4;\">"
        f"{'<br>'.join(body_lines)}"
        "</div>"
        f"{image_section}"
        "</body></html>"
    )
    msg.attach(MIMEText(html, "html"))

    if plot_path is not None and Path(plot_path).exists():
        try:
            with open(plot_path, "rb") as img_file:
                img = MIMEImage(img_file.read(), _subtype="png")
            img.add_header("Content-ID", "<daily_tar_progress>")
            img.add_header("Content-Disposition", "inline", filename=Path(plot_path).name)
            msg.attach(img)
        except Exception as e:
            print(f"⚠️ Could not embed progress plot inline: {e}")

    try:
        with smtplib.SMTP(EMAIL_SETTINGS["smtp_server"], EMAIL_SETTINGS["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_SETTINGS["sender_email"], EMAIL_SETTINGS["sender_password"])
            server.sendmail(msg["From"], EMAIL_SETTINGS["recipients"], msg.as_string())
        print(f"📧 Daily summary email sent for {summary_date}")
    except Exception as e:
        print(f"❌ Failed to send daily summary email: {e}")



#LOG TIME OF EACH STEP
def init_log_file():
    """Initialize the CSV log file with headers."""
    headers = [
        "TAR Name",
        "Copy TAR to working directory",
        "Tar inventory",
        "Gray edge quarantine check",
        "Extract hitsmisses.txt",
        "Count images",
        "Create preview images",
        "Early preview classification",
        "Copy Background.tif",
        "Untar",
        "Extract and save EXIF metadata",
        "Classification and morphology extraction",
        "Export images by taxon",
        "Generate top species CSV",
        "Per-minute bio metrics",
        "Async cleanup scheduled",
        "Total pipeline time (h)",
        "Number of images in TAR",
        "Model used",
        "Logged at"
    ]

    if not Path(log_file_path).exists():
        with open(log_file_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(headers)
        print("⚙ Initialized new processing time log file.")
    else:
        try:
            with open(log_file_path, "r", newline="") as log_file:
                rows = list(csv.reader(log_file))

            if rows and rows[0] != headers:
                old_headers = rows[0]
                rewritten_rows = [headers]

                for row in rows[1:]:
                    row_by_header = {
                        header: row[idx] if idx < len(row) else ""
                        for idx, header in enumerate(old_headers)
                    }
                    rewritten_rows.append([row_by_header.get(header, "") for header in headers])

                with open(log_file_path, "w", newline="") as log_file:
                    writer = csv.writer(log_file)
                    writer.writerows(rewritten_rows)

                print("⚙ Log file header updated for current pipeline steps.")
            else:
                print("⚙ Log file already exists, appending new entries.")

        except Exception as e:
            print(f"⚠️ Could not validate/update log file header: {e}")
            print("⚙ Log file already exists, appending new entries.")

# for the logfiles; these are the headers
step_names = [
    "Copy TAR to working directory",
    "Tar inventory",
    "Gray edge quarantine check",
    "Extract hitsmisses.txt",
    "Count images",
    "Create preview images",
    "Early preview classification",
    "Copy Background.tif",
    "Untar",
    "Extract and save EXIF metadata",
    "Classification and morphology extraction",
    "Export images by taxon",
    "Generate top species CSV",
    "Per-minute bio metrics",
    "Async cleanup scheduled"
]

def log_time_to_file(tar_name, times_dict, num_images):
    total_hours = sum(times_dict.values()) / 3600.0
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [tar_name] + [times_dict.get(name, 0.0) for name in step_names] \
          + [total_hours, num_images, TIMESTAMP, timestamp]

    with open(log_file_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(row)

    #print(f"✅ Logged times for {tar_name} to file (total {total_hours:.2f} h).")


def track_time(start_time, module_name):
    """Calculate elapsed time and return the time taken."""
    elapsed_time = time.time() - start_time
    return elapsed_time


def remove_partial_outputs(tar_name, status_log):
    for suffix in REQUIRED_SUFFIXES:
        partial_file = source_dir / f"{tar_name}{suffix}"
        try:
            if partial_file.exists():
                partial_file.unlink()
                status_log.append(f"Removed partial output: {partial_file.name}")
        except Exception as rm_err:
            status_log.append(f"⚠️ Failed to remove {partial_file.name}: {rm_err}")




# === SETUP ===
taxon_export_root = source_dir / "by_taxon"
taxon_export_root.mkdir(parents=True, exist_ok=True)
MAX_IMAGES_PER_TAXON_FOLDER = 100
PREVIEW_HIGH_GAP_MARGIN = float(VALIDATION_CONFIG.get("preview_high_gap_margin", 0.999))
PREVIEW_HIGH_GAP_LIMIT = int(VALIDATION_CONFIG.get("preview_high_gap_limit", 150))
PREVIEW_LOW_GAP_LIMIT = int(VALIDATION_CONFIG.get("preview_low_gap_limit", 250))
PREVIEW_HIGH_GAP_SUFFIX = "__highgap"
PREVIEW_LOW_GAP_SUFFIX = "__lowgap"
VALIDATION_OUTPUT_SUFFIX = "_validated_cleaned.csv"
VALIDATION_USER = "CNN"
IGNORED_TIF_NAMES = {"Background.tif", "FlowCellEdges.tif"}

REQUIRED_SUFFIXES = [
    "_gpstag.csv",
    "_hitsmisses.txt",
    "_Background.tif",
    "_predictions_relative.json",
    "_image_properties.csv",
    "_topspecies.csv",
    "_bio-metrics.csv",
]


def all_required_outputs_exist(tar_name):
    return all((source_dir / f"{tar_name}{suffix}").exists() for suffix in REQUIRED_SUFFIXES)


os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)
if EMAIL_ENABLED:
    email_thread = threading.Thread(target=email_scheduler, daemon=True)
    email_thread.start()
init_log_file()  # Initialize log file right after setup


paths = {
    'tarred': work_dir / "data/tarred",
    'untarred': work_dir / "data/untarred",
    'output': work_dir / "output",
    'hitsmisses': work_dir / "data/hitsmisses",
    'preview': work_dir / "data/preview"
}

for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)

# Classification model setup
import os


# =========================
# ROOT = remove PI10 folder mistake
# =========================
BASE = Path.cwd().resolve()
print(BASE)
# if you're inside PI10/, go one level up


from pathlib import Path

BASE = Path(__file__).resolve().parent
print(BASE)

if BASE.name == "PI10":
    BASE = BASE.parent

# now BASE should be: .../planktonclass
print("\n🔎 BASE FIXED:", BASE)

TIMESTAMP = MODEL_CONFIG.get("timestamp", "2025-10-09_140052-anasimyia")
# model_root = _path_from_config(MODEL_CONFIG, "root", BASE / "models", PREDICT_CONFIG_PATH.parent)

model_root= _path_from_pi10_root(PATH_CONFIG, "model_dir", "not_processed", "models")


model_path = model_root / TIMESTAMP
CLASS_TRANSLATION_CSV = _path_from_config(
    PATH_CONFIG,
    "class_translation_csv",
    PI10_ROOT / "not_processed" / "models" / TIMESTAMP / "class_name_translation.csv",
    PI10_ROOT,
)

print("📁 model_path:", model_path)

# =========================
# FORCE PACKAGE PATHS
# =========================
plk_paths.homedir = str(BASE)
plk_paths.timestamp = TIMESTAMP

plk_paths.get_ts_splits_dir = lambda: str(model_path / "dataset_files")
plk_paths.get_checkpoints_dir = lambda: str(model_path / "ckpts")
plk_paths.get_conf_dir = lambda: str(model_path / "conf")

# =========================
# DEBUG OUTPUT
# =========================
print("\n📁 RESOLVED PATHS")
print("splits:", plk_paths.get_ts_splits_dir())
print("ckpts :", plk_paths.get_checkpoints_dir())
print("conf  :", plk_paths.get_conf_dir())


MODEL_NAME = MODEL_CONFIG.get("model_name", "final_model.h5")
TOP_K = int(MODEL_CONFIG.get("top_k", 3))


class_names = load_class_names(splits_dir=plk_paths.get_ts_splits_dir())
model = load_model(os.path.join(plk_paths.get_checkpoints_dir(), MODEL_NAME),
                   custom_objects=utils.get_custom_objects())
with open(os.path.join(plk_paths.get_conf_dir(), 'conf.json')) as f:
    conf = json.load(f)


# === HELPER FUNCTIONS ===
def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_m * np.arcsin(np.sqrt(a))


def get_usable_coordinates(exif_df):
    """Return numeric GPS coordinates from EXIF, or an empty DataFrame if none are usable."""
    if exif_df is None or exif_df.empty:
        return pd.DataFrame(columns=["GPSLatitude", "GPSLongitude"])

    if "GPSLatitude" not in exif_df.columns or "GPSLongitude" not in exif_df.columns:
        return pd.DataFrame(columns=["GPSLatitude", "GPSLongitude"])

    coords = exif_df[["GPSLatitude", "GPSLongitude"]].copy()
    coords["GPSLatitude"] = pd.to_numeric(coords["GPSLatitude"], errors="coerce")
    coords["GPSLongitude"] = pd.to_numeric(coords["GPSLongitude"], errors="coerce")
    coords = coords.dropna(subset=["GPSLatitude", "GPSLongitude"])

    return coords[
        coords["GPSLatitude"].between(-90, 90)
        & coords["GPSLongitude"].between(-180, 180)
    ]


def has_usable_coordinates(exif_df):
    return not get_usable_coordinates(exif_df).empty


def should_quarantine_location(exif_df):
    coords = get_usable_coordinates(exif_df)
    if coords.empty:
        return False

    distances = haversine_m(
        coords["GPSLatitude"],
        coords["GPSLongitude"],
        QUARANTINE_LAT,
        QUARANTINE_LON,
    )
    return (distances <= QUARANTINE_RADIUS_M).any()


def outputs_exist_for_tar(tar_file):
    stem = tar_file.stem
    return all((source_dir / f"{stem}{suffix}").exists() for suffix in REQUIRED_SUFFIXES)


def tar_date_prefix_value(tar_path):
    """Return the leading 8-digit date prefix as an int, or -1 if absent."""
    prefix = tar_path.stem[:8]
    return int(prefix) if prefix.isdigit() else -1


def to_gray(img):
    img = np.asarray(img)

    if img.ndim == 3:
        base = img[..., :3].astype(np.float32)
        gray = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    else:
        gray = img.astype(np.float32)

    if np.issubdtype(img.dtype, np.integer):
        gray = gray / np.iinfo(img.dtype).max * 255.0

    return gray


def edge_mean_gray(path, edge_fraction=GRAY_EDGE_FRACTION):
    img = tiff.imread(path)
    gray = to_gray(img)

    h, w = gray.shape
    bw = max(1, int(min(h, w) * edge_fraction))

    mask = np.zeros_like(gray, dtype=bool)
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True

    return float(gray[mask].mean())


def check_gray_edge_quarantine(
    extract_dir,
    tar_name,
    n_images=GRAY_EDGE_SAMPLE_SIZE,
    min_edge_mean=GRAY_EDGE_MEAN_MIN,
):
    print("Running gray-edge quarantine check...")

    tif_files = [
        p for p in Path(extract_dir).rglob("*.tif")
        if p.name not in IGNORED_TIF_NAMES
    ]

    if not tif_files:
        print("       No valid TIFFs found for gray-edge check; continuing.")
        return True, None, None

    sample = random.sample(tif_files, min(n_images, len(tif_files)))
    values = []

    for tif_path in sample:
        try:
            values.append(edge_mean_gray(tif_path))
        except Exception as e:
            print(f"       Failed gray-edge read for {tif_path.name}: {e}")

    if not values:
        print("       No readable TIFFs for gray-edge check; continuing.")
        return True, None, None

    edge_mean = sum(values) / len(values)
    print(
        f"       Edge mean grayscale: {edge_mean:.2f} "
        f"from {len(values)} image(s); threshold: {min_edge_mean}"
    )

    log_path = source_dir / "edge_mean_grayscale_quarantine_checks.csv"
    write_header = not log_path.exists()
    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "logged_at",
                    "tar_name",
                    "n_images_used",
                    "edge_mean_grayscale",
                    "threshold_min",
                    "result",
                ])

            result = (
                "quarantine"
                if min_edge_mean is not None and edge_mean < min_edge_mean
                else "ok"
            )
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tar_name,
                len(values),
                round(edge_mean, 2),
                min_edge_mean,
                result,
            ])
    except Exception as e:
        print(f"       Could not write gray-edge log: {e}")

    if min_edge_mean is not None and edge_mean < min_edge_mean:
        reason = f"edge mean grayscale {edge_mean:.2f} < {min_edge_mean}"
        return False, reason, edge_mean

    return True, None, edge_mean


def get_new_tar_files(source_dir):
    all_tar = list(source_dir.glob("*.tar"))

    # combine both quarantine folders
    quarantine_stems = set()
    quarantine_stems.update({tar.stem for tar in quarantine_bubbles_dir.glob("*.tar")})
    quarantine_stems.update({tar.stem for tar in quarantine_hitsmiss_dir.glob("*.tar")})
    quarantine_stems.update({tar.stem for tar in quarantine_gray_edge_dir.glob("*.tar")})
    quarantine_stems.update({tar.stem for tar in quarantine_raisingfactor_dir.glob("*.tar")})
    quarantine_stems.update({tar.stem for tar in quarantine_near_point_dir.glob("*.tar")})
    quarantine_stems.update({tar.stem for tar in quarantine_nogps_dir.glob("*.tar")})

    done_stems = {p.stem for p in source_dir.glob("*.done")}

    new_files = []
    for tar in all_tar:
        # skip if in quarantine
        if tar.stem in quarantine_stems:
            continue
        # skip if already marked done
        if tar.stem in done_stems:
            continue

        outputs_to_check = REQUIRED_SUFFIXES
        missing_output = False

        for suffix in outputs_to_check:
            expected = source_dir / f"{tar.stem}{suffix}"
            if not expected.exists():
                missing_output = True
                break

        if missing_output:
            new_files.append(tar)

    new_files.sort(key=lambda tar: (-tar_date_prefix_value(tar), tar.name.lower()))
    return new_files


def load_label_translation(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"⚠️ Translation CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    required_cols = {"original_label", "translated_label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Translation CSV is missing required column(s): {missing_cols}"
        )

    df = df[["original_label", "translated_label"]].dropna()

    return {
        str(row["original_label"]).strip(): str(row["translated_label"]).strip()
        for _, row in df.iterrows()
    }


def translate_label(label, translation_dict):
    label = str(label).strip()
    translated = translation_dict.get(label)

    if translated is None:
        print(f"⚠️ No translation found for label '{label}', using original label")
        translated = label

    return translated


def sanitize_taxon_name(name):
    name = str(name).strip() if name is not None else "unclassified"
    if not name:
        name = "unclassified"
    name = re.sub(r'-\d+$', '', name)
    name = re.sub(r'[<>:"/\|?*]+', '_', name)
    name = name.rstrip(' .')
    return name or "unclassified"


def unique_flattened_destination(dest_dir, src_name):
    src_name = Path(src_name).name
    candidate = dest_dir / src_name
    if not candidate.exists():
        return candidate

    stem = Path(src_name).stem
    suffix = Path(src_name).suffix
    counter = 2
    while True:
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def parse_prediction_lists(entry):
    labels = (
        entry.get(f"top{TOP_K}_labels", [])
        or entry.get("top3_labels", [])
        or entry.get("top2_labels", [])
    )
    probs = (
        entry.get(f"top{TOP_K}_probs", [])
        or entry.get("top3_probs", [])
        or entry.get("top2_probs", [])
    )

    if isinstance(labels, str):
        labels = [s.strip() for s in labels.split(",") if s.strip()]
    else:
        labels = list(labels or [])

    if isinstance(probs, str):
        raw_probs = [s.strip() for s in probs.split(",") if s.strip()]
    else:
        raw_probs = list(probs or [])

    clean_probs = []
    for value in raw_probs:
        try:
            clean_probs.append(float(value))
        except (TypeError, ValueError):
            clean_probs.append(None)

    while len(clean_probs) < len(labels):
        clean_probs.append(None)

    return labels, clean_probs


def classify_preview_bucket(top1_prob, top2_prob, margin_threshold=PREVIEW_HIGH_GAP_MARGIN):
    top1_prob = float(top1_prob) if top1_prob is not None else None
    top2_prob = float(top2_prob) if top2_prob is not None else 0.0
    margin = (top1_prob - top2_prob) if top1_prob is not None else None

    if margin is not None and margin >= margin_threshold:
        return "highgap", PREVIEW_HIGH_GAP_SUFFIX, PREVIEW_HIGH_GAP_LIMIT, margin

    return "lowgap", PREVIEW_LOW_GAP_SUFFIX, PREVIEW_LOW_GAP_LIMIT, margin



def export_images_by_top1_taxon(
    extract_dir,
    json_path,
    tar_name,
    export_root,
    high_gap_margin=PREVIEW_HIGH_GAP_MARGIN,
    high_gap_limit=PREVIEW_HIGH_GAP_LIMIT,
    low_gap_limit=PREVIEW_LOW_GAP_LIMIT,
):
    print("⚙ Exporting classified images by taxon (high-gap / low-gap preview buckets)...")

    json_path = Path(json_path)
    extract_dir = Path(extract_dir)
    export_root = Path(export_root)

    if not json_path.exists():
        print(f"       ⚠️ JSON file not found: {json_path}")
        return {
            "copied": 0,
            "missing": 0,
            "taxa": 0,
            "highgap_copied": 0,
            "lowgap_copied": 0,
            "skipped_limit": 0,
            "validation_csv": None,
        }

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON for taxon export: {e}")
        return {
            "copied": 0,
            "missing": 0,
            "taxa": 0,
            "highgap_copied": 0,
            "lowgap_copied": 0,
            "skipped_limit": 0,
            "validation_csv": None,
        }

    data = [entry for entry in data if isinstance(entry, dict)]
    label_translation = load_label_translation(CLASS_TRANSLATION_CSV)

    tar_export_dir = export_root / tar_name
    if tar_export_dir.exists():
        shutil.rmtree(tar_export_dir)
    tar_export_dir.mkdir(parents=True, exist_ok=True)

    prepared_entries = []
    missing = 0

    for entry in data:
        rel_path = entry.get("filepath", "")
        labels, probs = parse_prediction_lists(entry)

        if not labels:
            continue

        original_top1 = str(labels[0]).strip()
        translated_top1 = translate_label(original_top1, label_translation)

        top1 = sanitize_taxon_name(translated_top1)
        top2 = sanitize_taxon_name(labels[1]) if len(labels) > 1 else None
        top1_prob = probs[0] if len(probs) > 0 else None
        top2_prob = probs[1] if len(probs) > 1 else 0.0

        bucket, folder_suffix, default_limit, margin = classify_preview_bucket(
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            margin_threshold=high_gap_margin,
        )

        bucket_limit = high_gap_limit if bucket == "highgap" else low_gap_limit
        src = extract_dir / rel_path

        if not src.exists():
            missing += 1
            continue

        prepared_entries.append({
            "src": src,
            "top1": top1,
            "original_top1": original_top1,
            "translated_top1": translated_top1,
            "top2": top2,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "margin": margin,
            "bucket": bucket,
            "folder_suffix": folder_suffix,
            "bucket_limit": bucket_limit,
        })

    grouped = {}
    for item in prepared_entries:
        grouped.setdefault((item["top1"], item["bucket"]), []).append(item)

    selected = []
    skipped_limit = 0

    for (taxon, bucket), entries in grouped.items():
        rng = random.Random(f"{tar_name}|{taxon}|{bucket}")
        rng.shuffle(entries)
        limit = entries[0]["bucket_limit"] if entries else 0

        selected.extend(entries[:limit])
        skipped_limit += max(0, len(entries) - limit)

    copied = 0
    highgap_copied = 0
    lowgap_copied = 0
    taxa_counts = {}
    validation_rows = []

    for item in selected:
        folder_name = f'{item["top1"]}{item["folder_suffix"]}'
        dest_dir = tar_export_dir / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_flattened_destination(dest_dir, item["src"].name)
        shutil.copy2(item["src"], dest)

        copied += 1
        taxa_counts[item["top1"]] = taxa_counts.get(item["top1"], 0) + 1

        if item["bucket"] == "highgap":
            highgap_copied += 1
        else:
            lowgap_copied += 1

        validation_rows.append({
            "image": dest.name,
            "label": item["top1"],
            "original_label": item["original_top1"],
            "translated_label": item["translated_top1"],
            "user": VALIDATION_USER,
            "subset": item["bucket"],
            "folder": folder_name,
            "top1_prob": item["top1_prob"],
            "top2_prob": item["top2_prob"],
            "margin_top1_top2": item["margin"],
            "top2_label": item["top2"],
        })

    validation_csv = tar_export_dir / f"{tar_name}{VALIDATION_OUTPUT_SUFFIX}"
    validation_df = pd.DataFrame(
        validation_rows,
        columns=[
            "image",
            "label",
            "original_label",
            "translated_label",
            "user",
            "subset",
            "folder",
            "top1_prob",
            "top2_prob",
            "margin_top1_top2",
            "top2_label",
        ],
    )
    validation_df.to_csv(validation_csv, index=False)

    print(f"       ✅ Copied {copied} images into {len(taxa_counts)} taxa")
    print(
        f"       ✅ High-gap: {highgap_copied} copied (limit {high_gap_limit}/taxon, "
        f"margin ≥ {high_gap_margin:.2f})"
    )
    print(f"       ✅ Low-gap: {lowgap_copied} copied (limit {low_gap_limit}/taxon)")
    print(f"       ✅ Wrote validation CSV: {validation_csv.name} ({len(validation_df)} rows)")
    if missing:
        print(f"       ⚠️ Missing source files during taxon export: {missing}")
    if skipped_limit:
        print("       ⚠️ Skipped "
              f"{skipped_limit} images due to per-taxon bucket caps "
              f"({high_gap_limit} high-gap / {low_gap_limit} low-gap)")

    return {
        "copied": copied,
        "missing": missing,
        "taxa": len(taxa_counts),
        "highgap_copied": highgap_copied,
        "lowgap_copied": lowgap_copied,
        "skipped_limit": skipped_limit,
        "validation_csv": str(validation_csv),
    }


from time import time as timer
import gc
import stat

def _retry_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def clear_untarred_dir(dir_path, retries=8, delay=1.0):
    start_time = timer()
    dir_path = Path(dir_path)
    print(f"⚙ Clear and created local directories: {dir_path}")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            gc.collect()

            if dir_path.exists():
                shutil.rmtree(dir_path, onerror=_retry_remove_readonly)

            dir_path.mkdir(parents=True, exist_ok=True)

            elapsed_time = timer() - start_time
            print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
            return elapsed_time

        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"       ⚠️ Cleanup retry {attempt}/{retries - 1} for {dir_path}: {e}")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to clear directory {dir_path}: {e}") from e


def clear_temp_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_files = 0
    removed_dirs = 0

    for item in output_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                removed_files += 1
            elif item.is_dir():
                shutil.rmtree(item, onerror=_retry_remove_readonly)
                removed_dirs += 1
        except Exception as e:
            print(f"       ⚠️ Could not remove temp output item {item}: {e}")

    print(f"       ✅ Cleared temp output: {removed_files} file(s), {removed_dirs} folder(s).")


def remove_empty_dirs(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return

    removed = 0
    for dirpath, _, _ in os.walk(root_dir, topdown=False):
        path = Path(dirpath)
        if path == root_dir:
            continue

        try:
            if not any(path.iterdir()):
                path.rmdir()
                removed += 1
        except Exception as e:
            print(f"       ⚠️ Could not remove empty folder {path}: {e}")

    if removed:
        print(f"       ✅ Removed {removed} empty folder(s) from {root_dir}")


def extract_tar(tar_path, extract_to):
    start_time = timer()  # Start timing
    print(f"⚙ Untarring {tar_path.name}...")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)  # Extract the TAR file

    elapsed_time = timer() - start_time  # Calculate elapsed time
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return elapsed_time  # Return the time taken


def count_images_in_tar(extract_dir, tar_file):
    """Count the number of .tif images in the extracted directory."""
    print(f"⚙ Counting images in {tar_file.name}...")
    tif_files = list(extract_dir.rglob("*.tif"))
    print(f"       ✅ Found {len(tif_files)} .tif files")
    return len(tif_files)

def copy_background_tif(extract_dir, dest_path):
    start = timer()
    print("⚙ Copying Background..")

    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == "Background.tif":
                full_path = os.path.join(root, f)
                if not dest_path.exists():
                    shutil.copy(full_path, dest_path)
                elapsed_time = timer() - start
                print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
                return  # exit after success

    # If loop finishes without finding the file
    elapsed_time = timer() - start
    print(f"       ⚠️ Background.tif not found")



def extract_hitsmisses(tar_path, output_file, tar_file, status_log):
    start = timer()
    print("⚙ Fetching hits and misses...")

    with tarfile.open(tar_path) as tar:
        hits_file = next((m for m in tar.getmembers() if "hitsmisses.txt" in m.name.lower()), None)
        if hits_file:
            f = tar.extractfile(hits_file)
            df = pd.read_csv(f, header=None)
            df.columns = ['hits', 'misses']
            df['minute'] = range(len(df))
            df['tar_source'] = tar_path.stem
            df.to_csv(output_file, index=False)

            # Calculate RaisingFactor (sum of hits and misses divided by hits)
            df['RaisingFactor'] = df['hits']/(df['hits'] + df['misses'])

            #  Check row count
            if len(df) != REQUIRED_HITSMISSES_ROWS:
                reason = (
                    f"hitsmisses row count {len(df)} != "
                    f"{REQUIRED_HITSMISSES_ROWS}"
                )
                status_log.append(reason)
                print(f"       🚨 {reason}; will quarantine")

                # Optionally, clear the hitsmisses.txt if needed
                try:
                    if output_file.exists():
                        os.remove(output_file)
                        status_log.append(f"Removed hitsmisses.txt due to quarantine")
                except Exception as e:
                    status_log.append(f"⚠️ Failed to remove hitsmisses.txt: {e}")

                return False, reason, quarantine_hitsmiss_dir

            total_hits = df["hits"].sum()
            total_misses = df["misses"].sum()

            if total_hits > 0 and total_misses > total_hits * MAX_MISS_HIT_RATIO:
                reason = (
                    "misses too high "
                    f"({total_misses} > {MAX_MISS_HIT_RATIO:g}x {total_hits})"
                )
                status_log.append(reason)
                print(f"       🚨 {reason}; will quarantine")

                try:
                    if output_file.exists():
                        os.remove(output_file)
                        status_log.append("Removed hitsmisses.txt due to quarantine")
                except Exception as e:
                    status_log.append(f"âš ï¸ Failed to remove hitsmisses.txt: {e}")

                return False, reason, quarantine_raisingfactor_dir
        else:
            reason = "hitsmisses.txt not found in TAR"
            status_log.append(reason)
            print(f"       âš ï¸ hitsmisses.txt not found in {tar_file.name}")
            return False, reason, quarantine_hitsmiss_dir
    elapsed_time = timer() - start
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return True, None, None



def get_preview_sample_tifs(extract_dir, n=200):
    tif_files = [
        p for p in extract_dir.rglob("*.tif")
        if p.name not in {"Background.tif", "FlowCellEdges.tif"}
    ]

    if not tif_files:
        print(f"       ⚠️ No valid preview TIFFs found in {extract_dir}")
        return []

    if len(tif_files) <= n:
        return tif_files

    return random.sample(tif_files, n)


import os
import time
import json
import subprocess
import pandas as pd

def parse_exif_datetime_series(series):
    parsed = pd.to_datetime(series, format="%Y:%m:%d %H:%M:%S", errors="coerce")

    mask = parsed.isna() & series.notna()
    if mask.any():
        parsed_tz = pd.to_datetime(
            series.loc[mask],
            format="%Y:%m:%d %H:%M:%S%z",
            errors="coerce",
            utc=True
        )
        if getattr(parsed_tz.dt, "tz", None) is not None:
            parsed_tz = parsed_tz.dt.tz_localize(None)
        parsed.loc[mask] = parsed_tz

    return parsed


def extract_exif_metadata(tif_paths, tar_source, batch_size=200, exiftool_path=None):
    print("⚙ Extracting EXIF metadata in batch...")
    if exiftool_path is None:
        exiftool_path = globals()["exiftool_path"]
    exiftool_path = str(exiftool_path)
    exiftool_on_path = shutil.which(exiftool_path)
    if exiftool_on_path:
        exiftool_path = exiftool_on_path
    elif not Path(exiftool_path).exists():
        raise FileNotFoundError(
            f"ExifTool executable not found: {exiftool_path}. "
            "Install ExifTool or set paths.exiftool_path in predict_gpu_config.json."
        )

    tif_paths = [
        str(p) for p in tif_paths
        if os.path.basename(str(p)) not in IGNORED_TIF_NAMES
    ]
    if not tif_paths:
        print("       ⚠️ No valid TIFF files for EXIF extraction.")
        return pd.DataFrame()

    n_batches = (len(tif_paths) + batch_size - 1) // batch_size
    all_rows = []

    total_start_time = time.time()

    # ExifTool tag list once
    tags = [
        "-GPSLatitude",
        "-GPSLongitude",
        "-FileModifyDate",
        "-DateTimeOriginal",
        "-CreateDate",
        "-ModifyDate",
    ]

    # JSON output makes parsing reliable
    base_args = [
        exiftool_path,
        "-j",            # JSON output
        "-n",            # numeric values (e.g., GPS as decimals)
        "-api", "QuickTimeUTC",
        "-api", "ExifToolVersion=12.31",  # keep if you truly need it
    ] + tags

    for batch_idx in range(n_batches):
        batch = tif_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        args = base_args + batch

        try:
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"       ❌ Exiftool error (batch {batch_idx}): {result.stderr.strip()}")
                continue

            # ExifTool -j returns a JSON list of dicts (one per file)
            rows = json.loads(result.stdout) if result.stdout.strip() else []
            all_rows.extend(rows)

        except Exception as e:
            print(f"       ❌ Exception in batch {batch_idx}: {e}")
            continue

    elapsed = time.time() - total_start_time
    print(f"       ✅ Done in {elapsed:.2f} seconds ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["tar_source"] = tar_source

    if "SourceFile" in df.columns:
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(str(x)))

    if "tif_name" in df.columns:
        df = df[~df["tif_name"].isin(IGNORED_TIF_NAMES)].copy()
        if df.empty:
            return df

    # Optional: parse a preferred datetime column (pick one that exists)
    dt_col = next((c for c in ["FileModifyDate", "DateTimeOriginal", "CreateDate", "ModifyDate"] if c in df.columns),
                  None)
    if dt_col:
        # ExifTool dates often look like "YYYY:MM:DD HH:MM:SS" (sometimes with timezone)
        df[dt_col + "_parsed"] = parse_exif_datetime_series(df[dt_col])

        # If you really want a formatted string, avoid %-m/%-d on Windows (use %#m/%#d)
        # df[dt_col + "_fmt"] = df[dt_col + "_parsed"].dt.strftime("%#m/%#d/%Y  %#I:%M:%S %p")

    return df



def write_exif_csvs(df, tar_name, output_dir, backup_dir):
    if df is None or df.empty:
        raise ValueError(f"No EXIF metadata rows available for {tar_name}")

    # Ensure tif_name exists
    if "tif_name" not in df.columns and "SourceFile" in df.columns:
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(str(x)))

    if "tif_name" in df.columns:
        df = df[~df["tif_name"].isin(IGNORED_TIF_NAMES)].copy()

    # Accept multiple possible timestamp fields
    time_keys = ["DateTimeOriginal", "FileModifyDate", "CreateDate", "ModifyDate"]

    cols = []
    if "SourceFile" in df.columns: cols.append("SourceFile")
    cols.append("tif_name")

    for c in ["GPSLatitude", "GPSLongitude"] + time_keys:
        if c in df.columns:
            cols.append(c)

    df = df[cols]

    outname = f"{tar_name}_gpstag.csv"
    (output_dir / outname).parent.mkdir(parents=True, exist_ok=True)
    (backup_dir / outname).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / outname, index=False)
    df.to_csv(backup_dir / outname, index=False)
    #print(f"✅ Saved EXIF CSV with GPS/timestamps: {outname}")



def getImageRegionList(filename):
    image = imread(filename)
    if image.ndim == 3:
        image = rgb2gray(image)
    image_threshold = np.where(image > np.mean(image), 0., 1.0)
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))
    label_list = measure.label(image_dilated)
    label_list = (image_threshold * label_list).astype(int)
    return measure.regionprops(label_list)

def getMaxAreaDict(filename):
    regions = getImageRegionList(filename)
    if not regions:
        return {'object_additional_area': 0}
    r = max(regions, key=lambda x: x.area)
    return {
        'object_additional_diameter_equivalent': r.equivalent_diameter,
        'object_additional_length_minor_axis': r.minor_axis_length,
        'object_additional_length_major_axis': r.major_axis_length,
        'object_additional_eccentricity': r.eccentricity,
        'object_additional_area': r.area,
        'object_additional_perimeter': r.perimeter,
        'object_additional_orientation': r.orientation,
        'object_additional_area_convex': r.convex_area,
        'object_additional_area_filled': r.filled_area,
        'object_additional_box_min_row': r.bbox[0],
        'object_additional_box_max_row': r.bbox[2],
        'object_additional_box_min_col': r.bbox[1],
        'object_additional_box_max_col': r.bbox[3],
        'object_additional_ratio_extent': r.extent,
        'object_additional_ratio_solidity': r.solidity,
        'object_additional_inertia_tensor_eigenvalue1': r.inertia_tensor_eigvals[0],
        'object_additional_inertia_tensor_eigenvalue2': r.inertia_tensor_eigvals[1],
        'object_additional_moments_hu1': r.moments_hu[0],
        'object_additional_moments_hu2': r.moments_hu[1],
        'object_additional_moments_hu3': r.moments_hu[2],
        'object_additional_moments_hu4': r.moments_hu[3],
        'object_additional_moments_hu5': r.moments_hu[4],
        'object_additional_moments_hu6': r.moments_hu[5],
        'object_additional_moments_hu7': r.moments_hu[6],
        'object_additional_euler_number': r.euler_number,
        'object_additional_countcoords': len(r.coords)
    }

def classify_and_extract_regions(tar_file, extract_dir):
    start_time = time.time()
    base_name = tar_file.stem
    json_path = source_dir / f"{base_name}_predictions_relative.json"
    csv_path = source_dir / f"{base_name}_image_properties.csv"
    FILEPATHS = list(extract_dir.rglob("*.tif"))

    # Filter only useful files
    FILEPATHS = [p for p in FILEPATHS if "Background.tif" not in p.name and "FlowCellEdges.tif" not in p.name]

    if not FILEPATHS:
        print(f"⚠️ No valid .tif files in {base_name}, skipping.")
        return

    print(f"⚙ Predicting {len(FILEPATHS)} TIFF files")

    # Run prediction
    pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

    results_json = []
    results_csv = []

    for i, path in enumerate(FILEPATHS):
        rel_path = str(path.relative_to(extract_dir))

        # === JSON prediction ===
        labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
        probs = [float(pred_prob[i, j]) for j in range(TOP_K)]
        results_json.append({
            "filepath": rel_path,
            f"top{TOP_K}_labels": labels,
            f"top{TOP_K}_probs": probs
        })

        # === Morphology extraction ===
        try:
            props = getMaxAreaDict(path)
            props["filepath"] = rel_path
            results_csv.append(props)
        except Exception as e:
            print(f"       ❌ Error processing {rel_path}: {e}")

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    elapsed_time = time.time() - start_time
    print(f"       ✅ Done in {elapsed_time / 3600:.1f} hours.")

    # Save CSV
    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        #print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️No region properties written for {base_name}")
    return pred_lab  # at the end

import os
import json
import pandas as pd


####SET BARS
DEFAULT_TAXA_THRESHOLDS = {
    "diatom-setae": {"upper_threshold": 0.999, "diff_threshold": 0.99},
    "dinoflagellate_noctiluca-intact": {"upper_threshold": 0.99, "diff_threshold": 0.99},
}

BUBBLES_RULE = {"upper_threshold": 0.999, "diff_threshold": 0.99}
BUBBLES_SUBSTR = "bubbles"

DETRITUS_RULE = {"upper_threshold": 0.999, "diff_threshold": 0.99}
DETRITUS_SUBSTR = "noctiluca"


def generate_topspecies_csv(json_path,
                            taxa_thresholds=None,
                            decimals=6):
    print("⚙ Generating top species CSV")

    # Ensure taxa_thresholds is passed as a dictionary
    # Use default thresholds if none provided
    if taxa_thresholds is None:
        taxa_thresholds = DEFAULT_TAXA_THRESHOLDS

    if not isinstance(taxa_thresholds, dict) or not taxa_thresholds:
        print("       ❌ Taxa thresholds must be a non-empty dict.")
        return

    # Accept str or Path
    json_path = Path(json_path)

    if not json_path.exists():
        print(f"       ❌ JSON file not found: {json_path}")
        return

    # Read JSON
    try:
        with open(json_path, "r") as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON: {e}")
        return

    if not isinstance(data_list, list) or not data_list:
        print("       ⚠️ JSON is empty or invalid.")
        return

    rows = []
    for entry in data_list:
        filepath = entry.get("filepath", "")
        labels = (
            entry.get(f"top{TOP_K}_labels", [])
            or entry.get("top3_labels", [])
            or entry.get("top2_labels", [])
        )
        probs = (
            entry.get(f"top{TOP_K}_probs", [])
            or entry.get("top3_probs", [])
            or entry.get("top2_probs", [])
        )
        taxa = entry.get("taxa", "")  # Assuming taxa is in the entry

        # Coerce "labels/probs" in case they're comma-separated strings
        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",")]
        if isinstance(probs, str):
            try:
                probs = [float(s.strip()) for s in probs.split(",")]
            except ValueError:
                probs = []

        # Need at least top-2 to compute the margin
        if not filepath or len(labels) < 2 or len(probs) < 2:
            continue

        label = labels[0]
        prob1 = float(probs[0])
        prob2 = float(probs[1])

        # defaults
        upper_threshold = 0.999
        diff_threshold = 0.99

        # choose thresholds (exact match first)
        if taxa in taxa_thresholds:
            upper_threshold = taxa_thresholds[taxa].get("upper_threshold", upper_threshold)
            diff_threshold = taxa_thresholds[taxa].get("diff_threshold", diff_threshold)
        elif (isinstance(taxa, str) and BUBBLES_SUBSTR in taxa.lower()) or \
                (isinstance(label, str) and BUBBLES_SUBSTR in label.lower()):
            upper_threshold = BUBBLES_RULE["upper_threshold"]
            diff_threshold = BUBBLES_RULE["diff_threshold"]
        elif (isinstance(taxa, str) and DETRITUS_SUBSTR in taxa.lower()) or \
                (isinstance(label, str) and DETRITUS_SUBSTR in label.lower()):
            upper_threshold = DETRITUS_RULE["upper_threshold"]
            diff_threshold = DETRITUS_RULE["diff_threshold"]

        # Conditionally append _AI99 based on taxa-specific thresholds
        if prob1 > upper_threshold and (prob1 - prob2) > diff_threshold:
            label = f"{label}_AI99"

        rows.append({
            "filename": os.path.basename(filepath),
            "top_species": str(label).strip(),
            "confidence": prob1  # <-- keep top-1 probability
        })

    if not rows:
        print("       ⚠️ No valid predictions to save.")
        return

    # Write CSV next to the JSON with *_topspecies.csv name
    output_path = json_path.with_name(
        json_path.name.replace("_predictions_relative.json", "_topspecies.csv")
    )

    # Ensure column order + consistent float formatting
    df = pd.DataFrame(rows, columns=["filename", "top_species", "confidence"])
    df.to_csv(output_path, index=False, float_format=f"%.{decimals}f")
    print("       ✅ Done")


def check_preview_class_distribution(preview_tifs, threshold=PREVIEW_BUBBLE_THRESHOLD):
    print("⚙ Running preview classification check...")

    preview_tifs = list(preview_tifs or [])
    if not preview_tifs:
        print("       ⚠️ No preview sample available.")
        return True, None  # Allow pipeline to continue

    pred_lab, pred_prob = predict(model, preview_tifs, conf, top_K=1, filemode='local')
    top1_classes = [class_names[idx] for idx in pred_lab[:, 0]]

    class_counts = pd.Series(top1_classes).value_counts(normalize=True)
    print(f"       ✅ Class distribution in preview: {class_counts.to_dict()}")

    bubble_classes = [cls for cls in class_counts.index if 'bubbles' in cls.lower()]
    bubble_fraction = class_counts[bubble_classes].sum()

    if bubble_fraction > threshold:
        print(f"✅ Combined 'bubbles'-like classes exceed threshold ({threshold:.0%}): {bubble_fraction:.2%}, moved to quarantine")
        return False, 'bubbles'

    return True, None

def extract_only_morphology(tar_file, extract_dir, csv_path):
    base_name = tar_file.stem
    FILEPATHS = list(extract_dir.rglob("*.tif"))
    FILEPATHS = [p for p in FILEPATHS if "Background.tif" not in p.name and "FlowCellEdges.tif" not in p.name]

    if not FILEPATHS:
        print(f"       ⚠️ No valid .tif files in {base_name}, skipping morphology.")
        return

    print(f"       🧬 Extracting morphology for {len(FILEPATHS)} TIFFs...")

    results_csv = []

    for path in FILEPATHS:
        rel_path = str(path.relative_to(extract_dir))
        try:
            props = getMaxAreaDict(path)
            props["filepath"] = rel_path
            results_csv.append(props)
        except Exception as e:
            print(f"       ❌ Morphology error for {rel_path}: {e}")

    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️ No morphology data written for {base_name}")


def safe_bg_coords(bg_path):
    lat, lon = get_background_coordinates(bg_path)
    return (lat if lat is not None else 0, lon if lon is not None else 0)


def log_per_minute_metrics(tar_name, json_output, hits_file, exif_df, out_dir, num_images):
    import os

    # Volume per minute (fixed)
    V_m3 = 0.034  # 34 L/min = 0.034 m³/min

    try:
        # --- 0) Load hits/misses ---
        if not hits_file.exists():
            print(f"⚠️ hitsmisses.txt missing for {tar_name}")
            return

        try:
            df_hits = pd.read_csv(hits_file)  # with header
        except Exception:
            df_hits = pd.read_csv(hits_file, header=None)
            df_hits.columns = ["hits", "misses"]

        # --- 1) Directly assign tar_name to tar_source column ---
        df_hits["tar_source"] = tar_name

        if "minute" not in df_hits.columns:
            df_hits["minute"] = range(len(df_hits))

        # Ensure necessary columns exist
        if "GPSLatitude" not in df_hits.columns:
            df_hits["GPSLatitude"] = "NA"  # Initialize as NA if missing
        if "GPSLongitude" not in df_hits.columns:
            df_hits["GPSLongitude"] = "NA"  # Initialize as NA if missing
        if "total_images_in_tar" not in df_hits.columns:
            df_hits["total_images_in_tar"] = num_images  # You can set this directly from the input argument

        max_minute = int(df_hits["minute"].max())

        # --- 2) Prepare EXIF → assign minutes sequentially ---
        lat_by_minute, lon_by_minute = {}, {}
        if exif_df is not None and not exif_df.empty:
            df_exif = exif_df.copy()
            if "tif_name" not in df_exif.columns and "SourceFile" in df_exif.columns:
                df_exif["tif_name"] = df_exif["SourceFile"].apply(lambda x: os.path.basename(str(x)))
            if "tif_name" in df_exif.columns:
                df_exif = df_exif[~df_exif["tif_name"].isin(IGNORED_TIF_NAMES)].copy()

            ts_cols = [c for c in ["DateTimeOriginal", "FileModifyDate", "ModifyDate"] if c in df_exif.columns]

            if ts_cols:
                df_exif["capture_dt"] = pd.NaT
                for col in ts_cols:
                    s = pd.to_datetime(
                        df_exif[col],
                        format="%Y:%m:%d %H:%M:%S",  # EXIF datetime format
                        errors="coerce",
                        utc=True
                    )
                    # make tz-naive so it matches df_exif["capture_dt"]
                    if s.dt.tz is not None:
                        s = s.dt.tz_localize(None)

                    mask = df_exif["capture_dt"].isna()
                    df_exif.loc[mask, "capture_dt"] = s[mask]

                df_exif = df_exif.sort_values("capture_dt").reset_index(drop=True)

            else:
                df_exif = df_exif.reset_index(drop=True)

            n = len(df_exif)
            rows_per_min = max(1, n // (max_minute + 1))
            df_exif["minute"] = df_exif.index // rows_per_min
            df_exif["minute"] = df_exif["minute"].clip(0, max_minute)

            for c in ["GPSLatitude", "GPSLongitude"]:
                if c in df_exif.columns:
                    df_exif[c] = pd.to_numeric(df_exif[c], errors="coerce")

            coords_df = (df_exif
                         .dropna(subset=["GPSLatitude", "GPSLongitude"])
                         .groupby("minute", as_index=False)[["GPSLatitude", "GPSLongitude"]]
                         .median())
            lat_by_minute = dict(zip(coords_df["minute"], coords_df["GPSLatitude"]))
            lon_by_minute = dict(zip(coords_df["minute"], coords_df["GPSLongitude"]))

            exif_df = df_exif

        # --- 3) Map EXIF data to df_hits for GPS and fill in missing values ---
        # Assign GPS values from EXIF if they exist
        df_hits["GPSLatitude"] = df_hits["minute"].map(lambda m: lat_by_minute.get(m, "NA"))
        df_hits["GPSLongitude"] = df_hits["minute"].map(lambda m: lon_by_minute.get(m, "NA"))

        # --- 4) noctiluca + bubble counts per minute ---
        noct_counts = {m: 0 for m in df_hits["minute"]}
        bubble_counts = {m: 0 for m in df_hits["minute"]}

        top1_by_name = {}
        if json_output.exists() and exif_df is not None and "tif_name" in exif_df.columns:
            name_to_minute = exif_df.set_index("tif_name")["minute"].to_dict()
            with open(json_output, "r") as f:
                data = json.load(f)

            for entry in data:
                fname = os.path.basename(entry.get("filepath", ""))
                labels = (
                    entry.get(f"top{TOP_K}_labels", [])
                    or entry.get("top3_labels", [])
                    or entry.get("top2_labels", [])
                )
                if isinstance(labels, str):
                    labels = [s.strip() for s in labels.split(",")]

                if labels:
                    top1_by_name[fname] = labels[0]

                m = name_to_minute.get(fname, None)
                if m is not None and len(labels) >= 1:
                    if "noct" in labels[0].lower():
                        noct_counts[m] = noct_counts.get(m, 0) + 1
                    if any("bubb" in lab.lower() for lab in labels):
                        bubble_counts[m] = bubble_counts.get(m, 0) + 1

        # --- 5) Morphometrics ---
        diameter_mean = {m: 0 for m in df_hits["minute"]}
        diameter_sum = {m: 0 for m in df_hits["minute"]}

        img_props_path = out_dir / f"{tar_name}_image_properties.csv"
        if img_props_path.exists() and exif_df is not None and "tif_name" in exif_df.columns:
            df_props = pd.read_csv(img_props_path)
            if "filepath" in df_props.columns and "object_additional_diameter_equivalent" in df_props.columns:
                # build mapping: filename -> minute
                name_to_minute = exif_df.set_index("tif_name")["minute"].to_dict()

                for _, row in df_props.iterrows():
                    fname = os.path.basename(str(row["filepath"]))
                    m = name_to_minute.get(fname, None)
                    top1 = str(top1_by_name.get(fname, "")).lower()
                    if m is not None and "noct" in top1:
                        d = row["object_additional_diameter_equivalent"]
                        if pd.notna(d):
                            diameter_sum[m] = diameter_sum.get(m, 0) + d

                # Turn sums into means (divide by noctiluca_count where >0)
                for m in diameter_mean:
                    if noct_counts.get(m, 0) > 0:
                        diameter_mean[m] = diameter_sum[m] / noct_counts[m]
                    else:
                        diameter_sum[m] = 0
                        diameter_mean[m] = 0

        # attach to df_hits
        df_hits["noctiluca_diam_mean"] = df_hits["minute"].map(diameter_mean)
        df_hits["noctiluca_diam_sum"] = df_hits["minute"].map(diameter_sum)

        # --- 6) Merge and compute densities ---
        df_hits["noctiluca_count"] = df_hits["minute"].map(noct_counts)
        df_hits["bubbles"] = df_hits["minute"].map(bubble_counts)

        # Proportion of Noctiluca in Hits (for density)
        proportion_noctiluca_in_hits = df_hits["noctiluca_count"] / df_hits["hits"]

        # Calculate Noctiluca in Misses (based on the proportion in hits)
        df_hits["noctiluca_in_misses"] = proportion_noctiluca_in_hits * df_hits["misses"]

        # Total noctiluca count (hits + misses)
        df_hits["total_noctiluca"] = df_hits["noctiluca_count"] + df_hits["noctiluca_in_misses"]

        # Densities (individuals per m³)
        df_hits["noctiluca_density_ind_m3"] = df_hits["total_noctiluca"] / V_m3

        # Reorder columns
        df_hits = df_hits[[
            "tar_source", "minute", "hits", "misses",
            "bubbles", "noctiluca_count", "total_noctiluca", "noctiluca_density_ind_m3",
            "GPSLatitude", "GPSLongitude", "total_images_in_tar", "noctiluca_diam_mean", "noctiluca_diam_sum"
        ]]

        # Force proper dtypes before saving
        numeric_cols = [
            "minute", "hits", "misses",
            "bubbles","noctiluca_count", "total_noctiluca", "noctiluca_density_ind_m3",
            "GPSLatitude", "GPSLongitude", "total_images_in_tar", "noctiluca_diam_mean", "noctiluca_diam_sum"
        ]

        for col in numeric_cols:
            if col in df_hits.columns:
                df_hits[col] = pd.to_numeric(df_hits[col], errors="coerce")

        # Save clean CSV
        out_path = out_dir / f"{tar_name}_bio-metrics.csv"
        df_hits.to_csv(out_path, index=False, float_format="%.6f")  # control decimals
        print("       ✅ Done")

    except Exception as e:
        print(f"❌ Failed per-minute log for {tar_name}: {e}")
        try:
            out_path = out_dir / f"{tar_name}_bio-metrics.csv"
            df_hits.to_csv(out_path, index=False)
        except Exception:
            pass



def map_exif_to_minutes(exif_df, hits_len):
    # Example: use the file index pattern from filename "_0001_"
    exif_df["minute"] = None
    for i, row in exif_df.iterrows():
        fname = row.get("tif_name", "")
        for m in range(hits_len):
            if f"_{m:04d}_" in fname or f"_{m:03d}_" in fname:
                exif_df.at[i, "minute"] = m
                break
    return exif_df
def clean_coord(value):
    # If value is tuple or list, flatten to string
    if isinstance(value, (tuple, list)):
        return ",".join(map(str, value))
    return value if value is not None else "NA"


# === MAIN PROCESS ===
def process_tar(tar_file):
    tar_name = tar_file.stem
    print(f"\n🔧🔧🔧 PROCESSING {tar_name.upper()} 🔧🔧🔧")

    clear_temp_output_dir(paths["output"])
    remove_empty_dirs(paths["untarred"])

    # Tracking variables
    times_dict = {}
    num_images = 0
    status_log = []
    quarantined = False
    quarantine_reason = None
    already_logged = False
    pred_lab = None
    exif_df = None  # to pass into log_per_minute_metrics later

    if outputs_exist_for_tar(tar_file):
        print(f"📦 All outputs exist for {tar_name}, skipping.")
        return

    # Define paths
    json_output = source_dir / f"{tar_name}_predictions_relative.json"
    csv_output = source_dir / f"{tar_name}_image_properties.csv"
    topspecies_csv = source_dir / f"{tar_name}_topspecies.csv"
    exif_csv_path = source_dir / f"{tar_name}_gpstag.csv"
    hits_path = source_dir / f"{tar_name}_hitsmisses.txt"
    bg_path = source_dir / f"{tar_name}_Background.tif"
    tar_dest = paths['tarred'] / tar_file.name
    extract_dir = paths['untarred'] / tar_name

    try:
        # === Step 1: Copy TAR to work dir ===
        start_time = time.time()
        shutil.copy(tar_file, tar_dest)
        status_log.append("TAR copied to working directory")
        times_dict["Copy TAR to working directory"] = track_time(start_time, "Copy TAR to working directory")

        # === Step 2: Untar ===
        start_time = time.time()
        paths['untarred'].mkdir(parents=True, exist_ok=True)
        clear_untarred_dir(extract_dir)
        extract_tar(tar_dest, extract_dir)
        status_log.append("Untarred successfully")
        times_dict["Untar"] = track_time(start_time, "Untar")

        # === Step 2b: Gray-edge quarantine check ===
        start_time = time.time()
        should_continue, reason, gray_edge_mean = check_gray_edge_quarantine(extract_dir, tar_name)
        status_log.append(
            f"Gray-edge check result: {reason if reason else 'OK'}"
            + (f" (mean={gray_edge_mean:.2f})" if gray_edge_mean is not None else "")
        )
        times_dict["Gray edge quarantine check"] = track_time(
            start_time,
            "Gray edge quarantine check",
        )

        if not should_continue:
            quarantined = True
            quarantine_reason = reason
            quarantine_target = quarantine_gray_edge_dir / tar_file.name

            try:
                if quarantine_target.exists():
                    quarantine_target.unlink()
                if tar_file.exists():
                    shutil.move(str(tar_file), str(quarantine_target))
                    status_log.append(f"Moved to quarantine-gray-edge: {quarantine_target}")
                    print(f"Quarantined {tar_file.name}: gray-edge issue")
                remove_partial_outputs(tar_name, status_log)
            except Exception as mv_err:
                status_log.append(f"Gray-edge quarantine move failed: {mv_err}")

        # === Step 3: Extract hitsmisses.txt ===
        start_time = time.time()
        if quarantined:
            status_log.append("Skipped hitsmisses extraction because TAR was already quarantined")
        elif not hits_path.exists():
            ok, hitsmiss_reason, hitsmiss_quarantine_dir = extract_hitsmisses(
                tar_file,
                hits_path,
                tar_file,
                status_log,
            )
            if not ok:
                quarantined = True
                quarantine_reason = hitsmiss_reason or "hits/misses check failed"
                status_log.append(f"Quarantined: {quarantine_reason}")
                quarantine_target = (hitsmiss_quarantine_dir or quarantine_hitsmiss_dir) / tar_file.name
                try:
                    if quarantine_target.exists():
                        quarantine_target.unlink()
                    if tar_file.exists():
                        shutil.move(str(tar_file), str(quarantine_target))
                        status_log.append(f"Moved TAR to quarantine: {quarantine_target}")
                    remove_partial_outputs(tar_name, status_log)
                except Exception as mv_err:
                    status_log.append(f"⚠️ Quarantine move failed: {mv_err}")
        else:
            status_log.append("hitsmisses.txt already exists (skipped)")
        times_dict["Extract hitsmisses.txt"] = track_time(start_time, "Extract hitsmisses.txt")

        # === Step 4–11 only if not quarantined ===
        if not quarantined:
            # Step 4: Count images
            start_time = time.time()
            num_images = count_images_in_tar(extract_dir, tar_file)
            status_log.append(f"Number of images in TAR: {num_images}")
            times_dict["Count images"] = track_time(start_time, "Count images")

            # Step 5: Sample preview subset in memory (not stored)
            start_time = time.time()
            preview_sample_tifs = get_preview_sample_tifs(extract_dir, n=200)
            status_log.append(
                f"Preview subset sampled in memory ({len(preview_sample_tifs)} images, not stored)"
            )
            times_dict["Create preview images"] = track_time(start_time, "Create preview images")

            # Step 6: Early preview classification
            start_time = time.time()
            if json_output.exists():
                print("✅ Skipping preview classification (predictions already exist)")
                should_continue, reason = True, None
            else:
                should_continue, reason = check_preview_class_distribution(
                    preview_sample_tifs,
                    threshold=PREVIEW_BUBBLE_THRESHOLD,
                )
            status_log.append(f"Preview classification result: {reason if reason else 'OK'}")
            times_dict["Early preview classification"] = track_time(start_time, "Early preview classification")

            if not should_continue:
                quarantined = True
                quarantine_reason = reason
                quarantine_target = quarantine_bubbles_dir / tar_file.name
                try:
                    shutil.move(str(tar_file), str(quarantine_target))
                    remove_partial_outputs(tar_name, status_log)
                    print(f"🚨 Quarantined {tar_file.name} → bubble issue")
                except Exception as mv_err:
                    status_log.append(f"⚠️ Quarantine move failed: {mv_err}")
                status_log.append(f"Moved to quarantine due to '{quarantine_reason}' class")

            # Step 7–11 only if still not quarantined
            if not quarantined:
                # Step 7: Copy Background.tif
                start_time = time.time()
                if not bg_path.exists():
                    copy_background_tif(extract_dir, bg_path)
                    if bg_path.exists():
                        status_log.append("Background.tif copied successfully")
                    else:
                        status_log.append("❌ Background.tif missing after copy attempt")
                else:
                    status_log.append("Background.tif already exists (skipped)")
                times_dict["Copy Background.tif"] = track_time(start_time, "Copy Background.tif")

                # Step 8: Extract EXIF metadata (or load existing)
                start_time = time.time()
                if not exif_csv_path.exists():
                    tif_files = list(extract_dir.rglob("*.tif"))
                    exif_df = extract_exif_metadata(tif_files, tar_name)
                    if exif_df.empty:
                        raise RuntimeError(
                            f"EXIF extraction returned 0 rows for {tar_name}; "
                            "check ExifTool before continuing."
                        )
                    write_exif_csvs(exif_df, tar_name, paths['output'], source_dir)
                    status_log.append("EXIF metadata extracted and saved")
                else:
                    try:
                        exif_df = pd.read_csv(exif_csv_path)
                        if "tif_name" not in exif_df.columns and "SourceFile" in exif_df.columns:
                            exif_df["tif_name"] = exif_df["SourceFile"].apply(lambda x: os.path.basename(str(x)))
                        if "tif_name" in exif_df.columns:
                            filtered_exif_df = exif_df[~exif_df["tif_name"].isin(IGNORED_TIF_NAMES)].copy()
                            if len(filtered_exif_df) != len(exif_df):
                                exif_df = filtered_exif_df
                                exif_df.to_csv(exif_csv_path, index=False)
                        status_log.append("EXIF metadata loaded from CSV")
                    except Exception as rd_err:
                        status_log.append(f"⚠️ Failed to load existing EXIF CSV: {rd_err}")
                        exif_df = None
                    print("✅ Skipping EXIF extraction (already exists)")

                    # Debug print for GPS check
                    #if exif_df is not None and not exif_df.empty:
                    #    print("🔎 EXIF DataFrame head:")
                    #    print(exif_df.head(5).to_string())
                    #else:
                    #    print("⚠️ EXIF DataFrame is empty or missing columns")

                times_dict["Extract and save EXIF metadata"] = track_time(start_time, "Extract and save EXIF metadata")

                # Step 8b: GPS quarantine checks
                if not has_usable_coordinates(exif_df):
                    quarantined = True
                    quarantine_reason = "no usable GPS coordinates"
                    quarantine_target = quarantine_nogps_dir / tar_file.name

                    try:
                        if quarantine_target.exists():
                            quarantine_target.unlink()
                        if tar_file.exists():
                            shutil.move(str(tar_file), str(quarantine_target))
                            status_log.append(f"Moved to quarantine-nogps: {quarantine_reason}")
                            print(f"🚨 Quarantined {tar_file.name}: no usable GPS coordinates")
                        remove_partial_outputs(tar_name, status_log)
                    except Exception as mv_err:
                        status_log.append(f"⚠️ No-GPS quarantine move failed: {mv_err}")

                elif should_quarantine_location(exif_df):
                    quarantined = True
                    quarantine_reason = "within GPS quarantine radius"
                    quarantine_target = quarantine_near_point_dir / tar_file.name

                    try:
                        if quarantine_target.exists():
                            quarantine_target.unlink()
                        if tar_file.exists():
                            shutil.move(str(tar_file), str(quarantine_target))
                            status_log.append(
                                f"Moved to quarantine-location: {quarantine_reason}"
                            )
                            print(
                                f"🚨 Quarantined {tar_file.name}: collected within "
                                f"{QUARANTINE_RADIUS_M:g} m of quarantine point"
                            )
                        remove_partial_outputs(tar_name, status_log)
                    except Exception as mv_err:
                        status_log.append(f"⚠️ Location quarantine move failed: {mv_err}")

                if quarantined:
                    return

                # Step 9: Classification & morphology
                start_time = time.time()
                if not csv_output.exists():
                    if json_output.exists():
                        extract_only_morphology(tar_file, extract_dir, csv_output)
                        status_log.append("Image properties CSV created from existing predictions")
                    else:
                        pred_lab = classify_and_extract_regions(tar_file, extract_dir)
                        status_log.append("Classification and morphology run together (both files missing)")
                elif not json_output.exists():
                    pred_lab = classify_and_extract_regions(tar_file, extract_dir)
                    status_log.append("Re-ran full classification due to missing JSON")
                else:
                    status_log.append("Classification already exists (skipped)")
                times_dict["Classification and morphology extraction"] = track_time(
                    start_time, "Classification and morphology extraction"
                )

                # Step 10: Export images by top-1 taxon
                start_time = time.time()
                try:
                    if json_output.exists():
                        export_summary = export_images_by_top1_taxon(
                            extract_dir=extract_dir,
                            json_path=json_output,
                            tar_name=tar_name,
                            export_root=taxon_export_root
                        )
                        status_log.append(
                            f"Images exported by taxon ({export_summary['copied']} copied, {export_summary['taxa']} taxa; "
                            f"{export_summary['highgap_copied']} high-gap / {export_summary['lowgap_copied']} low-gap)"
                        )
                        if export_summary.get("validation_csv"):
                            status_log.append(
                                f"Validation CSV written: {Path(export_summary['validation_csv']).name}"
                            )
                        if export_summary["missing"]:
                            status_log.append(
                                f"⚠️ Taxon export missing source files: {export_summary['missing']}"
                            )
                    else:
                        status_log.append("Taxon image export skipped (JSON not found)")
                except Exception as export_err:
                    print(f"⚠️ Taxon export failed for {tar_name}: {export_err}")
                    status_log.append(f"⚠️ Taxon export failed: {export_err}")
                times_dict["Export images by taxon"] = track_time(start_time, "Export images by taxon")

                # Step 11: Generate top species CSV
                start_time = time.time()
                if not topspecies_csv.exists():
                    if json_output.exists():
                        generate_topspecies_csv(json_output)
                        status_log.append("Top species CSV generated")
                    else:
                        status_log.append("Top species CSV skipped (JSON not found)")
                else:
                    status_log.append("Top species CSV already exists (skipped)")
                times_dict["Generate top species CSV"] = track_time(
                    start_time, "Generate top species CSV"
                )

                # Step 12: Per-minute bio metrics
                start_time = time.time()
                try:
                    #print(f"⚙ Starting per-minute bio metrics")
                    log_per_minute_metrics(
                        tar_name,
                        json_output,
                        hits_path,
                        exif_df,
                        source_dir,
                        num_images
                    )

                    status_log.append("Per-minute bio metrics logged")
                except Exception as e:
                    print(f"❌ Failed per-minute log for {tar_name}: {e}")
                    status_log.append(f"❌ Failed per-minute bio metrics: {e}")
                times_dict["Per-minute bio metrics"] = track_time(start_time, "Per-minute bio metrics")

    except Exception as e:
        status_log.append(f"❌ Unexpected error: {e}")

    finally:
        # Cleanup tar + untar
        try:
            if tar_dest.exists():
                os.remove(tar_dest)
            clear_untarred_dir(extract_dir)
            clear_temp_output_dir(paths["output"])
            remove_empty_dirs(paths["untarred"])
        except Exception as cleanup_err:
            status_log.append(f"⚠️ Cleanup failed: {cleanup_err}")

        # Daily report
        daily_tar_reports.append({
            "tar_name": tar_name,
            "status_log": status_log,
            "quarantine_reason": quarantine_reason if quarantined else None
        })

        # Times log
        if not already_logged:
            try:
                log_time_to_file(tar_name, times_dict, num_images)
            except Exception as log_err:
                print(f"⚠️ Failed to log times for {tar_name}: {log_err}")

        print(f"🔧🔧🔧 DONE {tar_name} 🔧🔧🔧")

        # Mark done only when all required outputs exist
        if all_required_outputs_exist(tar_name):
            try:
                done_marker = source_dir / f"{tar_name}.done"
                with open(done_marker, "w") as f:
                    f.write(f"Processed at {datetime.datetime.now()}\n")
            except Exception as e:
                print(f"⚠️ Could not write done marker for {tar_name}: {e}")
        else:
            print(f"⚠️ Incomplete outputs for {tar_name}; not writing .done")




# === OPTION A TAR-STREAM OVERRIDES ===
# The functions below intentionally override earlier file-path based helpers.
# They preserve the remote worker/config/lock wrapper while making processing
# match ProcessingScript_OPTION_A_TAR_STREAM.py.

def export_images_by_top1_taxon(
    extract_dir,
    json_path,
    tar_name,
    export_root,
    high_gap_margin=PREVIEW_HIGH_GAP_MARGIN,
    high_gap_limit=PREVIEW_HIGH_GAP_LIMIT,
    low_gap_limit=PREVIEW_LOW_GAP_LIMIT,
):
    """Copy a capped validation subset by translated top-1 taxon.

    Fixes the previous undefined validation_csv return value and avoids reloading
    the translation CSV repeatedly by using the cached load_label_translation().
    """
    print("⚙ Exporting classified images by taxon (high-gap / low-gap preview buckets)...")

    json_path = Path(json_path)
    extract_dir = Path(extract_dir)
    export_root = Path(export_root)

    empty = {
        "copied": 0,
        "missing": 0,
        "taxa": 0,
        "highgap_copied": 0,
        "lowgap_copied": 0,
        "skipped_limit": 0,
        "validation_csv": None,
    }

    if not json_path.exists():
        print(f"       ⚠️ JSON file not found: {json_path}")
        return empty

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON for taxon export: {e}")
        return empty

    data = [entry for entry in data if isinstance(entry, dict)]
    label_translation = load_label_translation(str(CLASS_TRANSLATION_CSV))

    tar_export_dir = export_root / tar_name
    if tar_export_dir.exists():
        shutil.rmtree(tar_export_dir, onerror=_retry_remove_readonly)
    tar_export_dir.mkdir(parents=True, exist_ok=True)

    prepared_entries = []
    missing = 0

    for entry in data:
        rel_path = entry.get("filepath", "")
        labels, probs = parse_prediction_lists(entry)
        if not labels:
            continue

        original_top1 = str(labels[0]).strip()
        original_top2 = str(labels[1]).strip() if len(labels) > 1 else None
        translated_top1 = translate_label(original_top1, label_translation)
        translated_top2 = translate_label(original_top2, label_translation) if original_top2 else None

        top1 = sanitize_taxon_name(translated_top1)
        top2 = sanitize_taxon_name(translated_top2) if translated_top2 else None
        top1_prob = probs[0] if len(probs) > 0 else None
        top2_prob = probs[1] if len(probs) > 1 else 0.0

        same_translated_taxon = (
            translated_top2 is not None
            and sanitize_taxon_name(translated_top1) == sanitize_taxon_name(translated_top2)
        )
        same_taxon_high_confidence = (
            same_translated_taxon
            and top1_prob is not None
            and top2_prob is not None
            and float(top1_prob) >= 0.5
            and float(top2_prob) >= 0.1
        )

        if top1_prob is None:
            margin = None
        elif same_taxon_high_confidence:
            margin = float(top1_prob)
        elif top2_prob is not None:
            margin = float(top1_prob) - float(top2_prob)
        else:
            margin = float(top1_prob)

        if margin is not None and margin >= high_gap_margin:
            bucket = "highgap"
            folder_suffix = PREVIEW_HIGH_GAP_SUFFIX
            bucket_limit = high_gap_limit
        else:
            bucket = "lowgap"
            folder_suffix = PREVIEW_LOW_GAP_SUFFIX
            bucket_limit = low_gap_limit

        src = extract_dir / rel_path
        if not src.exists():
            missing += 1
            continue

        prepared_entries.append({
            "src": src,
            "top1": top1,
            "original_top1": original_top1,
            "translated_top1": translated_top1,
            "original_top2": original_top2,
            "translated_top2": translated_top2,
            "same_translated_taxon": same_translated_taxon,
            "top2": top2,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "margin": margin,
            "bucket": bucket,
            "folder_suffix": folder_suffix,
            "bucket_limit": bucket_limit,
            "same_taxon_high_confidence": same_taxon_high_confidence,
        })

    grouped = {}
    for item in prepared_entries:
        grouped.setdefault((item["top1"], item["bucket"]), []).append(item)

    selected = []
    skipped_limit = 0
    for (taxon, bucket), entries in grouped.items():
        rng = random.Random(f"{tar_name}|{taxon}|{bucket}")
        rng.shuffle(entries)
        limit = entries[0]["bucket_limit"] if entries else 0
        selected.extend(entries[:limit])
        skipped_limit += max(0, len(entries) - limit)

    copied = 0
    highgap_copied = 0
    lowgap_copied = 0
    taxa_counts = {}
    validation_rows = []

    for item in selected:
        folder_name = f'{item["top1"]}{item["folder_suffix"]}'
        dest_dir = tar_export_dir / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_flattened_destination(dest_dir, item["src"].name)
        shutil.copy2(item["src"], dest)

        copied += 1
        taxa_counts[item["top1"]] = taxa_counts.get(item["top1"], 0) + 1
        if item["bucket"] == "highgap":
            highgap_copied += 1
        else:
            lowgap_copied += 1

        validation_rows.append({
            "image": dest.name,
            "label": item["top1"],
            "original_label": item["original_top1"],
            "translated_label": item["translated_top1"],
            "user": VALIDATION_USER,
            "subset": item["bucket"],
            "folder": folder_name,
            "top1_prob": item["top1_prob"],
            "top2_prob": item["top2_prob"],
            "margin_top1_top2": item["margin"],
            "top2_label": item["top2"],
            "original_top2_label": item["original_top2"],
            "translated_top2_label": item["translated_top2"],
            "same_translated_taxon_t1_t2": item["same_translated_taxon"],
            "same_taxon_high_confidence": item["same_taxon_high_confidence"],
        })

    validation_csv = None
    if validation_rows:
        validation_csv = tar_export_dir / f"{tar_name}_validated_cleaned.csv"
        pd.DataFrame(validation_rows).to_csv(validation_csv, index=False)

    print(f"       ✅ Copied {copied} images into {len(taxa_counts)} taxa")
    print(
        f"       ✅ High-gap: {highgap_copied} copied (limit {high_gap_limit}/taxon, "
        f"margin ≥ {high_gap_margin:.2f})"
    )
    print(f"       ✅ Low-gap: {lowgap_copied} copied (limit {low_gap_limit}/taxon)")
    if validation_csv is not None:
        print(f"       ✅ Wrote validation CSV: {validation_csv.name} ({len(validation_rows)} rows)")
    if missing:
        print(f"       ⚠️ Missing source files during taxon export: {missing}")
    if skipped_limit:
        print(
            "       ⚠️ Skipped "
            f"{skipped_limit} images due to per-taxon bucket caps "
            f"({high_gap_limit} high-gap / {low_gap_limit} low-gap)"
        )

    return {
        "copied": copied,
        "missing": missing,
        "taxa": len(taxa_counts),
        "highgap_copied": highgap_copied,
        "lowgap_copied": lowgap_copied,
        "skipped_limit": skipped_limit,
        "validation_csv": str(validation_csv) if validation_csv is not None else None,
    }


def export_images_by_top1_taxon_from_tar(
    tar_path,
    member_info,
    json_path,
    tar_name,
    export_root,
    high_gap_margin=PREVIEW_HIGH_GAP_MARGIN,
    high_gap_limit=PREVIEW_HIGH_GAP_LIMIT,
    low_gap_limit=PREVIEW_LOW_GAP_LIMIT,
):
    """Copy capped validation images by taxon directly from TAR bytes."""
    print("⚙ Exporting classified images by taxon directly from TAR...")

    json_path = Path(json_path)
    export_root = Path(export_root)
    member_info = member_info or {}

    empty = {
        "copied": 0,
        "missing": 0,
        "taxa": 0,
        "highgap_copied": 0,
        "lowgap_copied": 0,
        "skipped_limit": 0,
        "validation_csv": None,
    }

    if not json_path.exists():
        print(f"       ⚠️ JSON file not found: {json_path}")
        return empty

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON for taxon export: {e}")
        return empty

    data = [entry for entry in data if isinstance(entry, dict)]
    label_translation = load_label_translation(str(CLASS_TRANSLATION_CSV))

    tar_export_dir = export_root / tar_name
    if tar_export_dir.exists():
        shutil.rmtree(tar_export_dir, onerror=_retry_remove_readonly)
    tar_export_dir.mkdir(parents=True, exist_ok=True)

    prepared_entries = []
    missing = 0

    for entry in data:
        rel_path = entry.get("filepath", "")
        labels, probs = parse_prediction_lists(entry)
        if not labels:
            continue

        if rel_path not in member_info:
            missing += 1
            continue

        original_top1 = str(labels[0]).strip()
        original_top2 = str(labels[1]).strip() if len(labels) > 1 else None
        translated_top1 = translate_label(original_top1, label_translation)
        translated_top2 = translate_label(original_top2, label_translation) if original_top2 else None

        top1 = sanitize_taxon_name(translated_top1)
        top2 = sanitize_taxon_name(translated_top2) if translated_top2 else None
        top1_prob = probs[0] if len(probs) > 0 else None
        top2_prob = probs[1] if len(probs) > 1 else 0.0

        same_translated_taxon = (
            translated_top2 is not None
            and sanitize_taxon_name(translated_top1) == sanitize_taxon_name(translated_top2)
        )
        same_taxon_high_confidence = (
            same_translated_taxon
            and top1_prob is not None
            and top2_prob is not None
            and float(top1_prob) >= 0.5
            and float(top2_prob) >= 0.1
        )

        if top1_prob is None:
            margin = None
        elif same_taxon_high_confidence:
            margin = float(top1_prob)
        elif top2_prob is not None:
            margin = float(top1_prob) - float(top2_prob)
        else:
            margin = float(top1_prob)

        if margin is not None and margin >= high_gap_margin:
            bucket = "highgap"
            folder_suffix = PREVIEW_HIGH_GAP_SUFFIX
            bucket_limit = high_gap_limit
        else:
            bucket = "lowgap"
            folder_suffix = PREVIEW_LOW_GAP_SUFFIX
            bucket_limit = low_gap_limit

        prepared_entries.append({
            "member_name": rel_path,
            "src_name": _tar_basename(rel_path),
            "top1": top1,
            "original_top1": original_top1,
            "translated_top1": translated_top1,
            "original_top2": original_top2,
            "translated_top2": translated_top2,
            "same_translated_taxon": same_translated_taxon,
            "top2": top2,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "margin": margin,
            "bucket": bucket,
            "folder_suffix": folder_suffix,
            "bucket_limit": bucket_limit,
            "same_taxon_high_confidence": same_taxon_high_confidence,
        })

    grouped = {}
    for item in prepared_entries:
        grouped.setdefault((item["top1"], item["bucket"]), []).append(item)

    selected = []
    skipped_limit = 0
    for (taxon, bucket), entries in grouped.items():
        rng = random.Random(f"{tar_name}|{taxon}|{bucket}")
        rng.shuffle(entries)
        limit = entries[0]["bucket_limit"] if entries else 0
        selected.extend(entries[:limit])
        skipped_limit += max(0, len(entries) - limit)

    copied = 0
    highgap_copied = 0
    lowgap_copied = 0
    taxa_counts = {}
    validation_rows = []

    for item in selected:
        folder_name = f'{item["top1"]}{item["folder_suffix"]}'
        dest_dir = tar_export_dir / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_flattened_destination(dest_dir, item["src_name"])

        try:
            write_tar_member_to_file(tar_path, item["member_name"], dest, member_info=member_info)
        except Exception as e:
            missing += 1
            print(f"       ⚠️ Could not copy {item['member_name']} from TAR: {e}")
            continue

        copied += 1
        taxa_counts[item["top1"]] = taxa_counts.get(item["top1"], 0) + 1
        if item["bucket"] == "highgap":
            highgap_copied += 1
        else:
            lowgap_copied += 1

        validation_rows.append({
            "image": dest.name,
            "label": item["top1"],
            "original_label": item["original_top1"],
            "translated_label": item["translated_top1"],
            "user": VALIDATION_USER,
            "subset": item["bucket"],
            "folder": folder_name,
            "top1_prob": item["top1_prob"],
            "top2_prob": item["top2_prob"],
            "margin_top1_top2": item["margin"],
            "top2_label": item["top2"],
            "original_top2_label": item["original_top2"],
            "translated_top2_label": item["translated_top2"],
            "same_translated_taxon_t1_t2": item["same_translated_taxon"],
            "same_taxon_high_confidence": item["same_taxon_high_confidence"],
        })

    validation_csv = None
    if validation_rows:
        validation_csv = tar_export_dir / f"{tar_name}_validated_cleaned.csv"
        pd.DataFrame(validation_rows).to_csv(validation_csv, index=False)

    print(f"       ✅ Copied {copied} images into {len(taxa_counts)} taxa")
    print(
        f"       ✅ High-gap: {highgap_copied} copied (limit {high_gap_limit}/taxon, "
        f"margin ≥ {high_gap_margin:.2f})"
    )
    print(f"       ✅ Low-gap: {lowgap_copied} copied (limit {low_gap_limit}/taxon)")
    if validation_csv is not None:
        print(f"       ✅ Wrote validation CSV: {validation_csv.name} ({len(validation_rows)} rows)")
    if missing:
        print(f"       ⚠️ Missing/could-not-copy source files during taxon export: {missing}")
    if skipped_limit:
        print(
            "       ⚠️ Skipped "
            f"{skipped_limit} images due to per-taxon bucket caps "
            f"({high_gap_limit} high-gap / {low_gap_limit} low-gap)"
        )

    return {
        "copied": copied,
        "missing": missing,
        "taxa": len(taxa_counts),
        "highgap_copied": highgap_copied,
        "lowgap_copied": lowgap_copied,
        "skipped_limit": skipped_limit,
        "validation_csv": str(validation_csv) if validation_csv is not None else None,
    }


from time import time as timer
import gc
import stat

def _retry_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def clear_untarred_dir(dir_path, retries=8, delay=1.0):
    start_time = timer()
    dir_path = Path(dir_path)
    print(f"⚙ Clear and created local directories: {dir_path}")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            gc.collect()

            if dir_path.exists():
                shutil.rmtree(dir_path, onerror=_retry_remove_readonly)

            dir_path.mkdir(parents=True, exist_ok=True)

            elapsed_time = timer() - start_time
            print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
            return elapsed_time

        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"       ⚠️ Cleanup retry {attempt}/{retries - 1} for {dir_path}: {e}")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to clear directory {dir_path}: {e}") from e

def clear_temp_output_dir(output_dir):
    """
    Clear temporary output files created in the local work_dir/output folder.

    These are only intermediate copies. Final outputs are saved to source_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    removed_files = 0
    removed_dirs = 0

    for item in output_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                removed_files += 1
            elif item.is_dir():
                shutil.rmtree(item, onerror=_retry_remove_readonly)
                removed_dirs += 1
        except Exception as e:
            print(f"       ⚠️ Could not remove temp output item {item}: {e}")

    print(
        f"       ✅ Cleared temp output: {removed_files} file(s), "
        f"{removed_dirs} folder(s)."
    )


def remove_empty_dirs(root_dir):
    """
    Remove empty leftover folders below root_dir.
    """
    root_dir = Path(root_dir)

    if not root_dir.exists():
        return

    removed = 0

    # Walk deepest folders first.
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        path = Path(dirpath)

        # Do not remove the root itself.
        if path == root_dir:
            continue

        try:
            if not any(path.iterdir()):
                path.rmdir()
                removed += 1
        except Exception as e:
            print(f"       ⚠️ Could not remove empty folder {path}: {e}")

    if removed:
        print(f"       ✅ Removed {removed} empty folder(s) from {root_dir}")



def extract_tar(tar_path, extract_to):
    start_time = timer()  # Start timing
    print(f"⚙ Untarring {tar_path.name}...")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)  # Extract the TAR file

    elapsed_time = timer() - start_time  # Calculate elapsed time
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return elapsed_time  # Return the time taken




def _safe_member_target(base_dir, member_name):
    """Return a safe target path for a TAR member, or raise on path traversal."""
    base_dir = Path(base_dir).resolve()
    # TAR member names are POSIX-style; strip leading slashes and normalize.
    clean_name = posixpath.normpath(str(member_name).lstrip("/"))
    if clean_name in {"", "."} or clean_name.startswith("../"):
        raise ValueError(f"Unsafe TAR member path: {member_name}")
    target = (base_dir / clean_name).resolve()
    if os.path.commonpath([str(base_dir), str(target)]) != str(base_dir):
        raise ValueError(f"Unsafe TAR member path: {member_name}")
    return target


def _delete_tree_worker(path):
    try:
        shutil.rmtree(path, onerror=_retry_remove_readonly)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"       ⚠️ Background cleanup failed for {path}: {e}")


def fast_clear_dir(dir_path, async_cleanup=None):
    """Create an empty directory quickly by renaming old content away.

    If async_cleanup is enabled, the expensive deletion happens in a daemon
    thread while the next TAR can start. This removes the 5-10 minute wait that
    happens when deleting ~100k extracted TIFFs synchronously.
    """
    start_time = timer()
    dir_path = Path(dir_path)
    if async_cleanup is None:
        async_cleanup = ASYNC_CLEANUP

    if dir_path.exists():
        trash = dir_path.with_name(f"{dir_path.name}.__delete__.{uuid.uuid4().hex}")
        try:
            dir_path.rename(trash)
            if async_cleanup:
                threading.Thread(target=_delete_tree_worker, args=(trash,), daemon=True).start()
            else:
                _delete_tree_worker(trash)
        except Exception as e:
            print(f"       ⚠️ Fast rename cleanup failed for {dir_path}: {e}; using blocking rmtree")
            shutil.rmtree(dir_path, onerror=_retry_remove_readonly)

    dir_path.mkdir(parents=True, exist_ok=True)
    elapsed = timer() - start_time
    print(f"       ✅ Prepared empty directory in {elapsed:.2f} seconds: {dir_path}")
    return elapsed


def schedule_delete_dir(dir_path, async_cleanup=None):
    """Rename a directory away and delete it without recreating it."""
    start_time = timer()
    dir_path = Path(dir_path)
    if async_cleanup is None:
        async_cleanup = ASYNC_CLEANUP
    if not dir_path.exists():
        return 0.0
    trash = dir_path.with_name(f"{dir_path.name}.__delete__.{uuid.uuid4().hex}")
    try:
        dir_path.rename(trash)
        if async_cleanup:
            threading.Thread(target=_delete_tree_worker, args=(trash,), daemon=True).start()
        else:
            _delete_tree_worker(trash)
    except Exception as e:
        print(f"       ⚠️ Rename delete failed for {dir_path}: {e}; using blocking rmtree")
        shutil.rmtree(dir_path, onerror=_retry_remove_readonly)
    elapsed = timer() - start_time
    print(f"       ✅ Cleanup scheduled in {elapsed:.2f} seconds: {dir_path}")
    return elapsed


def safe_extractall_python(tar_path, extract_to):
    extract_to = Path(extract_to)
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            _safe_member_target(extract_to, member.name)
        tar.extractall(path=extract_to)


def extract_tar_fast(tar_path, extract_to):
    """Extract a TAR using native tar when available, with safe Python fallback."""
    start_time = timer()
    tar_path = Path(tar_path)
    extract_to = Path(extract_to)
    print(f"⚙ Fast untarring {tar_path.name}...")
    fast_clear_dir(extract_to, async_cleanup=False)

    native_tar = shutil.which("tar")
    used_native = False
    if FAST_NATIVE_TAR and TRUSTED_TAR_INPUTS and native_tar:
        try:
            result = subprocess.run(
                [native_tar, "-xf", str(tar_path), "-C", str(extract_to)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                used_native = True
            else:
                print(f"       ⚠️ Native tar failed; falling back to Python tarfile: {result.stderr.strip()[:500]}")
                fast_clear_dir(extract_to, async_cleanup=False)
        except Exception as e:
            print(f"       ⚠️ Native tar unavailable/failed; falling back to Python tarfile: {e}")
            fast_clear_dir(extract_to, async_cleanup=False)

    if not used_native:
        safe_extractall_python(tar_path, extract_to)

    elapsed_time = timer() - start_time
    method = "native tar" if used_native else "python tarfile"
    print(f"       ✅ Done in {elapsed_time:.2f} seconds using {method}.")
    return elapsed_time


def _tar_basename(member_name):
    return posixpath.basename(str(member_name).replace("\\", "/"))


def _is_tif_member_name(name):
    return str(name).lower().endswith((".tif", ".tiff"))


def _is_ignored_tif_name(name):
    return _tar_basename(name) in IGNORED_TIF_NAMES


def tar_inventory(tar_path):
    """List useful members once without extracting the TAR.

    The member_info map stores byte offsets and sizes for uncompressed .tar
    files. That lets the TAR-stream predictor read each TIFF with one seek/read,
    without tarfile.getmember() lookups and without writing files to disk.
    """
    start = timer()
    tar_path = Path(tar_path)
    print(f"⚙ Reading TAR inventory for {tar_path.name}...")
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

    names = [m.name for m in members]
    member_info = {
        m.name: {
            "offset_data": getattr(m, "offset_data", None),
            "size": getattr(m, "size", None),
        }
        for m in members
    }

    tif_all = [n for n in names if _is_tif_member_name(n)]
    tif_images = [n for n in tif_all if not _is_ignored_tif_name(n)]
    background = next((n for n in names if _tar_basename(n) == "Background.tif"), None)
    flow_edges = next((n for n in names if _tar_basename(n) == "FlowCellEdges.tif"), None)
    hitsmisses = next((n for n in names if "hitsmisses.txt" in _tar_basename(n).lower()), None)

    elapsed = timer() - start
    print(
        f"       ✅ Inventory in {elapsed:.2f}s: {len(tif_all)} TIFF(s), "
        f"{len(tif_images)} classifiable image(s)"
    )
    return {
        "all_file_names": names,
        "member_info": member_info,
        "tif_all": tif_all,
        "tif_images": tif_images,
        "background": background,
        "flow_edges": flow_edges,
        "hitsmisses": hitsmisses,
    }, elapsed


# === OPTION A: TAR-stream filemode for planktonclass.predict ===
class TarImageRef:
    """Reference to one image inside an uncompressed TAR archive.

    planktonclass.predict() accepts a list of arbitrary inputs and forwards each
    item to planktonclass.data_utils.load_image(input, filemode=...). By patching
    load_image to understand filemode="tar", we can pass TarImageRef instances
    instead of local file paths.
    """
    __slots__ = ("tar_path", "member_name", "offset_data", "size")

    def __init__(self, tar_path, member_name, offset_data=None, size=None):
        self.tar_path = str(tar_path)
        self.member_name = str(member_name)
        self.offset_data = None if offset_data is None else int(offset_data)
        self.size = None if size is None else int(size)

    @property
    def name(self):
        return self.member_name

    def __repr__(self):
        return f"TarImageRef({Path(self.tar_path).name}!{self.member_name})"


_TAR_STREAM_THREAD_LOCAL = threading.local()
_ORIGINAL_PLANKTON_LOAD_IMAGE = None


def _get_thread_file_handle(path):
    path = str(Path(path))
    cache = getattr(_TAR_STREAM_THREAD_LOCAL, "file_handles", None)
    if cache is None:
        cache = {}
        _TAR_STREAM_THREAD_LOCAL.file_handles = cache
    handle = cache.get(path)
    if handle is None or handle.closed:
        handle = open(path, "rb")
        cache[path] = handle
    return handle


def close_tar_stream_file_cache():
    cache = getattr(_TAR_STREAM_THREAD_LOCAL, "file_handles", None)
    if not cache:
        return
    for handle in list(cache.values()):
        try:
            handle.close()
        except Exception:
            pass
    cache.clear()


def _tar_ref_from_member_name(tar_path, member_name, member_info=None):
    info = (member_info or {}).get(member_name, {})
    return TarImageRef(
        tar_path=tar_path,
        member_name=member_name,
        offset_data=info.get("offset_data"),
        size=info.get("size"),
    )


def tar_image_refs(tar_path, member_names, member_info=None):
    return [
        _tar_ref_from_member_name(tar_path, member_name, member_info=member_info)
        for member_name in list(member_names or [])
    ]


def _read_tar_member_bytes(ref_or_tuple):
    """Read one TAR member as bytes without extracting it.

    Fast path: uncompressed .tar with offset_data/size from tar_inventory().
    Fallback: tarfile.extractfile(member_name), useful if offsets are missing.
    """
    if isinstance(ref_or_tuple, TarImageRef):
        ref = ref_or_tuple
    elif isinstance(ref_or_tuple, (tuple, list)) and len(ref_or_tuple) >= 2:
        # Allows ad-hoc inputs: (tar_path, member_name, offset_data, size)
        ref = TarImageRef(
            ref_or_tuple[0],
            ref_or_tuple[1],
            ref_or_tuple[2] if len(ref_or_tuple) > 2 else None,
            ref_or_tuple[3] if len(ref_or_tuple) > 3 else None,
        )
    else:
        raise ValueError(
            "filemode='tar' expects TarImageRef or "
            "(tar_path, member_name, offset_data, size)"
        )

    if ref.offset_data is not None and ref.size is not None:
        handle = _get_thread_file_handle(ref.tar_path)
        handle.seek(ref.offset_data)
        return handle.read(ref.size)

    with tarfile.open(ref.tar_path, "r:*") as tar:
        src = tar.extractfile(ref.member_name)
        if src is None:
            raise FileNotFoundError(f"Could not open TAR member: {ref.member_name}")
        return src.read()


def load_image_from_tar_ref(ref):
    """Load TAR-contained image exactly like planktonclass.data_utils.load_image.

    planktonclass uses cv2.imread(..., cv2.IMREAD_COLOR) and converts BGR to RGB.
    Here we use cv2.imdecode(..., cv2.IMREAD_COLOR) on the member bytes and then
    the same BGR->RGB conversion, so the downstream augmentation/preprocessing
    path remains the normal planktonclass path.
    """
    import cv2

    data = _read_tar_member_bytes(ref)
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image from TAR member: {ref}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def enable_planktonclass_tar_filemode():
    """Patch planktonclass.data_utils.load_image to support filemode='tar'."""
    global _ORIGINAL_PLANKTON_LOAD_IMAGE
    import planktonclass.data_utils as plk_data_utils

    if getattr(plk_data_utils.load_image, "_pi10_tar_stream_enabled", False):
        return

    _ORIGINAL_PLANKTON_LOAD_IMAGE = plk_data_utils.load_image

    def load_image_patched(filename, filemode="local"):
        if filemode == "tar":
            return load_image_from_tar_ref(filename)
        return _ORIGINAL_PLANKTON_LOAD_IMAGE(filename, filemode=filemode)

    load_image_patched._pi10_tar_stream_enabled = True
    load_image_patched._pi10_original_load_image = _ORIGINAL_PLANKTON_LOAD_IMAGE
    plk_data_utils.load_image = load_image_patched


def predict_tar_members(model_obj, tar_path, member_names, conf_obj, member_info=None,
                        top_K=None, crop_num=10, merge=False):
    """Run the normal planktonclass.predict() on images inside a TAR.

    This is Option A: planktonclass.predict() is still used, but it now receives
    TarImageRef objects and filemode='tar' instead of local image paths.
    """
    enable_planktonclass_tar_filemode()
    refs = tar_image_refs(tar_path, member_names, member_info=member_info)
    try:
        return predict(
            model_obj,
            refs,
            conf_obj,
            top_K=top_K,
            crop_num=crop_num,
            filemode="tar",
            merge=merge,
        )
    finally:
        close_tar_stream_file_cache()


def read_tiff_array_from_tar(tar_path, member_name, member_info=None):
    ref = _tar_ref_from_member_name(tar_path, member_name, member_info=member_info)
    data = _read_tar_member_bytes(ref)
    return tiff.imread(io.BytesIO(data))


def write_tar_member_to_file(tar_path, member_name, dest_path, member_info=None):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ref = _tar_ref_from_member_name(tar_path, member_name, member_info=member_info)
    data = _read_tar_member_bytes(ref)
    with open(dest_path, "wb") as f:
        f.write(data)


def ensure_full_extract_for_fallback(working_tar, extract_dir, times_dict, status_log, reason):
    """Extract only when a fallback path still needs real files."""
    if Path(extract_dir).exists() and any(Path(extract_dir).iterdir()):
        return
    print(f"⚠️ Full extraction fallback required: {reason}")
    elapsed = extract_tar_fast(working_tar, extract_dir)
    times_dict["Untar"] = times_dict.get("Untar", 0.0) + elapsed
    status_log.append(f"Full TAR extraction fallback completed: {reason}")


def _edge_mean_gray_array(img, edge_fraction=GRAY_EDGE_FRACTION):
    gray = to_gray(img)
    h, w = gray.shape
    bw = max(1, int(min(h, w) * edge_fraction))
    mask = np.zeros_like(gray, dtype=bool)
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return float(gray[mask].mean())


def check_gray_edge_quarantine_from_tar(
    tar_path,
    image_member_names,
    tar_name,
    n_images=GRAY_EDGE_SAMPLE_N_IMAGES,
    edge_fraction=GRAY_EDGE_FRACTION,
    min_edge_mean=GRAY_EDGE_MEAN_MIN,
):
    print("⚙ Running gray-edge quarantine check directly from TAR sample...")
    image_member_names = list(image_member_names or [])
    if not image_member_names:
        print("       ⚠️ No valid TIFF files found for gray-edge check; continuing.")
        return True, None, None

    sample = random.sample(image_member_names, min(n_images, len(image_member_names)))
    values = []
    with tarfile.open(tar_path) as tar:
        for member_name in sample:
            try:
                f = tar.extractfile(member_name)
                if f is None:
                    continue
                data = f.read()
                img = tiff.imread(io.BytesIO(data))
                values.append(_edge_mean_gray_array(img, edge_fraction=edge_fraction))
            except Exception as e:
                print(f"       ⚠️ Could not read {_tar_basename(member_name)} for gray-edge check: {e}")

    if not values:
        print("       ⚠️ Gray-edge check had no readable TIFF files; continuing.")
        return True, None, None

    edge_mean = sum(values) / len(values)
    print(
        f"       ✅ Edge mean grayscale: {edge_mean:.2f} "
        f"from {len(values)} image(s); threshold: {min_edge_mean}"
    )
    if min_edge_mean is not None and edge_mean < min_edge_mean:
        reason = f"edge mean grayscale {edge_mean:.2f} < {min_edge_mean}"
        return False, reason, edge_mean
    return True, None, edge_mean


def extract_tar_members_to_dir(tar_path, member_names, dest_dir):
    """Extract selected TAR members for preview classification only."""
    dest_dir = Path(dest_dir)
    fast_clear_dir(dest_dir, async_cleanup=False)
    extracted_paths = []
    with tarfile.open(tar_path) as tar:
        for member_name in member_names:
            try:
                target = _safe_member_target(dest_dir, member_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member_name)
                if src is None:
                    continue
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                extracted_paths.append(target)
            except Exception as e:
                print(f"       ⚠️ Could not extract preview member {member_name}: {e}")
    return extracted_paths


def get_preview_sample_from_inventory(image_member_names, n=PREVIEW_SAMPLE_N):
    image_member_names = list(image_member_names or [])
    if not image_member_names:
        return []
    if len(image_member_names) <= n:
        return image_member_names
    return random.sample(image_member_names, n)


def copy_background_from_tar(tar_path, background_member_name, dest_path):
    start = timer()
    print("⚙ Copying Background.tif directly from TAR...")
    dest_path = Path(dest_path)
    if not background_member_name:
        print("       ⚠️ Background.tif not found")
        return timer() - start
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        src = tar.extractfile(background_member_name)
        if src is None:
            print("       ⚠️ Background.tif could not be opened")
            return timer() - start
        with open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    elapsed = timer() - start
    print(f"       ✅ Done in {elapsed:.2f} seconds.")
    return elapsed


def extract_hitsmisses_from_tar(tar_path, hits_member_name, output_file, tar_file, status_log):
    start = timer()
    print("⚙ Fetching hits and misses directly from TAR...")
    MAX_MISS_HIT_RATIO = 50

    if not hits_member_name:
        status_log.append("hitsmisses.txt not found in TAR")
        print(f"       ⚠️ hitsmisses.txt not found in {tar_file.name}")
        return False, "missing_hitsmisses"

    with tarfile.open(tar_path) as tar:
        f = tar.extractfile(hits_member_name)
        if f is None:
            status_log.append("hitsmisses.txt could not be opened in TAR")
            return False, "missing_hitsmisses"
        df = pd.read_csv(f, header=None)

    df.columns = ["hits", "misses"]
    df["minute"] = range(len(df))
    df["tar_source"] = Path(tar_file).stem
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    df["RaisingFactor"] = df["hits"] / (df["hits"] + df["misses"])

    if len(df) != 10:
        status_log.append(f"hitsmisses had only {len(df)} rows")
        print(f"       🚨 hitsmisses had only {len(df)} rows; will quarantine")
        if output_file.exists():
            output_file.unlink()
        return False, "bad_hitsmisses_row_count"

    total_hits = df["hits"].sum()
    total_misses = df["misses"].sum()
    if total_hits > 0 and total_misses > total_hits * MAX_MISS_HIT_RATIO:
        status_log.append(
            f"hitsmisses misses too high: {total_misses} > {MAX_MISS_HIT_RATIO}x {total_hits}"
        )
        print(
            f"       🚨 hitsmisses misses too high "
            f"({total_misses} > {MAX_MISS_HIT_RATIO}x {total_hits}); will quarantine"
        )
        if output_file.exists():
            output_file.unlink()
        return False, "too_many_misses"

    elapsed = timer() - start
    print(f"       ✅ Done in {elapsed:.2f} seconds.")
    return True, None


def should_copy_tar_to_local(tar_file):
    """Auto-copy only when it is likely useful, e.g. Windows Y: -> D:."""
    if COPY_TAR_TO_LOCAL in {"1", "true", "yes", "y", "on"}:
        return True
    if COPY_TAR_TO_LOCAL in {"0", "false", "no", "n", "off"}:
        return False
    try:
        src_drive = Path(tar_file).drive.lower()
        work_drive = Path(work_dir).drive.lower()
        return bool(src_drive and work_drive and src_drive != work_drive)
    except Exception:
        return False


def prepare_working_tar(tar_file, tar_dest):
    """Return the TAR path to read from and whether it is a temporary copy."""
    tar_file = Path(tar_file)
    tar_dest = Path(tar_dest)
    if should_copy_tar_to_local(tar_file):
        start = timer()
        tar_dest.parent.mkdir(parents=True, exist_ok=True)
        if tar_dest.exists():
            tar_dest.unlink()
        print(f"⚙ Copying TAR to local work disk: {tar_dest}")
        shutil.copy2(tar_file, tar_dest)
        elapsed = timer() - start
        print(f"       ✅ Done in {elapsed:.2f} seconds.")
        return tar_dest, True, elapsed
    return tar_file, False, 0.0


def move_tar_to_quarantine(original_tar, quarantine_target, status_log, reason):
    try:
        quarantine_target = Path(quarantine_target)
        quarantine_target.parent.mkdir(parents=True, exist_ok=True)
        if quarantine_target.exists():
            quarantine_target.unlink()
        shutil.move(str(original_tar), str(quarantine_target))
        status_log.append(f"Moved TAR to quarantine: {quarantine_target} ({reason})")
        return True
    except Exception as mv_err:
        status_log.append(f"⚠️ Quarantine move failed: {mv_err}")
        return False


def to_gray(img):
    img = np.asarray(img)
    original_dtype = img.dtype

    if img.ndim == 3:
        rgb = img[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        gray = img.astype(np.float32)

    if np.issubdtype(original_dtype, np.integer):
        gray = gray / np.iinfo(original_dtype).max * 255.0

    return gray


def edge_mean_gray(path, edge_fraction=GRAY_EDGE_FRACTION):
    img = tiff.imread(path)
    gray = to_gray(img)

    h, w = gray.shape
    bw = max(1, int(min(h, w) * edge_fraction))

    mask = np.zeros_like(gray, dtype=bool)
    mask[:bw, :] = True      # top
    mask[-bw:, :] = True     # bottom
    mask[:, :bw] = True      # left
    mask[:, -bw:] = True     # right

    return float(gray[mask].mean())



def check_gray_edge_quarantine(
    extract_dir,
    tar_name,
    n_images=GRAY_EDGE_SAMPLE_N_IMAGES,
    edge_fraction=GRAY_EDGE_FRACTION,
    min_edge_mean=GRAY_EDGE_MEAN_MIN,
):
    print("⚙ Running gray-edge quarantine check...")

    tif_files = [
        p for p in Path(extract_dir).rglob("*.tif")
        if p.name not in IGNORED_TIF_NAMES
    ]

    if not tif_files:
        print("       ⚠️ No valid TIFF files found for gray-edge check; continuing.")
        return True, None, None

    sample = random.sample(tif_files, min(n_images, len(tif_files)))
    values = []
    failed_images = 0

    for tif_path in sample:
        try:
            values.append(edge_mean_gray(tif_path, edge_fraction=edge_fraction))
        except Exception as e:
            failed_images += 1
            print(f"       ⚠️ Could not read {tif_path.name} for gray-edge check: {e}")

    if not values:
        print("       ⚠️ Gray-edge check had no readable TIFF files; continuing.")
        return True, None, None

    edge_mean = sum(values) / len(values)

    print(
        f"       ✅ Edge mean grayscale: {edge_mean:.2f} "
        f"from {len(values)} image(s); minimum threshold: {min_edge_mean}"
    )

    if min_edge_mean is not None and edge_mean < min_edge_mean:
        reason = f"edge mean grayscale {edge_mean:.2f} < {min_edge_mean}"
        return False, reason, edge_mean

    return True, None, edge_mean


def count_images_in_tar(extract_dir, tar_file):
    """Count the number of .tif images in the extracted directory."""
    print(f"⚙ Counting images in {tar_file.name}...")
    tif_files = list(extract_dir.rglob("*.tif"))
    print(f"       ✅ Found {len(tif_files)} .tif files")
    return len(tif_files)

def copy_background_tif(extract_dir, dest_path):
    start = timer()
    print("⚙ Copying Background..")

    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == "Background.tif":
                full_path = os.path.join(root, f)
                if not dest_path.exists():
                    shutil.copy(full_path, dest_path)
                elapsed_time = timer() - start
                print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
                return  # exit after success

    # If loop finishes without finding the file
    elapsed_time = timer() - start
    print(f"       ⚠️ Background.tif not found")



def extract_hitsmisses(tar_path, output_file, tar_file, status_log):
    start = timer()
    print("⚙ Fetching hits and misses...")

    MAX_MISS_HIT_RATIO = 50

    with tarfile.open(tar_path) as tar:
        hits_file = next(
            (m for m in tar.getmembers() if "hitsmisses.txt" in m.name.lower()),
            None
        )

        if not hits_file:
            status_log.append("hitsmisses.txt not found in TAR")
            print(f"       ⚠️ hitsmisses.txt not found in {tar_file.name}")
            return False, "missing_hitsmisses"

        f = tar.extractfile(hits_file)
        df = pd.read_csv(f, header=None)
        df.columns = ["hits", "misses"]
        df["minute"] = range(len(df))
        df["tar_source"] = tar_path.stem

        # Write only after successful extraction, but still remove it if invalid.
        df.to_csv(output_file, index=False)

        df["RaisingFactor"] = df["hits"] / (df["hits"] + df["misses"])

        if len(df) != 10:
            status_log.append(f"hitsmisses had only {len(df)} rows")
            print(f"       🚨 hitsmisses had only {len(df)} rows; will quarantine")

            try:
                if output_file.exists():
                    output_file.unlink()
                    status_log.append("Removed hitsmisses.txt due to quarantine")
            except Exception as e:
                status_log.append(f"⚠️ Failed to remove hitsmisses.txt: {e}")

            return False, "bad_hitsmisses_row_count"

        total_hits = df["hits"].sum()
        total_misses = df["misses"].sum()

        if total_hits > 0 and total_misses > total_hits * MAX_MISS_HIT_RATIO:
            status_log.append(
                f"hitsmisses misses too high: "
                f"{total_misses} > {MAX_MISS_HIT_RATIO}x {total_hits}"
            )
            print(
                f"       🚨 hitsmisses misses too high "
                f"({total_misses} > {MAX_MISS_HIT_RATIO}x {total_hits}); will quarantine"
            )

            try:
                if output_file.exists():
                    output_file.unlink()
                    status_log.append("Removed hitsmisses.txt due to quarantine")
            except Exception as e:
                status_log.append(f"⚠️ Failed to remove hitsmisses.txt: {e}")

            return False, "too_many_misses"

    elapsed_time = timer() - start
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return True, None


def get_preview_sample_tifs(extract_dir, n=200):
    tif_files = [
        p for p in extract_dir.rglob("*.tif")
        if p.name not in {"Background.tif", "FlowCellEdges.tif"}
    ]

    if not tif_files:
        print(f"       ⚠️ No valid preview TIFFs found in {extract_dir}")
        return []

    if len(tif_files) <= n:
        return tif_files

    return random.sample(tif_files, n)


import os
import time
import json
import subprocess
import pandas as pd

def parse_exif_datetime_series(series):
    parsed = pd.to_datetime(series, format="%Y:%m:%d %H:%M:%S", errors="coerce")

    mask = parsed.isna() & series.notna()
    if mask.any():
        parsed_tz = pd.to_datetime(
            series.loc[mask],
            format="%Y:%m:%d %H:%M:%S%z",
            errors="coerce",
            utc=True
        )
        if getattr(parsed_tz.dt, "tz", None) is not None:
            parsed_tz = parsed_tz.dt.tz_localize(None)
        parsed.loc[mask] = parsed_tz

    return parsed


def _ratio_to_float(value):
    if value is None:
        return None
    try:
        # Pillow IFDRational
        return float(value)
    except Exception:
        pass
    try:
        return float(value[0]) / float(value[1])
    except Exception:
        return None


def _gps_dms_to_decimal(dms, ref):
    if not dms or len(dms) < 3:
        return None
    parts = [_ratio_to_float(x) for x in dms[:3]]
    if any(x is None for x in parts):
        return None
    deg, minutes, seconds = parts
    value = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref or "").upper() in {"S", "W"}:
        value *= -1
    return value


def extract_exif_metadata_from_tar_python(tar_path, image_member_names, member_info, tar_source):
    """Best-effort in-memory EXIF extraction from TIFFs inside a TAR.

    This avoids full extraction when the TIFF stores standard EXIF/GPS tags.
    If it returns no usable GPS, process_tar() can fall back to the existing
    ExifTool path before making a quarantine decision.
    """
    print("⚙ Extracting EXIF metadata directly from TAR with Pillow/tifffile...")
    start = timer()
    rows = []

    try:
        from PIL import Image
    except Exception as e:
        print(f"       ⚠️ Pillow is not available for TAR EXIF extraction: {e}")
        return pd.DataFrame()

    for member_name in tqdm(list(image_member_names or []), desc="TAR EXIF", leave=False):
        try:
            ref = _tar_ref_from_member_name(tar_path, member_name, member_info=member_info)
            data = _read_tar_member_bytes(ref)
            row = {
                "SourceFile": f"{Path(tar_path).name}::{member_name}",
                "tif_name": _tar_basename(member_name),
                "tar_source": tar_source,
            }

            with Image.open(io.BytesIO(data)) as img:
                exif = img.getexif()
                if exif:
                    # Standard TIFF/EXIF date fields.
                    row["ModifyDate"] = exif.get(306)
                    row["DateTimeOriginal"] = exif.get(36867)
                    row["CreateDate"] = exif.get(36868)

                    gps_ifd = None
                    try:
                        gps_ifd = exif.get_ifd(34853)  # GPSInfo IFD
                    except Exception:
                        gps_ifd = exif.get(34853)

                    if gps_ifd:
                        lat = _gps_dms_to_decimal(gps_ifd.get(2), gps_ifd.get(1))
                        lon = _gps_dms_to_decimal(gps_ifd.get(4), gps_ifd.get(3))
                        if lat is not None:
                            row["GPSLatitude"] = lat
                        if lon is not None:
                            row["GPSLongitude"] = lon

            rows.append(row)
        except Exception as e:
            # Keep going; ExifTool fallback can still be used if GPS is missing.
            rows.append({
                "SourceFile": f"{Path(tar_path).name}::{member_name}",
                "tif_name": _tar_basename(member_name),
                "tar_source": tar_source,
                "exif_error": str(e),
            })

    elapsed = timer() - start
    df = pd.DataFrame(rows)
    print(f"       ✅ Done in {elapsed:.2f} seconds ({len(df)} rows)")
    return df


def extract_exif_metadata(tif_paths, tar_source, batch_size=200, exiftool_path=None):
    """Extract EXIF/GPS metadata faster.

    When given a directory, this runs one recursive ExifTool scan instead of
    spawning hundreds of small batches. When given a list, it uses an ExifTool
    argfile to avoid Windows command-line length limits.
    """
    print("⚙ Extracting EXIF metadata with ExifTool...")

    if exiftool_path is None:
        exiftool_path = globals().get("exiftool_path") or os.getenv("PI10_EXIFTOOL_PATH", "exiftool")

    exiftool_path = str(exiftool_path)
    resolved_exiftool = shutil.which(exiftool_path)
    if resolved_exiftool:
        exiftool_path = resolved_exiftool
    elif not Path(exiftool_path).exists() and Path(exiftool_path).name.startswith("exiftool"):
        exiftool_path = shutil.which("exiftool") or exiftool_path

    tags = [
        "-GPSLatitude",
        "-GPSLongitude",
        "-FileModifyDate",
        "-DateTimeOriginal",
        "-CreateDate",
        "-ModifyDate",
    ]
    base_args = [
        exiftool_path,
        "-j",
        "-n",
        "-api", "QuickTimeUTC",
    ] + tags

    total_start_time = time.time()
    temp_argfile = None
    try:
        if isinstance(tif_paths, (str, Path)) and Path(tif_paths).is_dir():
            root_dir = Path(tif_paths)
            args = base_args + ["-r", "-ext", "tif", "-ext", "tiff", str(root_dir)]
        else:
            # Backward-compatible list input, but use one argfile rather than many subprocesses.
            clean_paths = [
                str(p) for p in tif_paths
                if os.path.basename(str(p)) not in IGNORED_TIF_NAMES
            ]
            if not clean_paths:
                print("       ⚠️ No valid TIFF files for EXIF extraction.")
                return pd.DataFrame()
            import tempfile
            temp_argfile = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="\n")
            for p in clean_paths:
                temp_argfile.write(str(p) + "\n")
            temp_argfile.close()
            args = base_args + ["-@", temp_argfile.name]

        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"       ❌ ExifTool error: {result.stderr.strip()[:1000]}")
            return pd.DataFrame()

        rows = json.loads(result.stdout) if result.stdout.strip() else []
    except Exception as e:
        print(f"       ❌ Exception in EXIF extraction: {e}")
        return pd.DataFrame()
    finally:
        if temp_argfile is not None:
            try:
                os.unlink(temp_argfile.name)
            except Exception:
                pass

    elapsed = time.time() - total_start_time
    print(f"       ✅ Done in {elapsed:.2f} seconds ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["tar_source"] = tar_source
    if "SourceFile" in df.columns:
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(str(x)))

    if "tif_name" in df.columns:
        df = df[~df["tif_name"].isin(IGNORED_TIF_NAMES)].copy()
        if df.empty:
            return df

    dt_col = next((c for c in ["FileModifyDate", "DateTimeOriginal", "CreateDate", "ModifyDate"] if c in df.columns), None)
    if dt_col:
        df[dt_col + "_parsed"] = parse_exif_datetime_series(df[dt_col])

    return df


def write_exif_csvs(df, tar_name, output_dir, backup_dir):
    # Ensure tif_name exists
    if "tif_name" not in df.columns and "SourceFile" in df.columns:
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(str(x)))

    if "tif_name" in df.columns:
        df = df[~df["tif_name"].isin(IGNORED_TIF_NAMES)].copy()

    # Accept multiple possible timestamp fields
    time_keys = ["DateTimeOriginal", "FileModifyDate", "CreateDate", "ModifyDate"]

    cols = []
    if "SourceFile" in df.columns: cols.append("SourceFile")
    cols.append("tif_name")

    for c in ["GPSLatitude", "GPSLongitude"] + time_keys:
        if c in df.columns:
            cols.append(c)

    df = df[cols]

    outname = f"{tar_name}_gpstag.csv"
    (output_dir / outname).parent.mkdir(parents=True, exist_ok=True)
    (backup_dir / outname).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / outname, index=False)
    df.to_csv(backup_dir / outname, index=False)
    #print(f"✅ Saved EXIF CSV with GPS/timestamps: {outname}")



def getImageRegionList(filename):
    image = imread(filename)
    if image.ndim == 3:
        image = rgb2gray(image)
    image_threshold = np.where(image > np.mean(image), 0., 1.0)
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))
    label_list = measure.label(image_dilated)
    label_list = (image_threshold * label_list).astype(int)
    return measure.regionprops(label_list)

def getMaxAreaDict(filename):
    regions = getImageRegionList(filename)
    if not regions:
        return {'object_additional_area': 0}
    r = max(regions, key=lambda x: x.area)
    return {
        'object_additional_diameter_equivalent': r.equivalent_diameter,
        'object_additional_length_minor_axis': r.minor_axis_length,
        'object_additional_length_major_axis': r.major_axis_length,
        'object_additional_eccentricity': r.eccentricity,
        'object_additional_area': r.area,
        'object_additional_perimeter': r.perimeter,
        'object_additional_orientation': r.orientation,
        'object_additional_area_convex': r.convex_area,
        'object_additional_area_filled': r.filled_area,
        'object_additional_box_min_row': r.bbox[0],
        'object_additional_box_max_row': r.bbox[2],
        'object_additional_box_min_col': r.bbox[1],
        'object_additional_box_max_col': r.bbox[3],
        'object_additional_ratio_extent': r.extent,
        'object_additional_ratio_solidity': r.solidity,
        'object_additional_inertia_tensor_eigenvalue1': r.inertia_tensor_eigvals[0],
        'object_additional_inertia_tensor_eigenvalue2': r.inertia_tensor_eigvals[1],
        'object_additional_moments_hu1': r.moments_hu[0],
        'object_additional_moments_hu2': r.moments_hu[1],
        'object_additional_moments_hu3': r.moments_hu[2],
        'object_additional_moments_hu4': r.moments_hu[3],
        'object_additional_moments_hu5': r.moments_hu[4],
        'object_additional_moments_hu6': r.moments_hu[5],
        'object_additional_moments_hu7': r.moments_hu[6],
        'object_additional_euler_number': r.euler_number,
        'object_additional_countcoords': len(r.coords)
    }

def _valid_classification_tifs(extract_dir):
    extract_dir = Path(extract_dir)
    return sorted(
        [p for p in extract_dir.rglob("*.tif") if p.name not in IGNORED_TIF_NAMES],
        key=lambda p: str(p).lower(),
    )


def _morphology_worker(path_and_rel):
    path, rel_path = path_and_rel
    try:
        props = getMaxAreaDict(path)
        props["filepath"] = rel_path
        return props, None
    except Exception as e:
        return None, f"{rel_path}: {e}"


def _extract_morphology_rows(filepaths, extract_dir):
    tasks = [(str(path), str(path.relative_to(extract_dir))) for path in filepaths]
    results_csv = []
    errors = []

    if MORPHOLOGY_WORKERS <= 1 or len(tasks) < 1000:
        iterator = map(_morphology_worker, tasks)
    else:
        pool = ThreadPoolExecutor(max_workers=MORPHOLOGY_WORKERS)
        iterator = pool.map(_morphology_worker, tasks, chunksize=64)

    try:
        for props, err in tqdm(iterator, total=len(tasks), desc="Morphology", leave=False):
            if props is not None:
                results_csv.append(props)
            elif err:
                errors.append(err)
    finally:
        if 'pool' in locals():
            pool.shutdown(wait=True)

    if errors:
        print(f"       ⚠️ Morphology errors for {len(errors)} image(s); first: {errors[0]}")
    return results_csv


def getImageRegionListFromArray(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = rgb2gray(image)
    image_threshold = np.where(image > np.mean(image), 0., 1.0)
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))
    label_list = measure.label(image_dilated)
    label_list = (image_threshold * label_list).astype(int)
    return measure.regionprops(label_list)


def getMaxAreaDictFromArray(image):
    regions = getImageRegionListFromArray(image)
    if not regions:
        return {'object_additional_area': 0}
    r = max(regions, key=lambda x: x.area)
    return {
        'object_additional_diameter_equivalent': r.equivalent_diameter,
        'object_additional_length_minor_axis': r.minor_axis_length,
        'object_additional_length_major_axis': r.major_axis_length,
        'object_additional_eccentricity': r.eccentricity,
        'object_additional_area': r.area,
        'object_additional_perimeter': r.perimeter,
        'object_additional_orientation': r.orientation,
        'object_additional_area_convex': r.convex_area,
        'object_additional_area_filled': r.filled_area,
        'object_additional_box_min_row': r.bbox[0],
        'object_additional_box_max_row': r.bbox[2],
        'object_additional_box_min_col': r.bbox[1],
        'object_additional_box_max_col': r.bbox[3],
        'object_additional_ratio_extent': r.extent,
        'object_additional_ratio_solidity': r.solidity,
        'object_additional_inertia_tensor_eigenvalue1': r.inertia_tensor_eigvals[0],
        'object_additional_inertia_tensor_eigenvalue2': r.inertia_tensor_eigvals[1],
        'object_additional_moments_hu1': r.moments_hu[0],
        'object_additional_moments_hu2': r.moments_hu[1],
        'object_additional_moments_hu3': r.moments_hu[2],
        'object_additional_moments_hu4': r.moments_hu[3],
        'object_additional_moments_hu5': r.moments_hu[4],
        'object_additional_moments_hu6': r.moments_hu[5],
        'object_additional_moments_hu7': r.moments_hu[6],
        'object_additional_euler_number': r.euler_number,
        'object_additional_countcoords': len(r.coords)
    }


def _morphology_tar_worker(task):
    tar_path, member_name, info = task
    try:
        info_map = {member_name: info} if info else None
        image = read_tiff_array_from_tar(tar_path, member_name, info_map)
        props = getMaxAreaDictFromArray(image)
        props["filepath"] = member_name
        return props, None
    except Exception as e:
        return None, f"{member_name}: {e}"


def _extract_morphology_rows_from_tar(tar_path, member_names, member_info):
    member_names = list(member_names or [])
    tasks = [(str(tar_path), name, (member_info or {}).get(name)) for name in member_names]
    results_csv = []
    errors = []

    if MORPHOLOGY_WORKERS <= 1 or len(tasks) < 1000:
        iterator = map(_morphology_tar_worker, tasks)
    else:
        pool = ThreadPoolExecutor(max_workers=MORPHOLOGY_WORKERS)
        iterator = pool.map(_morphology_tar_worker, tasks, chunksize=64)

    try:
        for props, err in tqdm(iterator, total=len(tasks), desc="TAR morphology", leave=False):
            if props is not None:
                results_csv.append(props)
            elif err:
                errors.append(err)
    finally:
        if 'pool' in locals():
            pool.shutdown(wait=True)
        close_tar_stream_file_cache()

    if errors:
        print(f"       ⚠️ TAR morphology errors for {len(errors)} image(s); first: {errors[0]}")
    return results_csv


def classify_and_extract_regions_from_tar(tar_file, tar_path, image_member_names, inventory,
                                          write_morphology=True):
    start_time = time.time()
    base_name = Path(tar_file).stem
    json_path = source_dir / f"{base_name}_predictions_relative.json"
    csv_path = source_dir / f"{base_name}_image_properties.csv"
    image_member_names = sorted(list(image_member_names or []), key=lambda x: str(x).lower())
    member_info = (inventory or {}).get("member_info", {})

    if not image_member_names:
        print(f"⚠️ No valid .tif members in {base_name}, skipping.")
        return None

    if not json_path.exists():
        print(f"⚙ Predicting {len(image_member_names)} TIFF files directly from TAR")
        pred_lab, pred_prob = predict_tar_members(
            model,
            tar_path,
            image_member_names,
            conf,
            member_info=member_info,
            top_K=TOP_K,
        )

        results_json = []
        for i, member_name in enumerate(image_member_names):
            labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
            probs = [float(pred_prob[i, j]) for j in range(TOP_K)]
            results_json.append({
                "filepath": member_name,
                f"top{TOP_K}_labels": labels,
                f"top{TOP_K}_probs": probs,
            })

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, separators=(",", ":"))
    else:
        pred_lab = None
        print("✅ Skipping TAR-stream prediction (JSON already exists)")

    if write_morphology and not csv_path.exists():
        print("⚙ Extracting morphology properties directly from TAR...")
        results_csv = _extract_morphology_rows_from_tar(tar_path, image_member_names, member_info)
        if results_csv:
            pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        else:
            print(f"       ⚠️ No region properties written for {base_name}")

    elapsed_time = time.time() - start_time
    print(f"       ✅ TAR-stream classification/morphology done in {elapsed_time / 3600:.1f} hours.")
    return pred_lab


def classify_and_extract_regions(tar_file, extract_dir):
    start_time = time.time()
    base_name = tar_file.stem
    json_path = source_dir / f"{base_name}_predictions_relative.json"
    csv_path = source_dir / f"{base_name}_image_properties.csv"
    FILEPATHS = _valid_classification_tifs(extract_dir)

    if not FILEPATHS:
        print(f"⚠️ No valid .tif files in {base_name}, skipping.")
        return None

    print(f"⚙ Predicting {len(FILEPATHS)} TIFF files")
    pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

    results_json = []
    for i, path in enumerate(FILEPATHS):
        rel_path = str(path.relative_to(extract_dir))
        labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
        probs = [float(pred_prob[i, j]) for j in range(TOP_K)]
        results_json.append({
            "filepath": rel_path,
            f"top{TOP_K}_labels": labels,
            f"top{TOP_K}_probs": probs,
        })

    # Compact JSON is much faster/smaller than indent=2 for ~100k entries.
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, separators=(",", ":"))

    print("⚙ Extracting morphology properties...")
    results_csv = _extract_morphology_rows(FILEPATHS, extract_dir)
    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
    else:
        print(f"       ⚠️ No region properties written for {base_name}")

    elapsed_time = time.time() - start_time
    print(f"       ✅ Classification + morphology done in {elapsed_time / 3600:.1f} hours.")
    return pred_lab


import os
import json
import pandas as pd


####SET BARS
DEFAULT_TAXA_THRESHOLDS = {
    "diatom-setae": {"upper_threshold": 0.999, "diff_threshold": 0.99}, #0.8 and 0.5
    "dinoflagellate_noctiluca-intact": {"upper_threshold": 0.99, "diff_threshold": 0.99}, #0.1 and 0.09
}

BUBBLES_RULE = {"upper_threshold": 0.999, "diff_threshold": 0.99} #0.1 and 0.09
BUBBLES_SUBSTR = "bubbles"

DETRITUS_RULE = {"upper_threshold": 0.999, "diff_threshold": 0.99} #0.1 and 0.09
DETRITUS_SUBSTR = "noctiluca"


def generate_topspecies_csv(json_path,
                            taxa_thresholds=None,
                            decimals=6):
    print("⚙ Generating top species CSV")

    # Keep existing threshold behaviour for optional _AI99 suffix on top_species.
    if taxa_thresholds is None:
        taxa_thresholds = DEFAULT_TAXA_THRESHOLDS

    if not isinstance(taxa_thresholds, dict) or not taxa_thresholds:
        print("       ❌ Taxa thresholds must be a non-empty dict.")
        return

    json_path = Path(json_path)

    if not json_path.exists():
        print(f"       ❌ JSON file not found: {json_path}")
        return

    try:
        with open(json_path, "r") as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON: {e}")
        return

    if not isinstance(data_list, list) or not data_list:
        print("       ⚠️ JSON is empty or invalid.")
        return

    rows = []

    for entry in data_list:
        filepath = entry.get("filepath", "")

        labels = entry.get(f"top{TOP_K}_labels", []) or entry.get("top2_labels", [])
        probs = entry.get(f"top{TOP_K}_probs", []) or entry.get("top2_probs", [])

        taxa = entry.get("taxa", "")

        # Coerce labels/probs if they somehow arrived as comma-separated strings.
        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",") if s.strip()]
        else:
            labels = list(labels or [])

        if isinstance(probs, str):
            parsed_probs = []
            for s in probs.split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    parsed_probs.append(float(s))
                except ValueError:
                    parsed_probs.append(None)
            probs = parsed_probs
        else:
            probs = list(probs or [])

        # Need at least top-1. Top-2 can be missing, but normally should exist.
        if not filepath or len(labels) < 1 or len(probs) < 1:
            continue

        label_t1 = str(labels[0]).strip()
        prob1 = float(probs[0]) if probs[0] is not None else None

        label_t2 = str(labels[1]).strip() if len(labels) >= 2 else None
        prob2 = float(probs[1]) if len(probs) >= 2 and probs[1] is not None else None

        if prob1 is None:
            continue

        # Keep raw top-1 before optional _AI99 suffix.
        top_species = label_t1

        # Defaults
        upper_threshold = 0.999
        diff_threshold = 0.99

        # Existing threshold rules
        if taxa in taxa_thresholds:
            upper_threshold = taxa_thresholds[taxa].get("upper_threshold", upper_threshold)
            diff_threshold = taxa_thresholds[taxa].get("diff_threshold", diff_threshold)
        elif (isinstance(taxa, str) and BUBBLES_SUBSTR in taxa.lower()) or \
                (isinstance(top_species, str) and BUBBLES_SUBSTR in top_species.lower()):
            upper_threshold = BUBBLES_RULE["upper_threshold"]
            diff_threshold = BUBBLES_RULE["diff_threshold"]
        elif (isinstance(taxa, str) and DETRITUS_SUBSTR in taxa.lower()) or \
                (isinstance(top_species, str) and DETRITUS_SUBSTR in top_species.lower()):
            upper_threshold = DETRITUS_RULE["upper_threshold"]
            diff_threshold = DETRITUS_RULE["diff_threshold"]

        # Apply optional _AI99 only to top_species/top1.
        # Leave top_species_t2 raw, so downstream translation/gap logic is clear.
        if prob2 is not None and prob1 > upper_threshold and (prob1 - prob2) > diff_threshold:
            top_species = f"{top_species}_AI99"

        rows.append({
            "filename": os.path.basename(filepath),
            "top_species": top_species,
            "confidence": prob1,
            "top_species_t2": label_t2,
            "confidence_t2": prob2
        })

    if not rows:
        print("       ⚠️ No valid predictions to save.")
        return

    output_path = json_path.with_name(
        json_path.name.replace("_predictions_relative.json", "_topspecies.csv")
    )

    df = pd.DataFrame(
        rows,
        columns=[
            "filename",
            "top_species",
            "confidence",
            "top_species_t2",
            "confidence_t2"
        ]
    )

    df.to_csv(output_path, index=False, float_format=f"%.{decimals}f")

    print(f"       ✅ Done: {output_path.name} ({len(df)} rows)")


def check_preview_class_distribution(preview_tifs, threshold=PREVIEW_BUBBLE_THRESHOLD):    #more then 40% bubbles, move to quarantaine!!!
    print("⚙ Running preview classification check...")

    preview_tifs = list(preview_tifs or [])
    if not preview_tifs:
        print("       ⚠️ No preview sample available.")
        return True, None  # Allow pipeline to continue

    pred_lab, pred_prob = predict(model, preview_tifs, conf, top_K=1, filemode='local')
    top1_classes = [class_names[idx] for idx in pred_lab[:, 0]]

    class_counts = pd.Series(top1_classes).value_counts(normalize=True)
    print(f"       ✅ Class distribution in preview: {class_counts.to_dict()}")

    bubble_classes = [cls for cls in class_counts.index if 'bubbles' in cls.lower()]
    bubble_fraction = class_counts[bubble_classes].sum()

    if bubble_fraction > threshold:
        print(f"✅ Combined 'bubbles'-like classes exceed threshold ({threshold:.0%}): {bubble_fraction:.2%}, moved to quarantine")
        return False, 'bubbles'

    return True, None

def check_preview_class_distribution_from_tar(tar_path, preview_member_names, inventory, threshold=PREVIEW_BUBBLE_THRESHOLD):
    print("⚙ Running preview classification check directly from TAR...")

    preview_member_names = list(preview_member_names or [])
    if not preview_member_names:
        print("       ⚠️ No preview sample available.")
        return True, None

    pred_lab, pred_prob = predict_tar_members(
        model,
        tar_path,
        preview_member_names,
        conf,
        member_info=(inventory or {}).get("member_info", {}),
        top_K=1,
    )
    top1_classes = [class_names[idx] for idx in pred_lab[:, 0]]

    class_counts = pd.Series(top1_classes).value_counts(normalize=True)
    print(f"       ✅ Class distribution in preview: {class_counts.to_dict()}")

    bubble_classes = [cls for cls in class_counts.index if 'bubbles' in cls.lower()]
    bubble_fraction = class_counts[bubble_classes].sum()

    if bubble_fraction > threshold:
        print(f"✅ Combined 'bubbles'-like classes exceed threshold ({threshold:.0%}): {bubble_fraction:.2%}, moved to quarantine")
        return False, 'bubbles'

    return True, None


def extract_only_morphology_from_tar(tar_file, tar_path, image_member_names, csv_path, inventory):
    base_name = Path(tar_file).stem
    image_member_names = sorted(list(image_member_names or []), key=lambda x: str(x).lower())
    member_info = (inventory or {}).get("member_info", {})

    if not image_member_names:
        print(f"       ⚠️ No valid .tif members in {base_name}, skipping morphology.")
        return

    print(f"       🧬 Extracting morphology for {len(image_member_names)} TIFFs directly from TAR with {MORPHOLOGY_WORKERS} worker(s)...")
    results_csv = _extract_morphology_rows_from_tar(tar_path, image_member_names, member_info)

    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️ No morphology data written for {base_name}")


def extract_only_morphology(tar_file, extract_dir, csv_path):
    base_name = tar_file.stem
    FILEPATHS = _valid_classification_tifs(extract_dir)

    if not FILEPATHS:
        print(f"       ⚠️ No valid .tif files in {base_name}, skipping morphology.")
        return

    print(f"       🧬 Extracting morphology for {len(FILEPATHS)} TIFFs with {MORPHOLOGY_WORKERS} worker(s)...")
    results_csv = _extract_morphology_rows(FILEPATHS, extract_dir)

    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️ No morphology data written for {base_name}")


def safe_bg_coords(bg_path):
    lat, lon = get_background_coordinates(bg_path)
    return (lat if lat is not None else 0, lon if lon is not None else 0)



def log_per_minute_metrics(tar_name, json_output, hits_file, exif_df, out_dir, num_images):
    import os

    # Volume per minute (fixed)
    V_m3 = 0.034  # 34 L/min = 0.034 m³/min

    # Target taxa/groups for per-minute biometrics.
    # Matching is done after lowercasing and replacing '-' with '_'.
    TAXON_GROUPS = {
        "appendicularia": ["appendicularia"],
        "copepod_calanoida": ["copepod_calanoida", "copepod_calanoid"],
        "diatoms": ["diatom"],
        "dinoflagellate": ["dinoflagellate"],
        "meroplankton": ["mero"],
    }

    def normalize_label(label):
        return str(label or "").strip().lower().replace("-", "_")

    def labels_for_entry(entry):
        labels = entry.get(f"top{TOP_K}_labels", []) or entry.get("top2_labels", []) or entry.get("top1_labels", [])
        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",") if s.strip()]
        return list(labels or [])

    def groups_for_label(label):
        normalized = normalize_label(label)
        groups = []
        for group_name, patterns in TAXON_GROUPS.items():
            if any(pattern in normalized for pattern in patterns):
                groups.append(group_name)
        return groups

    try:
        # --- 0) Load hits/misses ---
        if not hits_file.exists():
            print(f"⚠️ hitsmisses.txt missing for {tar_name}")
            return

        try:
            df_hits = pd.read_csv(hits_file)  # with header
        except Exception:
            df_hits = pd.read_csv(hits_file, header=None)
            df_hits.columns = ["hits", "misses"]

        # --- 1) Directly assign tar_name to tar_source column ---
        df_hits["tar_source"] = tar_name

        if "minute" not in df_hits.columns:
            df_hits["minute"] = range(len(df_hits))

        # Ensure necessary columns exist
        if "GPSLatitude" not in df_hits.columns:
            df_hits["GPSLatitude"] = "NA"
        if "GPSLongitude" not in df_hits.columns:
            df_hits["GPSLongitude"] = "NA"
        if "total_images_in_tar" not in df_hits.columns:
            df_hits["total_images_in_tar"] = num_images

        max_minute = int(df_hits["minute"].max())
        minutes = list(df_hits["minute"])

        # --- 2) Prepare EXIF → assign minutes sequentially ---
        lat_by_minute, lon_by_minute = {}, {}
        if exif_df is not None and not exif_df.empty:
            df_exif = exif_df.copy()
            if "tif_name" not in df_exif.columns and "SourceFile" in df_exif.columns:
                df_exif["tif_name"] = df_exif["SourceFile"].apply(lambda x: os.path.basename(str(x)))
            if "tif_name" in df_exif.columns:
                df_exif = df_exif[~df_exif["tif_name"].isin(IGNORED_TIF_NAMES)].copy()

            ts_cols = [c for c in ["DateTimeOriginal", "FileModifyDate", "ModifyDate"] if c in df_exif.columns]

            if ts_cols:
                df_exif["capture_dt"] = pd.NaT
                for col in ts_cols:
                    s = pd.to_datetime(
                        df_exif[col],
                        format="%Y:%m:%d %H:%M:%S",
                        errors="coerce",
                        utc=True
                    )
                    # make tz-naive so it matches df_exif["capture_dt"]
                    if s.dt.tz is not None:
                        s = s.dt.tz_localize(None)

                    mask = df_exif["capture_dt"].isna()
                    df_exif.loc[mask, "capture_dt"] = s[mask]

                df_exif = df_exif.sort_values("capture_dt").reset_index(drop=True)

            else:
                df_exif = df_exif.reset_index(drop=True)

            n = len(df_exif)
            rows_per_min = max(1, n // (max_minute + 1))
            df_exif["minute"] = df_exif.index // rows_per_min
            df_exif["minute"] = df_exif["minute"].clip(0, max_minute)

            for c in ["GPSLatitude", "GPSLongitude"]:
                if c in df_exif.columns:
                    df_exif[c] = pd.to_numeric(df_exif[c], errors="coerce")

            coords_df = (df_exif
                         .dropna(subset=["GPSLatitude", "GPSLongitude"])
                         .groupby("minute", as_index=False)[["GPSLatitude", "GPSLongitude"]]
                         .median())
            lat_by_minute = dict(zip(coords_df["minute"], coords_df["GPSLatitude"]))
            lon_by_minute = dict(zip(coords_df["minute"], coords_df["GPSLongitude"]))

            exif_df = df_exif

        # --- 3) Map EXIF data to df_hits for GPS and fill in missing values ---
        df_hits["GPSLatitude"] = df_hits["minute"].map(lambda m: lat_by_minute.get(m, "NA"))
        df_hits["GPSLongitude"] = df_hits["minute"].map(lambda m: lon_by_minute.get(m, "NA"))

        # --- 4) Taxon counts + bubble QC counts per minute ---
        taxon_counts = {
            group_name: {m: 0 for m in minutes}
            for group_name in TAXON_GROUPS
        }
        bubble_counts = {m: 0 for m in minutes}
        top1_by_name = {}
        groups_by_name = {}

        if json_output.exists() and exif_df is not None and "tif_name" in exif_df.columns:
            name_to_minute = exif_df.set_index("tif_name")["minute"].to_dict()
            with open(json_output, "r") as f:
                data = json.load(f)

            for entry in data:
                fname = os.path.basename(entry.get("filepath", ""))
                labels = labels_for_entry(entry)

                if labels:
                    top1 = labels[0]
                    top1_by_name[fname] = top1
                    matched_groups = groups_for_label(top1)
                    groups_by_name[fname] = matched_groups
                else:
                    matched_groups = []

                m = name_to_minute.get(fname, None)
                if m is not None:
                    for group_name in matched_groups:
                        taxon_counts[group_name][m] = taxon_counts[group_name].get(m, 0) + 1

                    # Keep bubbles as a QC count, using all available predicted labels.
                    if any("bubb" in normalize_label(label) for label in labels):
                        bubble_counts[m] = bubble_counts.get(m, 0) + 1

        # --- 5) Morphometrics per target taxon/group ---
        # size_sum_count is the observed sum from classified image objects only.
        # size_sum_abundance is calculated later as size_mean * abundance, giving
        # an abundance-standardized total size / biomass proxy for that minute.
        size_sum_count = {
            group_name: {m: 0 for m in minutes}
            for group_name in TAXON_GROUPS
        }
        size_mean = {
            group_name: {m: 0 for m in minutes}
            for group_name in TAXON_GROUPS
        }

        img_props_path = out_dir / f"{tar_name}_image_properties.csv"
        if img_props_path.exists() and exif_df is not None and "tif_name" in exif_df.columns:
            df_props = pd.read_csv(img_props_path)
            if "filepath" in df_props.columns and "object_additional_diameter_equivalent" in df_props.columns:
                # build mapping: filename -> minute
                name_to_minute = exif_df.set_index("tif_name")["minute"].to_dict()

                for _, row in df_props.iterrows():
                    fname = os.path.basename(str(row["filepath"]))
                    m = name_to_minute.get(fname, None)
                    if m is None:
                        continue

                    d = row["object_additional_diameter_equivalent"]
                    if pd.isna(d):
                        continue

                    for group_name in groups_by_name.get(fname, []):
                        size_sum_count[group_name][m] = size_sum_count[group_name].get(m, 0) + d

                for group_name in TAXON_GROUPS:
                    for m in minutes:
                        count = taxon_counts[group_name].get(m, 0)
                        if count > 0:
                            size_mean[group_name][m] = size_sum_count[group_name][m] / count
                        else:
                            size_sum_count[group_name][m] = 0
                            size_mean[group_name][m] = 0

        # --- 6) Merge and compute abundance + densities ---
        df_hits["bubbles"] = df_hits["minute"].map(bubble_counts)

        for group_name in TAXON_GROUPS:
            count_col = f"{group_name}_count"
            abundance_col = f"{group_name}_abundance"
            density_col = f"{group_name}_density_ind_m3"
            size_mean_col = f"{group_name}_size_mean"
            size_sum_col = f"{group_name}_size_sum"
            size_sum_abundance_col = f"{group_name}_size_sum_abundance"

            df_hits[count_col] = df_hits["minute"].map(taxon_counts[group_name])

            hits_numeric = pd.to_numeric(df_hits["hits"], errors="coerce")
            misses_numeric = pd.to_numeric(df_hits["misses"], errors="coerce")
            count_numeric = pd.to_numeric(df_hits[count_col], errors="coerce")

            # Estimate count in misses from observed proportion in hits.
            # If hits == 0, set the proportion to 0 to avoid division by zero.
            proportion_in_hits = (count_numeric / hits_numeric).replace([np.inf, -np.inf], np.nan).fillna(0)
            estimated_in_misses = proportion_in_hits * misses_numeric.fillna(0)
            df_hits[abundance_col] = count_numeric.fillna(0) + estimated_in_misses
            df_hits[density_col] = df_hits[abundance_col] / V_m3

            # Size mean and size_sum are measured only from observed/classified images.
            # size_sum_abundance standardizes the size sum by the hit/miss-corrected abundance
            # and can be used as a total size / biomass proxy for the minute.
            df_hits[size_mean_col] = df_hits["minute"].map(size_mean[group_name])
            df_hits[size_sum_col] = df_hits["minute"].map(size_sum_count[group_name])
            df_hits[size_sum_abundance_col] = df_hits[size_mean_col] * df_hits[abundance_col]

        df_hits["model_name"] = TIMESTAMP

        # Reorder columns
        base_cols = [
            "tar_source", "model_name", "minute", "hits", "misses", "bubbles",
            "GPSLatitude", "GPSLongitude", "total_images_in_tar"
        ]

        taxon_cols = []
        for group_name in TAXON_GROUPS:
            taxon_cols.extend([
                f"{group_name}_count",
                f"{group_name}_abundance",
                f"{group_name}_density_ind_m3",
                f"{group_name}_size_mean",
                f"{group_name}_size_sum",
                f"{group_name}_size_sum_abundance",
            ])

        df_hits = df_hits[base_cols + taxon_cols]

        # Force proper dtypes before saving
        numeric_cols = [
            "minute", "hits", "misses", "bubbles",
            "GPSLatitude", "GPSLongitude", "total_images_in_tar",
        ] + taxon_cols

        for col in numeric_cols:
            if col in df_hits.columns:
                df_hits[col] = pd.to_numeric(df_hits[col], errors="coerce")

        # Save clean CSV
        out_path = out_dir / f"{tar_name}_bio-metrics.csv"
        df_hits.to_csv(out_path, index=False, float_format="%.6f")
        print("       ✅ Done")

    except Exception as e:
        print(f"❌ Failed per-minute log for {tar_name}: {e}")
        try:
            out_path = out_dir / f"{tar_name}_bio-metrics.csv"
            df_hits.to_csv(out_path, index=False)
        except Exception:
            pass

def map_exif_to_minutes(exif_df, hits_len):
    # Example: use the file index pattern from filename "_0001_"
    exif_df["minute"] = None
    for i, row in exif_df.iterrows():
        fname = row.get("tif_name", "")
        for m in range(hits_len):
            if f"_{m:04d}_" in fname or f"_{m:03d}_" in fname:
                exif_df.at[i, "minute"] = m
                break
    return exif_df
def clean_coord(value):
    # If value is tuple or list, flatten to string
    if isinstance(value, (tuple, list)):
        return ",".join(map(str, value))
    return value if value is not None else "NA"


# === MAIN PROCESS ===
def _load_existing_exif_csv(exif_csv_path):
    try:
        exif_df = pd.read_csv(exif_csv_path)
        if "tif_name" not in exif_df.columns and "SourceFile" in exif_df.columns:
            exif_df["tif_name"] = exif_df["SourceFile"].apply(lambda x: os.path.basename(str(x)))
        if "tif_name" in exif_df.columns:
            exif_df = exif_df[~exif_df["tif_name"].isin(IGNORED_TIF_NAMES)].copy()
        return exif_df
    except Exception as rd_err:
        print(f"       ⚠️ Failed to load existing EXIF CSV: {rd_err}")
        return None


def process_tar(tar_file):
    tar_file = Path(tar_file)
    tar_name = tar_file.stem
    print(f"\n🔧🔧🔧 PROCESSING {tar_name.upper()} 🔧🔧🔧")

    times_dict = {}
    num_images = 0
    status_log = []
    quarantined = False
    quarantine_reason = None
    already_logged = False
    exif_df = None
    working_tar = tar_file
    copied_working_tar = False

    if outputs_exist_for_tar(tar_file):
        print(f"📦 All outputs exist for {tar_name}, skipping.")
        return

    json_output = source_dir / f"{tar_name}_predictions_relative.json"
    csv_output = source_dir / f"{tar_name}_image_properties.csv"
    topspecies_csv = source_dir / f"{tar_name}_topspecies.csv"
    exif_csv_path = source_dir / f"{tar_name}_gpstag.csv"
    hits_path = source_dir / f"{tar_name}_hitsmisses.txt"
    bg_path = source_dir / f"{tar_name}_Background.tif"
    tar_dest = paths['tarred'] / tar_file.name
    extract_dir = paths['untarred'] / tar_name
    preview_dir = paths['preview'] / tar_name

    try:
        clear_temp_output_dir(paths["output"])

        # Step 1: optional local copy. On Linux/default this is skipped to avoid
        # duplicating a huge TAR; on Windows Y: -> D: it auto-copies by default.
        working_tar, copied_working_tar, copy_elapsed = prepare_working_tar(tar_file, tar_dest)
        times_dict["Copy TAR to working directory"] = copy_elapsed
        if copied_working_tar:
            status_log.append("TAR copied to local working directory")
        else:
            status_log.append("TAR read in-place; local copy skipped")

        # Step 2: inventory only. This replaces expensive rglob counting before extraction.
        inventory, inv_elapsed = tar_inventory(working_tar)
        times_dict["Tar inventory"] = inv_elapsed
        num_images = len(inventory["tif_all"])
        num_classifiable = len(inventory["tif_images"])
        status_log.append(f"TAR inventory: {num_images} TIFFs; {num_classifiable} classifiable TIFFs")

        # Step 2b: gray-edge from a small sample directly inside the TAR.
        start_time = time.time()
        should_continue, reason, gray_edge_mean = check_gray_edge_quarantine_from_tar(
            working_tar,
            inventory["tif_images"],
            tar_name,
        )
        times_dict["Gray edge quarantine check"] = track_time(start_time, "Gray edge quarantine check")
        if gray_edge_mean is not None:
            status_log.append(f"Gray-edge mean grayscale: {gray_edge_mean:.2f}")
        else:
            status_log.append("Gray-edge quarantine check skipped/no readable TIFFs")

        if not should_continue:
            quarantined = True
            quarantine_reason = reason
            if move_tar_to_quarantine(tar_file, quarantine_gray_edge_dir / tar_file.name, status_log, quarantine_reason):
                print(f"🚨 Quarantined {tar_file.name} → gray-edge issue")
                remove_partial_outputs(tar_name, status_log)

        # Step 3: hits/misses directly from TAR.
        if not quarantined:
            start_time = time.time()
            if not hits_path.exists():
                ok, hitsmiss_reason = extract_hitsmisses_from_tar(
                    working_tar,
                    inventory["hitsmisses"],
                    hits_path,
                    tar_file,
                    status_log,
                )
                if not ok:
                    quarantined = True
                    quarantine_reason = hitsmiss_reason
                    quarantine_target = (
                        quarantine_raisingfactor_dir / tar_file.name
                        if hitsmiss_reason == "too_many_misses"
                        else quarantine_hitsmiss_dir / tar_file.name
                    )
                    if move_tar_to_quarantine(tar_file, quarantine_target, status_log, quarantine_reason):
                        print(f"🚨 Quarantined {tar_file.name} → {hitsmiss_reason}")
                        remove_partial_outputs(tar_name, status_log)
            else:
                status_log.append("hitsmisses.txt already exists (skipped)")
            times_dict["Extract hitsmisses.txt"] = track_time(start_time, "Extract hitsmisses.txt")

        # Step 4: Count images from inventory.
        if not quarantined:
            start_time = time.time()
            print(f"⚙ Counting images in {tar_file.name} from TAR inventory...")
            print(f"       ✅ Found {num_images} .tif files ({num_classifiable} classifiable)")
            status_log.append(f"Number of TIFFs in TAR: {num_images}; classifiable: {num_classifiable}")
            times_dict["Count images"] = track_time(start_time, "Count images")

        # Step 5: prepare the 200-image preview sample.
        # With TAR_STREAM_PREDICT this creates only in-memory TAR references;
        # no preview TIFFs are written to disk.
        preview_sample_paths = []
        preview_member_names = []
        if not quarantined:
            start_time = time.time()
            preview_member_names = get_preview_sample_from_inventory(inventory["tif_images"], n=PREVIEW_SAMPLE_N)
            if TAR_STREAM_PREDICT:
                print(f"⚙ Prepared {len(preview_member_names)} preview TAR references (no extraction)")
                status_log.append(f"Preview subset prepared as TAR refs ({len(preview_member_names)} images, not extracted)")
            else:
                preview_sample_paths = extract_tar_members_to_dir(working_tar, preview_member_names, preview_dir)
                status_log.append(f"Preview subset extracted only for checks ({len(preview_sample_paths)} images)")
            times_dict["Create preview images"] = track_time(start_time, "Create preview images")

        # Step 6: early preview classification.
        if not quarantined:
            start_time = time.time()
            if json_output.exists():
                print("✅ Skipping preview classification (predictions already exist)")
                should_continue, reason = True, None
            elif TAR_STREAM_PREDICT:
                should_continue, reason = check_preview_class_distribution_from_tar(
                    working_tar,
                    preview_member_names,
                    inventory,
                    threshold=PREVIEW_BUBBLE_THRESHOLD,
                )
            else:
                should_continue, reason = check_preview_class_distribution(preview_sample_paths, threshold=PREVIEW_BUBBLE_THRESHOLD)
            status_log.append(f"Preview classification result: {reason if reason else 'OK'}")
            times_dict["Early preview classification"] = track_time(start_time, "Early preview classification")

            if not should_continue:
                quarantined = True
                quarantine_reason = reason
                if move_tar_to_quarantine(tar_file, quarantine_bubbles_dir / tar_file.name, status_log, quarantine_reason):
                    print(f"🚨 Quarantined {tar_file.name} → bubble issue")
                    remove_partial_outputs(tar_name, status_log)

        # Step 7: copy Background.tif directly from TAR.
        if not quarantined:
            start_time = time.time()
            if not bg_path.exists():
                copy_background_from_tar(working_tar, inventory["background"], bg_path)
                status_log.append("Background.tif copied successfully" if bg_path.exists() else "❌ Background.tif missing after copy attempt")
            else:
                status_log.append("Background.tif already exists (skipped)")
            times_dict["Copy Background.tif"] = track_time(start_time, "Copy Background.tif")

        # Only do the full extraction if a downstream step still needs real file paths.
        # Option A removes the file-path requirement for prediction. Additional
        # stream paths below can also avoid extraction for morphology, EXIF, and
        # taxon-preview export.
        needs_full_extract = False
        needs_full_extract = needs_full_extract or (not exif_csv_path.exists() and not TAR_STREAM_EXIF_PYTHON)
        needs_full_extract = needs_full_extract or (not json_output.exists() and not TAR_STREAM_PREDICT)
        needs_full_extract = needs_full_extract or (not csv_output.exists() and not TAR_STREAM_MORPHOLOGY)
        needs_full_extract = needs_full_extract or (
            EXPORT_TAXON_PREVIEWS
            and json_output.exists()
            and not TAR_STREAM_EXPORT_TAXON_PREVIEWS
        )

        if not quarantined and needs_full_extract:
            times_dict["Untar"] = extract_tar_fast(working_tar, extract_dir)
            status_log.append("Full TAR extraction completed for remaining file-path based step(s)")
        else:
            times_dict["Untar"] = 0.0
            if not quarantined:
                status_log.append("Full TAR extraction skipped; TAR-stream paths were enough")

        # Step 8: EXIF metadata.
        if not quarantined:
            start_time = time.time()
            if not exif_csv_path.exists():
                if TAR_STREAM_EXIF_PYTHON:
                    exif_df = extract_exif_metadata_from_tar_python(
                        working_tar,
                        inventory["tif_images"],
                        inventory.get("member_info", {}),
                        tar_name,
                    )
                    write_exif_csvs(exif_df, tar_name, paths['output'], source_dir)

                    # Avoid false no-GPS quarantine if Pillow cannot read the
                    # camera's GPS EXIF. Fall back to the old ExifTool path.
                    if (
                        TAR_STREAM_EXIF_FALLBACK_FULL_EXTRACT
                        and not has_usable_coordinates(exif_df)
                    ):
                        print("       ⚠️ TAR EXIF found no usable GPS; falling back to full extract + ExifTool before quarantine")
                        ensure_full_extract_for_fallback(
                            working_tar,
                            extract_dir,
                            times_dict,
                            status_log,
                            "ExifTool GPS fallback",
                        )
                        exif_df = extract_exif_metadata(extract_dir, tar_name)
                        write_exif_csvs(exif_df, tar_name, paths['output'], source_dir)

                    status_log.append("EXIF metadata extracted and saved")
                else:
                    if not extract_dir.exists():
                        ensure_full_extract_for_fallback(
                            working_tar,
                            extract_dir,
                            times_dict,
                            status_log,
                            "EXIF extraction",
                        )
                    exif_df = extract_exif_metadata(extract_dir, tar_name)
                    write_exif_csvs(exif_df, tar_name, paths['output'], source_dir)
                    status_log.append("EXIF metadata extracted and saved")
            else:
                exif_df = _load_existing_exif_csv(exif_csv_path)
                status_log.append("EXIF metadata loaded from CSV")
                print("✅ Skipping EXIF extraction (already exists)")
            times_dict["Extract and save EXIF metadata"] = track_time(start_time, "Extract and save EXIF metadata")

        # Step 8b: location quarantine.
        if not quarantined:
            if not has_usable_coordinates(exif_df):
                quarantined = True
                quarantine_reason = "no usable GPS coordinates"
                if move_tar_to_quarantine(tar_file, quarantine_nogps_dir / tar_file.name, status_log, quarantine_reason):
                    print(f"🚨 Quarantined {tar_file.name}: no usable GPS coordinates")
                    remove_partial_outputs(tar_name, status_log)
            elif should_quarantine_location(exif_df):
                quarantined = True
                quarantine_reason = "within GPS quarantine radius"
                if move_tar_to_quarantine(tar_file, quarantine_near_point_dir / tar_file.name, status_log, quarantine_reason):
                    print(f"🚨 Quarantined {tar_file.name}: collected within GPS quarantine radius")
                    remove_partial_outputs(tar_name, status_log)

        # Step 9: classification + morphology.
        if not quarantined:
            start_time = time.time()
            if not csv_output.exists():
                if json_output.exists():
                    if TAR_STREAM_MORPHOLOGY:
                        extract_only_morphology_from_tar(
                            tar_file,
                            working_tar,
                            inventory["tif_images"],
                            csv_output,
                            inventory,
                        )
                    else:
                        if not extract_dir.exists():
                            ensure_full_extract_for_fallback(
                                working_tar,
                                extract_dir,
                                times_dict,
                                status_log,
                                "morphology fallback",
                            )
                        extract_only_morphology(tar_file, extract_dir, csv_output)
                    status_log.append("Image properties CSV created from existing predictions")
                else:
                    if TAR_STREAM_PREDICT:
                        classify_and_extract_regions_from_tar(
                            tar_file,
                            working_tar,
                            inventory["tif_images"],
                            inventory,
                            write_morphology=TAR_STREAM_MORPHOLOGY,
                        )
                        if not TAR_STREAM_MORPHOLOGY and not csv_output.exists():
                            ensure_full_extract_for_fallback(
                                working_tar,
                                extract_dir,
                                times_dict,
                                status_log,
                                "morphology after TAR-stream prediction",
                            )
                            extract_only_morphology(tar_file, extract_dir, csv_output)
                        status_log.append("Classification run from TAR stream; morphology handled by configured path")
                    else:
                        if not extract_dir.exists():
                            ensure_full_extract_for_fallback(
                                working_tar,
                                extract_dir,
                                times_dict,
                                status_log,
                                "classification fallback",
                            )
                        classify_and_extract_regions(tar_file, extract_dir)
                        status_log.append("Classification and morphology run together")
            elif not json_output.exists():
                if TAR_STREAM_PREDICT:
                    classify_and_extract_regions_from_tar(
                        tar_file,
                        working_tar,
                        inventory["tif_images"],
                        inventory,
                        write_morphology=False,
                    )
                    status_log.append("Prediction JSON created from TAR stream")
                else:
                    if not extract_dir.exists():
                        ensure_full_extract_for_fallback(
                            working_tar,
                            extract_dir,
                            times_dict,
                            status_log,
                            "prediction JSON fallback",
                        )
                    classify_and_extract_regions(tar_file, extract_dir)
                    status_log.append("Re-ran full classification due to missing JSON")
            else:
                status_log.append("Classification and morphology already exist (skipped)")
            times_dict["Classification and morphology extraction"] = track_time(start_time, "Classification and morphology extraction")

        # Step 10: taxon preview export.
        if not quarantined:
            start_time = time.time()
            try:
                if EXPORT_TAXON_PREVIEWS and json_output.exists() and extract_dir.exists():
                    export_summary = export_images_by_top1_taxon(
                        extract_dir=extract_dir,
                        json_path=json_output,
                        tar_name=tar_name,
                        export_root=taxon_export_root,
                    )
                    status_log.append(
                        f"Images exported by taxon ({export_summary['copied']} copied, {export_summary['taxa']} taxa; "
                        f"{export_summary['highgap_copied']} high-gap / {export_summary['lowgap_copied']} low-gap)"
                    )
                    if export_summary.get("validation_csv"):
                        status_log.append(f"Validation CSV written: {Path(export_summary['validation_csv']).name}")
                    if export_summary["missing"]:
                        status_log.append(f"⚠️ Taxon export missing source files: {export_summary['missing']}")
                elif EXPORT_TAXON_PREVIEWS and json_output.exists() and TAR_STREAM_EXPORT_TAXON_PREVIEWS:
                    export_summary = export_images_by_top1_taxon_from_tar(
                        tar_path=working_tar,
                        member_info=inventory.get("member_info", {}),
                        json_path=json_output,
                        tar_name=tar_name,
                        export_root=taxon_export_root,
                    )
                    status_log.append(
                        f"Images exported by taxon from TAR ({export_summary['copied']} copied, {export_summary['taxa']} taxa; "
                        f"{export_summary['highgap_copied']} high-gap / {export_summary['lowgap_copied']} low-gap)"
                    )
                    if export_summary.get("validation_csv"):
                        status_log.append(f"Validation CSV written: {Path(export_summary['validation_csv']).name}")
                    if export_summary["missing"]:
                        status_log.append(f"⚠️ Taxon export missing source files: {export_summary['missing']}")
                elif EXPORT_TAXON_PREVIEWS and json_output.exists():
                    status_log.append("Taxon image export skipped because full extraction was not available and TAR export disabled")
                else:
                    status_log.append("Taxon image export disabled or JSON missing")
            except Exception as export_err:
                print(f"⚠️ Taxon export failed for {tar_name}: {export_err}")
                status_log.append(f"⚠️ Taxon export failed: {export_err}")
            times_dict["Export images by taxon"] = track_time(start_time, "Export images by taxon")

        # Step 11: top species.
        if not quarantined:
            start_time = time.time()
            if not topspecies_csv.exists():
                if json_output.exists():
                    generate_topspecies_csv(json_output)
                    status_log.append("Top species CSV generated")
                else:
                    status_log.append("Top species CSV skipped (JSON not found)")
            else:
                status_log.append("Top species CSV already exists (skipped)")
            times_dict["Generate top species CSV"] = track_time(start_time, "Generate top species CSV")

        # Step 12: per-minute bio metrics.
        if not quarantined:
            start_time = time.time()
            try:
                log_per_minute_metrics(tar_name, json_output, hits_path, exif_df, source_dir, num_images)
                status_log.append("Per-minute bio metrics logged")
            except Exception as e:
                print(f"❌ Failed per-minute log for {tar_name}: {e}")
                status_log.append(f"❌ Failed per-minute bio metrics: {e}")
            times_dict["Per-minute bio metrics"] = track_time(start_time, "Per-minute bio metrics")

    except Exception as e:
        status_log.append(f"❌ Unexpected error: {e}")
        print(f"❌ Unexpected error while processing {tar_name}: {e}")

    finally:
        cleanup_start = time.time()
        try:
            if copied_working_tar and working_tar.exists():
                working_tar.unlink()
            if preview_dir.exists():
                schedule_delete_dir(preview_dir, async_cleanup=True)
            if extract_dir.exists():
                schedule_delete_dir(extract_dir, async_cleanup=True)
            clear_temp_output_dir(paths["output"])
        except Exception as cleanup_err:
            status_log.append(f"⚠️ Cleanup failed: {cleanup_err}")
        times_dict["Async cleanup scheduled"] = track_time(cleanup_start, "Async cleanup scheduled")

        daily_tar_reports.append({
            "tar_name": tar_name,
            "status_log": status_log,
            "quarantine_reason": quarantine_reason if quarantined else None,
        })

        if not already_logged:
            try:
                log_time_to_file(tar_name, times_dict, num_images)
            except Exception as log_err:
                print(f"⚠️ Failed to log times for {tar_name}: {log_err}")

        print(f"🔧🔧🔧 DONE {tar_name} 🔧🔧🔧")

        if all_required_outputs_exist(tar_name):
            try:
                done_marker = source_dir / f"{tar_name}.done"
                with open(done_marker, "w", encoding="utf-8") as f:
                    f.write(f"Processed at {datetime.datetime.now()}\n")
            except Exception as e:
                print(f"⚠️ Could not write done marker for {tar_name}: {e}")

# === WORKER LOCKS ===
LOCK_STALE_AFTER_SECONDS = max(0.0, STARTUP_ARGS.stale_lock_hours * 3600.0)
LOCK_HEARTBEAT_SECONDS = 300.0


def _lock_is_stale(lockfile):
    if LOCK_STALE_AFTER_SECONDS <= 0:
        return False

    try:
        age_seconds = time.time() - lockfile.stat().st_mtime
    except FileNotFoundError:
        return False

    return age_seconds > LOCK_STALE_AFTER_SECONDS


def acquire_tar_lock(tar_file):
    lockfile = source_dir / f"{tar_file.stem}.lock"

    if lockfile.exists() and _lock_is_stale(lockfile):
        try:
            lockfile.unlink()
            print(f"Removed stale lock: {lockfile.name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Could not remove stale lock {lockfile.name}: {e}")
            return None

    metadata = {
        "tar": tar_file.name,
        "worker": WORKER_LABEL,
        "pid": os.getpid(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }

    try:
        fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    return lockfile


def release_tar_lock(lockfile):
    if lockfile is None:
        return

    try:
        lockfile.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Could not remove lock {lockfile.name}: {e}")


def start_lock_heartbeat(lockfile):
    stop_event = threading.Event()

    if lockfile is None or LOCK_STALE_AFTER_SECONDS <= 0:
        return stop_event, None

    interval = min(LOCK_HEARTBEAT_SECONDS, max(60.0, LOCK_STALE_AFTER_SECONDS / 4.0))

    def heartbeat():
        while not stop_event.wait(interval):
            try:
                os.utime(lockfile, None)
            except FileNotFoundError:
                return
            except Exception as e:
                print(f"Could not refresh lock {lockfile.name}: {e}")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    return stop_event, thread


def stop_lock_heartbeat(stop_event, thread):
    stop_event.set()
    if thread is not None:
        thread.join(timeout=2.0)


def process_available_tars_once():
    new_files = get_new_tar_files(source_dir)
    pipeline_count = len(new_files)

    if pipeline_count == 0:
        print(f"[{time.ctime()}] No new .tar files.")
        return False

    print(f"[{time.ctime()}] {pipeline_count} .tar file(s) in the processing pipeline.")
    processed_any = False

    for tar_file in new_files:
        lockfile = acquire_tar_lock(tar_file)
        if lockfile is None:
            continue

        heartbeat_stop, heartbeat_thread = start_lock_heartbeat(lockfile)
        try:
            processed_any = True
            process_tar(tar_file)
        except Exception as e:
            print(f"Failed to process {tar_file.name}: {e}")
        finally:
            stop_lock_heartbeat(heartbeat_stop, heartbeat_thread)
            release_tar_lock(lockfile)

    if not processed_any:
        print(f"[{time.ctime()}] All queued .tar files are already locked by other workers.")

    return processed_any


# === CONTINUOUS WATCH ===
print("Watching for new .tar files (press Ctrl+C to stop)...")
print(f"Worker: {WORKER_LABEL}")
print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '(not set)')}")
print(f"Scratch work dir: {work_dir}")
while True:
    process_available_tars_once()
    if STARTUP_ARGS.once:
        break

    print(f"Rechecking in {STARTUP_ARGS.sleep_seconds} seconds...")
    time.sleep(STARTUP_ARGS.sleep_seconds)
