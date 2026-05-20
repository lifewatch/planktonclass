"""
Project: VLIZ Pi-10 processing pipeline
Module:  VLIZ_Pi-10_processing.py

Metadata
--------
- Authors: Jonas Mortelmans <jonas.mortelmans@vliz.be>, Wout Decrop <wout.decrop@vliz.be>
- Created: 2025-10-03
- Updated: 2025-10-03
- Version: 1.0.0
- Documentation: Mortelmans J., Decrop W., Heynderickx H., Cattrijsse A., Depaepe M., Van Walraeven L., Scott J., Van Oevelen D., Deneudt K., Muniz C. (2025, submitted). High-throughput image classification and morphometry though the Pi-10 imaging pipeline
- Source: https://github.com/lifewatch/planktonclass/tree/PI10
"""

##to do: add startDate directly to bio-metrics
##to do : do save backgorunds , even it i move to quarantaine


# === LIBRARIES ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

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

PI10_ROOT = _resolve_path(
    _require_config_value(PATH_CONFIG, "pi10_root"),
    PREDICT_CONFIG_PATH.parent,
)

# === LOGGING ===
log_dir = _path_from_pi10_root(PATH_CONFIG, "log_dir", "not_processed", "GPU_ENVIRONMENT", "logs")
log_dir.mkdir(parents=True, exist_ok=True)

now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file_path = log_dir / f"processing_times_{now_time}.csv"

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
work_dir = _path_from_pi10_root(
    PATH_CONFIG,
    "work_dir",
    "not_processed",
    "GPU_ENVIRONMENT",
    "PI10_tempUntarred",
)
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
MAX_MISS_HIT_RATIO = float(VALIDATION_CONFIG.get("max_miss_hit_ratio", 100))
REQUIRED_HITSMISSES_ROWS = int(VALIDATION_CONFIG.get("required_hitsmisses_rows", 10))
GRAY_EDGE_SAMPLE_SIZE = int(VALIDATION_CONFIG.get("gray_edge_sample_size", 20))
GRAY_EDGE_FRACTION = float(VALIDATION_CONFIG.get("gray_edge_fraction", 0.01))
_gray_edge_mean_min = VALIDATION_CONFIG.get("gray_edge_mean_min", 150.0)
GRAY_EDGE_MEAN_MIN = (
    None if _gray_edge_mean_min in (None, "") else float(_gray_edge_mean_min)
)
QUARANTINE_LAT = float(GPS_QUARANTINE_CONFIG.get("latitude", 51.235293843807796))
QUARANTINE_LON = float(GPS_QUARANTINE_CONFIG.get("longitude", 2.9310864728604327))
QUARANTINE_RADIUS_M = float(GPS_QUARANTINE_CONFIG.get("radius_m", 50))

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
EMAIL_ENABLED = all([
    EMAIL_SETTINGS['smtp_server'],
    EMAIL_SETTINGS['smtp_port'],
    EMAIL_SETTINGS['sender_email'],
    EMAIL_SETTINGS['sender_password'],
    EMAIL_SETTINGS['recipients'],
])
if not EMAIL_ENABLED:
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
        if (now.hour == 11 and now.minute == 0
                and last_afternoon_summary_sent_day != current_date):
            send_daily_summary_email(now.strftime('%Y-%m-%d'), daily_tar_reports)
            last_afternoon_summary_sent_day = current_date

        time.sleep(60)  # check every minute

def send_daily_summary_email(summary_date, report_data):
    import math
    global source_dir

    subject = f"[PI10] Daily Summary - {summary_date}"
    all_tar_files = list(source_dir.glob("*.tar"))
    tar_stems = {tar.stem for tar in all_tar_files}

    # === Count current files ===
    current_counts = {
        "tar": len(all_tar_files),
        "gpstag": len(list(source_dir.glob("*_gpstag.csv"))),
        "predictions": len(list(source_dir.glob("*_predictions_relative.json"))),
        "image_props": len(list(source_dir.glob("*_image_properties.csv"))),
        "topspecies": len(list(source_dir.glob("*_topspecies.csv"))),
        "hitsmisses": len(list(source_dir.glob("*_hitsmisses.txt"))),
        "backgrounds": len(list(source_dir.glob("*_Background.tif"))),
    }

    # === Load yesterday's counts from file ===
    delta_file = Path("daily_tar_count.json")
    yesterday_counts = {k: 0 for k in current_counts}
    if delta_file.exists():
        try:
            with open(delta_file, 'r') as f:
                yesterday_counts.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Could not read yesterday's count: {e}")

    # === Calculate deltas ===
    deltas = {k: current_counts[k] - yesterday_counts.get(k, 0) for k in current_counts}
    tar_delta = deltas["tar"]

    # === Save today’s counts for tomorrow ===
    try:
        with open(delta_file, 'w') as f:
            json.dump(current_counts, f)
    except Exception as e:
        print(f"⚠️ Could not write today's count: {e}")

    # === Calculate to-do (missing output per TAR) ===
    required_outputs = {
        "_gpstag.csv": ("GPS data", 0.5),                   # 30 mins per file
        "_hitsmisses.txt": ("Hits/Misses", 10 / 3600),      # 10 sec per file
        "_Background.tif": ("Background.tif", 10 / 3600),   # 10 sec per file
        "_predictions_relative.json": ("Predictions (JSON)", 3),  # 3 hours per file
        "_image_properties.csv": ("Image Properties (CSV)", 0.5), # 30 mins per file
        "_topspecies.csv": ("Top Species CSV", 2 / 60),     # 2 mins per file
    }

    todo_counts = {}
    raw_time_estimations = {}
    formatted_time_estimations = {}

    for suffix, (label, per_file_hours) in required_outputs.items():
        count = sum(not (source_dir / f"{stem}{suffix}").exists() for stem in tar_stems)
        todo_counts[label] = count
        total_hours = count * per_file_hours
        raw_time_estimations[label] = total_hours

        # Format time
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

    # === Build email body ===
    body_lines = []
    body_lines.append(f"🆕 **{abs(tar_delta)} TARs {'extra' if tar_delta >= 0 else 'less'} compared to yesterday**")
    body_lines.append("=" * 60)
    body_lines.append("")

    body_lines.append(f"**Summary for {summary_date}**")
    body_lines.append(f"TARs entirely processed today: {len(report_data)}")
    body_lines.append("")

    body_lines.append("**Folder Totals vs Yesterday:**")
    for key, label in [
        ("tar", "TAR files"),
        ("gpstag", "GPS data (gpstag.csv)"),
        ("predictions", "Predictions (JSON)"),
        ("image_props", "Image Properties (CSV)"),
        ("topspecies", "Top Species CSV"),
        ("hitsmisses", "Hits/Misses TXT"),
        ("backgrounds", "Background.tif"),
    ]:
        delta = deltas[key]
        sign = "+" if delta >= 0 else "-"
        body_lines.append(f"- {label}: {current_counts[key]} ({sign}{abs(delta)})")

    body_lines.append("")
    body_lines.append("**To-do by output module (missing files):**")
    for label in todo_counts:
        count = todo_counts[label]
        formatted_time = formatted_time_estimations[label]
        body_lines.append(f"- {label}: {count} files missing ({formatted_time})")

    body_lines.append("")
    body_lines.append(f"**Total estimated processing time left:** {total_time_str}")

    # === Send email ===
    msg = MIMEText("\n".join(body_lines))
    msg['Subject'] = subject
    msg['From'] = EMAIL_SETTINGS['sender_email']
    msg['To'] = ", ".join(EMAIL_SETTINGS['recipients'])

    try:
        with smtplib.SMTP(EMAIL_SETTINGS['smtp_server'], EMAIL_SETTINGS['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_SETTINGS['sender_email'], EMAIL_SETTINGS['sender_password'])
            server.sendmail(msg['From'], EMAIL_SETTINGS['recipients'], msg.as_string())
        print(f"📧 Daily summary email sent for {summary_date}")
    except Exception as e:
        print(f"❌ Failed to send daily summary email: {e}")



#LOG TIME OF EACH STEP
def init_log_file():
    """Initialize the CSV log file with headers."""
    headers = [
        "TAR Name",
        "Copy TAR to working directory",
        "Untar",
        "Gray edge quarantine check",
        "Extract hitsmisses.txt",
        "Count images",
        "Create preview images",
        "Early preview classification",
        "Copy Background.tif",
        "Extract and save EXIF metadata",
        "Classification and morphology extraction",
        "Generate top species CSV",
        "Per-minute bio metrics",
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
        print("⚙ Log file already exists, appending new entries.")

# for the logfiles; these are the headers
step_names = [
    "Copy TAR to working directory",
    "Untar",
    "Gray edge quarantine check",
    "Extract hitsmisses.txt",
    "Count images",
    "Create preview images",
    "Early preview classification",
    "Copy Background.tif",
    "Extract and save EXIF metadata",
    "Classification and morphology extraction",
    "Generate top species CSV",
    "Per-minute bio metrics"
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
PREVIEW_HIGH_GAP_MARGIN = 0.995
PREVIEW_HIGH_GAP_LIMIT = 100
PREVIEW_LOW_GAP_LIMIT = 250
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
    'hitsmisses': work_dir / "data/hitsmisses"
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


def check_preview_class_distribution(preview_tifs, threshold=0.3):
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
                should_continue, reason = check_preview_class_distribution(preview_sample_tifs, threshold=0.9)
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


# === CONTINUOUS WATCH ===
print("⚙  Watching for new .tar files (press Ctrl+C to stop)...")
while True:
    now = datetime.datetime.now()
    today_str = now.strftime('%Y-%m-%d')

    new_files = get_new_tar_files(source_dir)
    pipeline_count = len(new_files)

    if pipeline_count == 0:
        print(f"[{time.ctime()}] No new .tar files. Sleeping...")
        time.sleep(3600)
    else:
        print(f"[{time.ctime()}] ✅ {pipeline_count} .tar file(s) in the processing pipeline.")
        for tar_file in new_files:
            lockfile = source_dir / f"{tar_file.stem}.lock"
            if lockfile.exists():
                continue
            try:
                lockfile.touch(exist_ok=False)
                process_tar(tar_file)
            except Exception as e:
                print(f"❌ Failed to process {tar_file.name}: {e}")
            finally:
                if lockfile.exists():
                    lockfile.unlink()
        print("🔁 Rechecking in 1 hour...")
        time.sleep(3600)
