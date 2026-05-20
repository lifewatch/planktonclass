import os
from pathlib import Path
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Point the installed package to this repo's local config before importing planktonclass.
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["planktonclass_CONFIG"] = str(REPO_ROOT / "config.yaml")
os.environ["planktonclass_CONFIG"] = str(REPO_ROOT / "config.yaml")

import absl.logging
from tensorflow.keras.models import load_model

from planktonclass import paths as plk_paths, utils
from planktonclass.data_utils import load_class_names

absl.logging.set_verbosity(absl.logging.ERROR)

TIMESTAMP = "2025-10-09_140052-anasimyia"
MODEL_NAME = "final_model.h5"

plk_paths.timestamp = TIMESTAMP

class_names = load_class_names(splits_dir=plk_paths.get_ts_splits_dir())
model_path = os.path.join(plk_paths.get_checkpoints_dir(), MODEL_NAME)
conf_path = os.path.join(plk_paths.get_conf_dir(), "conf.json")

model = load_model(model_path, custom_objects=utils.get_custom_objects())

with open(conf_path) as f:
    conf = json.load(f)

print("Loaded model:", model_path)
print("Loaded config:", conf_path)
print("Number of classes:", len(class_names))
