import os
from pathlib import Path
import planktonclass.paths as plk_paths

# =========================
# ROOT = remove PI10 folder mistake
# =========================
BASE = Path.cwd().resolve()

# if you're inside PI10/, go one level up
if BASE.name == "PI10":
    BASE = BASE.parent

# now BASE should be: .../planktonclass
print("\n🔎 BASE FIXED:", BASE)

TIMESTAMP = "2025-10-09_140052-anasimyia"
model_path = BASE / "models" / TIMESTAMP

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

print("\n✅ EXISTS CHECK")
for p in [
    plk_paths.get_ts_splits_dir(),
    plk_paths.get_checkpoints_dir(),
    plk_paths.get_conf_dir()
]:
    print(p, "->", Path(p).exists())