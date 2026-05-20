"""
Command-line interface for planktonclass.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from importlib.resources import files

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import yaml

from planktonclass import config, model_utils, paths, runtime


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROJECT_CONFIG_NAME = "config.yaml"


def _resource_path(*parts):
    return os.fspath(files("planktonclass").joinpath(*parts))


DEFAULT_NOTEBOOKS_DIR = _resource_path("resources", "notebooks")
DEFAULT_DEMO_IMAGES_DIR = _resource_path("resources", "demo-images")
DEFAULT_DEMO_SPLITS_DIR = _resource_path("resources", "dataset_files")
DEFAULT_TRANSFORMATION_DATA_DIR = _resource_path("resources", "data_transformation")
PRETRAINED_MODEL_NAME = model_utils.DEFAULT_PRETRAINED_MODEL
TRAINED_MODEL_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def _default_config_path():
    cwd_config = os.path.abspath(DEFAULT_PROJECT_CONFIG_NAME)
    if os.path.exists(cwd_config):
        return cwd_config
    return config.DEFAULT_CONFIG_PATH


def _resolve_config_argument(config_path=None, target=None):
    if config_path:
        return os.path.abspath(config_path)

    if target:
        target = os.path.abspath(target)
        if os.path.isdir(target):
            candidate = os.path.join(target, DEFAULT_PROJECT_CONFIG_NAME)
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"No {DEFAULT_PROJECT_CONFIG_NAME} found in project directory: {target}"
                )
            return candidate
        if os.path.isfile(target):
            return target
        raise FileNotFoundError(f"Target does not exist: {target}")

    return _default_config_path()


def _apply_config(conf_path):
    config.set_config_path(conf_path)
    paths.CONF = config.get_conf_dict()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_placeholder(path, contents=""):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(contents)


def _copy_tree(src, dst):
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Missing resource directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _resolve_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Executable not found in PATH: {name}")
    return executable


def _resolve_project_dir(directory=None, conf_path=None):
    if conf_path:
        _apply_config(os.path.abspath(conf_path))
        return paths.get_base_dir()
    if directory is not None:
        return os.path.abspath(directory)
    return os.path.abspath(".")


def _display_path(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    except ValueError:
        return path


def _list_model_timestamps():
    models_dir = paths.get_models_dir()
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models directory found: {models_dir}")

    timestamps = sorted(
        [
            name
            for name in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, name))
            and TRAINED_MODEL_TIMESTAMP_PATTERN.match(name)
        ]
    )
    if not timestamps:
        raise FileNotFoundError(f"No models found in: {models_dir}")
    return timestamps


def _select_latest_trained_timestamp(models_dir):
    timestamps = sorted(
        name
        for name in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, name))
        and TRAINED_MODEL_TIMESTAMP_PATTERN.match(name)
    )
    if not timestamps:
        raise FileNotFoundError(f"No timestamped model runs found in: {models_dir}")
    return timestamps[-1]


def _resolve_model_timestamp(models_dir, explicit_timestamp=None):
    if explicit_timestamp:
        run_dir = os.path.join(models_dir, explicit_timestamp)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Model run not found: {run_dir}")
        return explicit_timestamp
    return _select_latest_trained_timestamp(models_dir)


def _resolve_checkpoint_name(run_dir, explicit_ckpt=None):
    ckpt_dir = os.path.join(run_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints directory found: {ckpt_dir}")

    available = sorted(
        name
        for name in os.listdir(ckpt_dir)
        if name.endswith((".keras", ".h5"))
    )
    if not available:
        raise FileNotFoundError(f"No supported checkpoints found in: {ckpt_dir}")

    if explicit_ckpt:
        if explicit_ckpt not in available:
            raise FileNotFoundError(
                f"Checkpoint {explicit_ckpt} not found in {ckpt_dir}. Available: {available}"
            )
        return explicit_ckpt

    for preferred in ("best_model.keras", "final_model.keras", "final_model.h5"):
        if preferred in available:
            return preferred

    return available[-1]


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return config.load_yaml_config(handle)


def _dump_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _copy_if_exists(src_root, dst_root, relative_path):
    src = os.path.join(src_root, relative_path)
    if os.path.exists(src):
        dst = os.path.join(dst_root, relative_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _prepare_docker_project_config(source_config_path, target_config_path, timestamp, ckpt_name):
    conf = _load_yaml(source_config_path)
    conf["general"]["base_directory"]["value"] = "."
    conf["general"]["images_directory"]["value"] = "."
    conf["testing"]["timestamp"]["value"] = timestamp
    conf["testing"]["ckpt_name"]["value"] = ckpt_name
    conf["testing"]["output_directory"]["value"] = "/tmp/planktonclass-predictions"
    _dump_yaml(target_config_path, conf)


def _copy_docker_build_context(context_dir, project_dir, timestamp, ckpt_name):
    package_files = [
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "VERSION",
        "MANIFEST.in",
        "Dockerfile",
    ]
    for relative_path in package_files:
        _copy_if_exists(PACKAGE_ROOT, context_dir, relative_path)

    package_src = os.path.join(PACKAGE_ROOT, "planktonclass")
    package_dst = os.path.join(context_dir, "planktonclass")
    shutil.copytree(package_src, package_dst, dirs_exist_ok=True)

    project_dst = os.path.join(context_dir, "project")
    os.makedirs(project_dst, exist_ok=True)

    _prepare_docker_project_config(
        os.path.join(project_dir, DEFAULT_PROJECT_CONFIG_NAME),
        os.path.join(project_dst, DEFAULT_PROJECT_CONFIG_NAME),
        timestamp=timestamp,
        ckpt_name=ckpt_name,
    )

    src_run_dir = os.path.join(project_dir, "models", timestamp)
    dst_models_dir = os.path.join(project_dst, "models")
    os.makedirs(dst_models_dir, exist_ok=True)
    shutil.copytree(
        src_run_dir,
        os.path.join(dst_models_dir, timestamp),
        dirs_exist_ok=True,
    )


def build_docker_image(args):
    project_dir = _resolve_project_dir(args.directory, args.config)
    models_dir = os.path.join(project_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models directory found: {models_dir}")

    timestamp = _resolve_model_timestamp(models_dir, args.timestamp)
    run_dir = os.path.join(models_dir, timestamp)
    ckpt_name = _resolve_checkpoint_name(run_dir, args.ckpt_name)
    image_tag = args.tag or f"planktonclass-inference:{timestamp.lower()}"
    docker_executable = _resolve_executable("docker")
    install_extras = "gpu" if getattr(args, "gpu", False) else ""
    base_image = args.base_image
    if getattr(args, "gpu", False) and base_image == "tensorflow/tensorflow:2.19.0":
        base_image = "tensorflow/tensorflow:2.19.0-gpu"

    with tempfile.TemporaryDirectory(prefix="planktonclass-docker-") as context_dir:
        _copy_docker_build_context(context_dir, project_dir, timestamp, ckpt_name)
        command = [
            docker_executable,
            "build",
            "--tag",
            image_tag,
            "--build-arg",
            f"base_image={base_image}",
            "--build-arg",
            f"install_extras={install_extras}",
            context_dir,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Docker build failed. "
                f"Project: {project_dir}, timestamp: {timestamp}, checkpoint: {ckpt_name}"
            )

    print(f"Docker image built successfully: {image_tag}")
    print(f"Model run: {timestamp}")
    print(f"Checkpoint: {ckpt_name}")
    print(f"Base image: {base_image}")
    if getattr(args, "gpu", False):
        print(f"Run with: docker run --gpus all -p 5000:5000 {image_tag}")
    else:
        print(f"Run with: docker run -p 5000:5000 {image_tag}")


def _choose_report_timestamp(explicit_timestamp=None):
    if explicit_timestamp:
        return explicit_timestamp

    timestamps = _list_model_timestamps()
    suggested = timestamps[-1]

    if len(timestamps) == 1:
        print(f"Only one model run found. Using: {suggested}")
        return suggested

    print("Available model runs:")
    for idx, timestamp in enumerate(timestamps, start=1):
        marker = " (suggested)" if timestamp == suggested else ""
        print(f"  {idx}. {timestamp}{marker}")

    print(f"Suggested most recent run: {suggested}")
    print("Press Enter to use the suggested run, or type a number from the list.")

    while True:
        selection = input("Report model selection: ").strip()
        if not selection:
            return suggested
        if selection.isdigit():
            choice = int(selection)
            if 1 <= choice <= len(timestamps):
                return timestamps[choice - 1]
        print(f"Please enter a number between 1 and {len(timestamps)}, or press Enter.")


def _choose_report_mode(explicit_mode=None):
    if explicit_mode:
        return explicit_mode

    print("Report detail level:")
    print("  1. quick (suggested) - core figures only")
    print("  2. full - also generates the threshold-based figures in results subfolders")
    print("Press Enter to use quick, or type 1 or 2.")

    while True:
        selection = input("Report mode selection: ").strip().lower()
        if selection in {"", "1", "quick"}:
            return "quick"
        if selection in {"2", "full"}:
            return "full"
        print("Please enter 1, 2, quick, full, or press Enter.")


def _choose_retrain_timestamp(explicit_timestamp=None):
    if explicit_timestamp:
        return explicit_timestamp

    timestamps = _list_model_timestamps()
    suggested = timestamps[-1]

    if len(timestamps) == 1:
        print(f"Only one previous model run found. Using: {suggested}")
        return suggested

    print("Available model runs to continue from:")
    for idx, timestamp in enumerate(timestamps, start=1):
        marker = " (suggested)" if timestamp == suggested else ""
        print(f"  {idx}. {timestamp}{marker}")

    print(f"Suggested most recent run: {suggested}")
    print("Press Enter to use the suggested run, or type a number from the list.")

    while True:
        selection = input("Retrain model selection: ").strip()
        if not selection:
            return suggested
        if selection.isdigit():
            choice = int(selection)
            if 1 <= choice <= len(timestamps):
                return timestamps[choice - 1]
        print(f"Please enter a number between 1 and {len(timestamps)}, or press Enter.")


def _resolve_retrain_target(config_arg=None, target_or_source=None, source=None):
    if source:
        return _resolve_config_argument(config_arg, target_or_source), source

    if target_or_source and not os.path.exists(target_or_source):
        return _resolve_config_argument(config_arg, None), target_or_source

    return _resolve_config_argument(config_arg, target_or_source), None


def init_project(args):
    target_dir = os.path.abspath(args.directory)
    config_path = os.path.join(target_dir, DEFAULT_PROJECT_CONFIG_NAME)

    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(
            f"{config_path} already exists. Use --force to overwrite it."
        )

    _ensure_dir(target_dir)
    _ensure_dir(os.path.join(target_dir, "data", "images"))
    _ensure_dir(os.path.join(target_dir, "data", "dataset_files"))
    _ensure_dir(os.path.join(target_dir, "models"))

    shutil.copyfile(config.DEFAULT_CONFIG_PATH, config_path)

    if args.demo:
        _copy_tree(DEFAULT_DEMO_IMAGES_DIR, os.path.join(target_dir, "data", "images"))
        _copy_tree(
            DEFAULT_DEMO_SPLITS_DIR,
            os.path.join(target_dir, "data", "dataset_files"),
        )
    else:
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "classes.txt"),
            "# one class name per line\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "train.txt"),
            "# relative/image/path.jpg\t0\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "val.txt"),
            "# relative/image/path.jpg\t0\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "test.txt"),
            "# relative/image/path.jpg\t0\n",
        )

    print(f"Initialized project at: {target_dir}")
    print(f"Config: {config_path}")
    print(f"Images: {os.path.join(target_dir, 'data', 'images')}")
    print(f"Dataset files: {os.path.join(target_dir, 'data', 'dataset_files')}")
    print(f"Models: {os.path.join(target_dir, 'models')}")
    if args.demo:
        print("Demo data copied into data/images and data/dataset_files.")


def validate_config(args):
    conf_path = _resolve_config_argument(args.config, getattr(args, "target", None))
    _apply_config(conf_path)
    print(f"Configuration OK: {config.CONF_PATH}")
    print(f"Base directory: {paths.get_base_dir()}")
    print(f"Images directory: {paths.get_images_dir()}")
    print(f"Splits directory: {paths.get_splits_dir()}")
    print(f"Models directory: {paths.get_models_dir()}")


def train_model(args):
    conf_path = _resolve_config_argument(args.config, getattr(args, "target", None))
    _apply_config(conf_path)

    from planktonclass.train_runfile import train_fn

    conf = config.get_conf_dict()
    conf["dataset"]["num_workers"] = args.workers
    if args.mode:
        conf["training"]["mode"] = args.mode
    if args.epochs is not None:
        conf["training"]["epochs"] = args.epochs
    if args.quick:
        conf["training"]["mode"] = "fast"
        conf["training"]["epochs"] = 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    train_fn(TIMESTAMP=timestamp, CONF=conf)


def retrain_model(args):
    conf_path, explicit_source = _resolve_retrain_target(
        config_arg=args.config,
        target_or_source=getattr(args, "target_or_source", None),
        source=getattr(args, "source", None),
    )
    _apply_config(conf_path)

    from planktonclass.train_runfile import train_fn

    conf = config.get_conf_dict()
    conf["dataset"]["num_workers"] = args.workers
    if args.mode:
        conf["training"]["mode"] = args.mode
    if args.epochs is not None:
        conf["training"]["epochs"] = args.epochs
    if args.quick:
        conf["training"]["mode"] = "fast"
        conf["training"]["epochs"] = 1

    models_dir = paths.get_models_dir()
    selected_timestamp = _choose_retrain_timestamp(explicit_source)
    selected_timestamp = _resolve_model_timestamp(models_dir, selected_timestamp)
    run_dir = os.path.join(models_dir, selected_timestamp)
    selected_ckpt = _resolve_checkpoint_name(run_dir, args.ckpt_name)

    conf["training"]["resume_from_timestamp"] = selected_timestamp
    conf["training"]["resume_from_ckpt_name"] = selected_ckpt

    print(f"Continuing training from: {selected_timestamp}")
    print(f"Checkpoint: {selected_ckpt}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    train_fn(TIMESTAMP=timestamp, CONF=conf)


def generate_report_cmd(args):
    conf_path = _resolve_config_argument(args.config, getattr(args, "target", None))
    _apply_config(conf_path)

    from planktonclass.report_utils import generate_report

    selected_timestamp = _choose_report_timestamp(args.timestamp)
    selected_mode = _choose_report_mode(args.mode)
    print("Starting report generation...")
    summary = generate_report(
        timestamp=selected_timestamp,
        mode=selected_mode,
        progress=lambda message: print(f"[report] {message}"),
    )
    print(f"Report generated for timestamp: {summary['timestamp']}")
    print(f"Mode: {summary['mode']}")
    print(f"Results: {_display_path(summary['results_dir'])}")
    print(f"Predictions: {_display_path(summary['predictions_file'])}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.3f}")
    print(f"Top-3 accuracy: {summary['top3_accuracy']:.3f}")
    print(f"Top-5 accuracy: {summary['top5_accuracy']:.3f}")
    print(f"Macro F1: {summary['macro_f1']:.3f}")
    print(f"Weighted F1: {summary['weighted_f1']:.3f}")


def run_api(args):
    conf_path = _resolve_config_argument(args.config, getattr(args, "target", None))
    env = os.environ.copy()
    env[config.CONFIG_ENV_VAR] = conf_path
    env["DEEPAAS_V2_MODEL"] = "planktonclass"

    command = [_resolve_executable("deepaas-run"), "--listen-ip", args.host]
    if args.port is not None:
        command.extend(["--listen-port", str(args.port)])

    completed = subprocess.run(command, env=env, check=False)
    raise SystemExit(completed.returncode)


def list_models(args):
    conf_path = _resolve_config_argument(args.config, getattr(args, "target", None))
    _apply_config(conf_path)

    models_dir = paths.get_models_dir()
    if not os.path.isdir(models_dir):
        print(f"No models directory found: {models_dir}")
        return

    entries = sorted(
        [
            name
            for name in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, name))
        ]
    )
    if not entries:
        print(f"No models found in: {models_dir}")
        return

    print(f"Models in {models_dir}:")
    for name in entries:
        if name in model_utils.PRETRAINED_MODELS:
            metadata = model_utils.get_pretrained_metadata(name)
            print(
                f"{name} | architecture={metadata['architecture']} | "
                f"version={metadata['version']} | checkpoint={metadata['checkpoint_name']}"
            )
        else:
            print(name)


def notebooks(args):
    project_dir = _resolve_project_dir(args.directory, args.config)
    target_dir = os.path.join(project_dir, "notebooks")
    transformation_dir = os.path.join(project_dir, "data", "data_transformation")
    _ensure_dir(project_dir)
    _ensure_dir(os.path.join(project_dir, "data"))

    if os.path.exists(target_dir) and not args.force:
        raise FileExistsError(
            f"{target_dir} already exists. Use --force to overwrite notebook files."
        )

    if os.path.exists(transformation_dir) and not args.force:
        print(
            f"Transformation data directory already exists: {transformation_dir}"
        )
        print("Use --force to refresh the packaged transformation data files.")

    _copy_tree(DEFAULT_NOTEBOOKS_DIR, target_dir)
    _copy_tree(DEFAULT_TRANSFORMATION_DATA_DIR, transformation_dir)
    print(f"Notebooks copied to: {target_dir}")
    print(f"Transformation data copied to: {transformation_dir}")


def download_pretrained(args):
    project_dir = _resolve_project_dir(args.directory, args.config)
    models_dir = os.path.join(project_dir, "models")
    selected_model = args.model or PRETRAINED_MODEL_NAME
    target_dir = model_utils.ensure_pretrained_model(
        models_dir,
        modelname=selected_model,
        version=args.version,
        force=args.force,
    )
    metadata = model_utils.get_pretrained_metadata(selected_model, args.version)
    print(f"Pretrained model available at: {target_dir}")
    print(f"Name: {metadata['name']}")
    print(f"Architecture: {metadata['architecture']}")
    print(f"Version: {metadata['version']}")
    print(f"Checkpoint: {metadata['checkpoint_name']}")


def doctor(args):
    print(runtime.format_doctor_report())


def build_parser():
    parser = argparse.ArgumentParser(prog="planktonclass")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create a local planktonclass project structure."
    )
    init_parser.add_argument("directory", nargs="?", default=".")
    init_parser.add_argument(
        "--demo",
        action="store_true",
        help="Populate the project with demo images and demo dataset files.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yaml in the target directory.",
    )
    init_parser.set_defaults(func=init_project)

    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a config file and print resolved paths."
    )
    validate_parser.add_argument(
        "target",
        nargs="?",
        help="Optional project directory or config.yaml path.",
    )
    validate_parser.add_argument("--config")
    validate_parser.set_defaults(func=validate_config)

    train_parser = subparsers.add_parser(
        "train", help="Train a model using a config file."
    )
    train_parser.add_argument(
        "target",
        nargs="?",
        help="Optional project directory or config.yaml path.",
    )
    train_parser.add_argument("--config")
    train_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataset preprocessing workers.",
    )
    train_parser.add_argument(
        "--mode",
        choices=["normal", "fast"],
        help="Override the training mode from the config file.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs from the config file.",
    )
    train_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test run: uses fast mode and 1 epoch.",
    )
    train_parser.set_defaults(func=train_model)

    retrain_parser = subparsers.add_parser(
        "retrain",
        help="Continue training from a previous local model run.",
    )
    retrain_parser.add_argument(
        "target_or_source",
        nargs="?",
        help="Optional project directory/config path, or a previous model timestamp when run from a project root.",
    )
    retrain_parser.add_argument(
        "source",
        nargs="?",
        help="Optional previous model timestamp to continue from.",
    )
    retrain_parser.add_argument("--config")
    retrain_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataset preprocessing workers.",
    )
    retrain_parser.add_argument(
        "--mode",
        choices=["normal", "fast"],
        help="Override the training mode from the config file.",
    )
    retrain_parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs from the config file.",
    )
    retrain_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test run: uses fast mode and 1 epoch.",
    )
    retrain_parser.add_argument(
        "--ckpt-name",
        help="Checkpoint name inside the selected previous run. Defaults to best_model.keras, then final_model.keras, then final_model.h5.",
    )
    retrain_parser.set_defaults(func=retrain_model)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate evaluation plots and metrics for a trained run.",
    )
    report_parser.add_argument(
        "target",
        nargs="?",
        help="Optional project directory or config.yaml path.",
    )
    report_parser.add_argument("--config")
    report_parser.add_argument(
        "--timestamp",
        help="Timestamped model directory to report on. Defaults to the latest run.",
    )
    report_parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        help="Report detail level. Quick skips the subfolder threshold plots.",
    )
    report_parser.set_defaults(func=generate_report_cmd)

    api_parser = subparsers.add_parser(
        "api", help="Launch the DEEPaaS API with a selected config file."
    )
    api_parser.add_argument(
        "target",
        nargs="?",
        help="Optional project directory or config.yaml path.",
    )
    api_parser.add_argument("--config")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=5000)
    api_parser.set_defaults(func=run_api)

    models_parser = subparsers.add_parser(
        "list-models", help="List models inside the configured models directory."
    )
    models_parser.add_argument(
        "target",
        nargs="?",
        help="Optional project directory or config.yaml path.",
    )
    models_parser.add_argument("--config")
    models_parser.set_defaults(func=list_models)

    notebooks_parser = subparsers.add_parser(
        "notebooks", help="Copy packaged notebooks into a project directory."
    )
    notebooks_parser.add_argument("directory", nargs="?", default=".")
    notebooks_parser.add_argument("--config")
    notebooks_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite notebook files in an existing notebooks directory.",
    )
    notebooks_parser.set_defaults(func=notebooks)

    pretrained_parser = subparsers.add_parser(
        "pretrained",
        help="Download a published pretrained model into a project models directory.",
    )
    pretrained_parser.add_argument("directory", nargs="?", default=".")
    pretrained_parser.add_argument("--config")
    pretrained_parser.add_argument(
        "--model",
        choices=model_utils.PRETRAINED_MODEL_CHOICES,
        default=PRETRAINED_MODEL_NAME,
        help="Published pretrained model to download.",
    )
    pretrained_parser.add_argument(
        "--version",
        default="latest",
        help="Published pretrained model version to download. Default: latest",
    )
    pretrained_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download into an existing pretrained model directory.",
    )
    pretrained_parser.set_defaults(func=download_pretrained)

    docker_parser = subparsers.add_parser(
        "docker",
        help="Build an inference Docker image for a trained model run.",
    )
    docker_parser.add_argument("directory", nargs="?", default=".")
    docker_parser.add_argument("--config")
    docker_parser.add_argument(
        "--timestamp",
        help="Timestamped model directory to package. Defaults to the latest trained run.",
    )
    docker_parser.add_argument(
        "--ckpt-name",
        help="Checkpoint name inside the selected model run. Defaults to best_model.keras, then final_model.keras, then final_model.h5.",
    )
    docker_parser.add_argument(
        "--tag",
        help="Docker image tag to build. Defaults to planktonclass-inference:<timestamp>.",
    )
    docker_parser.add_argument(
        "--base-image",
        default="tensorflow/tensorflow:2.19.0",
        help="Base Docker image to use for the inference container.",
    )
    docker_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Build a GPU-oriented inference image and print the matching `docker run --gpus all` command.",
    )
    docker_parser.set_defaults(func=build_docker_image)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect TensorFlow runtime visibility and suggest the right install path.",
    )
    doctor_parser.set_defaults(func=doctor)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
