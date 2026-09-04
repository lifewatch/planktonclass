"""
Miscellanous functions to handle models.

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification
"""

import json
import os
import logging
import tarfile
import tempfile
import zipfile
import shutil

import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.python.saved_model import (
    builder as saved_model_builder, )
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils import (
    predict_signature_def, )

from planktonclass import config, paths, utils

logger = logging.getLogger(__name__)

LEGACY_PRETRAINED_MODEL_NAME = "Phytoplankton_EfficientNetV2B0"
DEFAULT_PRETRAINED_MODEL = "FlowCam"
PRETRAINED_MODELS = {
    "FlowCam": {
        "architecture": "EfficientNetV2B0",
        "versions": {
            "latest": {
                "url": "https://zenodo.org/records/15269453/files/Phytoplankton_EfficientNetV2B0.tar.gz?download=1",
                "archive_type": "tar.gz",
                "source_dir_names": ["Phytoplankton_EfficientNetV2B0", "FlowCam"],
                "checkpoint_name": "final_model.h5",
            }
        },
    },
    "FlowCyto": {
        "architecture": "EfficientNetV2B0",
        "versions": {
            "latest": {
                "url": "https://zenodo.org/records/19709957/files/FlowCytoClassifier.zip?download=1",
                "archive_type": "zip",
                "source_dir_names": ["FlowCytoClassifier", "FlowCyto"],
                "checkpoint_name": "best_model.keras",
            }
        },
    },
    "PI10": {
        "architecture": "EfficientNetV2B0",
        "versions": {
            "latest": {
                "url": "https://zenodo.org/records/19663235/files/lifewatch/planktonclas-v1.0-PI10.zip?download=1",
                "archive_type": "zip",
                "source_dir_names": ["PI10", "planktonclass-v1.0-PI10"],
                "checkpoint_name": "best_model.keras",
            }
        },
    },
    "ZooScan": {
        "architecture": "EfficientNetV2B0",
        "versions": {
            "latest": {
                "url": "https://mda.vliz.be/download.php?file=VLIZ_00000321_697c8625a407e314649621",
                "archive_type": "zip",
                "source_dir_names": ["2026-01-12_103651", "ZooScan"],
                "checkpoint_name": "final_model.h5",
            }
        },
    },
}
PRETRAINED_MODEL_CHOICES = list(PRETRAINED_MODELS.keys())
PRETRAINED_MODEL_ALIASES = {
    LEGACY_PRETRAINED_MODEL_NAME: DEFAULT_PRETRAINED_MODEL,
}

model_modes = {
    "DenseNet121": "torch",
    "DenseNet169": "torch",
    "DenseNet201": "torch",
    "InceptionResNetV2": "tf",
    "EfficientNetV2B0": "tf",
    "InceptionV3": "tf",
    "MobileNet": "tf",
    "NASNetLarge": "tf",
    "NASNetMobile": "tf",
    "Xception": "tf",
    "ResNet50": "caffe",
    "VGG16": "caffe",
    "VGG19": "caffe",
    LEGACY_PRETRAINED_MODEL_NAME: "tf",
}


def _safe_extract_tar(archive_path, destination):
    destination = os.path.abspath(destination)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(destination, member.name))
            if os.path.commonpath([destination, member_path]) != destination:
                raise ValueError(f"Unsafe path found in tar archive: {member.name}")
            if member.isdir():
                os.makedirs(member_path, exist_ok=True)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            os.makedirs(os.path.dirname(member_path), exist_ok=True)
            with extracted, open(member_path, "wb") as dst:
                dst.write(extracted.read())


def _safe_extract_zip(archive_path, destination):
    destination = os.path.abspath(destination)
    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        for member in zip_handle.infolist():
            member_path = os.path.abspath(os.path.join(destination, member.filename))
            if os.path.commonpath([destination, member_path]) != destination:
                raise ValueError(f"Unsafe path found in zip archive: {member.filename}")
            if member.is_dir():
                os.makedirs(member_path, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(member_path), exist_ok=True)
            with zip_handle.open(member, "r") as src, open(member_path, "wb") as dst:
                dst.write(src.read())


def _resolve_pretrained_key(name):
    if not name:
        return DEFAULT_PRETRAINED_MODEL
    return PRETRAINED_MODEL_ALIASES.get(name, name)


def _get_pretrained_spec(name, version="latest"):
    resolved_name = _resolve_pretrained_key(name)
    if resolved_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model: {name}. Available: {sorted(PRETRAINED_MODELS)}"
        )
    versions = PRETRAINED_MODELS[resolved_name]["versions"]
    resolved_version = version or "latest"
    if resolved_version not in versions:
        raise ValueError(
            f"Unknown version {resolved_version!r} for pretrained model {resolved_name}. "
            f"Available: {sorted(versions)}"
        )
    return resolved_name, resolved_version, versions[resolved_version]


def get_pretrained_metadata(name, version="latest"):
    resolved_name, resolved_version, spec = _get_pretrained_spec(name, version)
    model_entry = PRETRAINED_MODELS[resolved_name]
    return {
        "name": resolved_name,
        "version": resolved_version,
        "architecture": model_entry["architecture"],
        "checkpoint_name": spec["checkpoint_name"],
        "url": spec["url"],
        "archive_type": spec["archive_type"],
    }


def _find_model_dir(extract_root, source_dir_names):
    extract_root = os.path.abspath(extract_root)

    for root, dirnames, _ in os.walk(extract_root):
        if os.path.basename(root) == "models":
            for source_dir_name in source_dir_names:
                candidate = os.path.join(root, source_dir_name)
                if os.path.isdir(candidate):
                    return candidate
        for source_dir_name in source_dir_names:
            if source_dir_name in dirnames:
                candidate = os.path.join(root, source_dir_name)
                ckpt_dir = os.path.join(candidate, "ckpts")
                if os.path.isdir(ckpt_dir):
                    return candidate

    candidates = []
    for root, dirnames, _ in os.walk(extract_root):
        for dirname in dirnames:
            candidate = os.path.join(root, dirname)
            ckpt_dir = os.path.join(candidate, "ckpts")
            if os.path.isdir(ckpt_dir):
                candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not locate a single model directory with a ckpts folder in the downloaded archive."
    )


def ensure_pretrained_model(models_dir, modelname=DEFAULT_PRETRAINED_MODEL, version="latest", force=False):
    """
    Ensure a published pretrained model exists locally under the project models directory.
    """
    resolved_name, resolved_version, spec = _get_pretrained_spec(modelname, version)
    target_dir = os.path.join(models_dir, resolved_name)
    os.makedirs(models_dir, exist_ok=True)

    if os.path.isdir(target_dir) and not force:
        return target_dir

    logger.info("Downloading pretrained model: %s (%s)", resolved_name, resolved_version)
    logger.info("Source: %s", spec["url"])

    temp_archive = None
    extract_dir = None
    try:
        with requests.get(spec["url"], stream=True, timeout=120) as response:
            response.raise_for_status()
            suffix = ".zip" if spec["archive_type"] == "zip" else ".tar.gz"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                temp_archive = tmp_file.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp_file.write(chunk)

        extract_dir = tempfile.mkdtemp(prefix="planktonclass-pretrained-")
        if spec["archive_type"] == "zip":
            _safe_extract_zip(temp_archive, extract_dir)
        else:
            _safe_extract_tar(temp_archive, extract_dir)

        source_dir = _find_model_dir(extract_dir, spec["source_dir_names"])
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
    finally:
        if temp_archive and os.path.exists(temp_archive):
            os.remove(temp_archive)
        if extract_dir and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Pretrained model download completed but expected directory was not created: {target_dir}"
        )

    logger.info("Pretrained model available at: %s", target_dir)
    return target_dir


def resolve_pretrained_selection(conf):
    pretrained_conf = conf.get("pretrained", {})
    if pretrained_conf.get("use_pretrained"):
        return _resolve_pretrained_key(pretrained_conf.get("name")), pretrained_conf.get("version", "latest")

    modelname = conf.get("model", {}).get("modelname")
    if modelname in PRETRAINED_MODEL_ALIASES:
        return _resolve_pretrained_key(modelname), "latest"
    if modelname in PRETRAINED_MODELS:
        return modelname, "latest"
    return None, None


def resolve_resume_selection(conf):
    training_conf = conf.get("training", {})
    resume_timestamp = training_conf.get("resume_from_timestamp")
    resume_ckpt_name = training_conf.get("resume_from_ckpt_name")
    if resume_timestamp:
        return resume_timestamp, resume_ckpt_name
    return None, None


def _find_nested_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer is not model:
            return layer
    return model


def _load_resume_model(models_dir, timestamp, ckpt_name=None):
    run_dir = os.path.join(models_dir, timestamp)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Model run not found: {run_dir}")

    ckpt_dir = os.path.join(run_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints directory found: {ckpt_dir}")

    available = sorted(
        name for name in os.listdir(ckpt_dir) if name.endswith((".keras", ".h5"))
    )
    if not available:
        raise FileNotFoundError(f"No supported checkpoints found in: {ckpt_dir}")

    selected_ckpt = ckpt_name
    if selected_ckpt:
        if selected_ckpt not in available:
            raise FileNotFoundError(
                f"Checkpoint {selected_ckpt} not found in {ckpt_dir}. Available: {available}"
            )
    else:
        for preferred in ("best_model.keras", "final_model.keras", "final_model.h5"):
            if preferred in available:
                selected_ckpt = preferred
                break
        if selected_ckpt is None:
            selected_ckpt = available[-1]

    model_path = os.path.join(ckpt_dir, selected_ckpt)
    logger.info("▌ Resuming training from checkpoint: %s/%s", timestamp, selected_ckpt)
    model = utils.load_model_compat(
        model_path,
        custom_objects=utils.get_custom_objects(),
        compile=False,
    )
    base_model = _find_nested_base_model(model)
    return model, base_model, {
        "source": "resume",
        "timestamp": timestamp,
        "checkpoint_name": selected_ckpt,
    }


def create_model(CONF):
    """
    Parameters
    ----------
    CONF : dict
        Contains relevant configuration parameters of the model
    """
    modelname = CONF["model"]["modelname"]
    pretrained_name, pretrained_version = resolve_pretrained_selection(CONF)
    resume_timestamp, resume_ckpt_name = resolve_resume_selection(CONF)

    models_dir = paths.get_models_dir()
    if resume_timestamp:
        model, base_model, model_info = _load_resume_model(
            models_dir,
            timestamp=resume_timestamp,
            ckpt_name=resume_ckpt_name,
        )

        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        output_units = output_shape[-1]
        expected_units = CONF["model"]["num_classes"]
        if expected_units is not None and output_units != expected_units:
            raise ValueError(
                "Resume checkpoint output size does not match the active dataset. "
                f"Checkpoint classes: {output_units}, config classes: {expected_units}."
            )
        return model, base_model, model_info

    local_model_name = pretrained_name or modelname
    if pretrained_name:
        ensure_pretrained_model(models_dir, modelname=pretrained_name, version=pretrained_version)
    local_model_path = os.path.join(models_dir, local_model_name)

    # Try to load from local directory first
    if os.path.isdir(local_model_path):
        logger.info("✓ Found local model directory: %s", local_model_name)
        # Look for inference checkpoint in the model directory
        ckpt_dir = os.path.join(local_model_path, "ckpts")

        ckpt_files = sorted(
            [
                f for f in os.listdir(ckpt_dir)
                if f.endswith((".keras", ".h5"))
            ]
        )

        if not ckpt_files:
            raise FileNotFoundError(f"No .keras or .h5 model found in {ckpt_dir}")

        keras_files = [f for f in ckpt_files if f.endswith(".keras")]
        selected_file = keras_files[-1] if keras_files else ckpt_files[-1]
        model_file = os.path.join(ckpt_dir, selected_file)
        logger.info("▌ Loading model checkpoint: %s", os.path.basename(model_file))
        base_model = utils.load_model_compat(
            model_file, custom_objects=utils.get_custom_objects()
        )
        logger.debug("✓ Model output shape: %s", base_model.output_shape)

        new_input = base_model.input
        x = Dense(CONF["model"]["num_classes"], activation="softmax", name="new_dense")(base_model.layers[-1].output)
        model = Model(inputs=new_input, outputs=x)
        # Add L2 regularization for all the layers in the whole model
        if CONF["training"]["l2_reg"]:
            for layer in model.layers:
                layer.kernel_regularizer = regularizers.l2(
                    CONF["training"]["l2_reg"])

        return model, base_model, {
            "source": "local_model_dir",
            "model_name": local_model_name,
            "checkpoint_name": os.path.basename(model_file),
        }
    
    # Fall back to tf.keras.applications if no local model found
    architecture_name = modelname
    if pretrained_name:
        architecture_name = PRETRAINED_MODELS[pretrained_name]["architecture"]
    architecture = getattr(applications, architecture_name)

    # create the base pre-trained model
    img_width, img_height = (
        CONF["model"]["image_size"],
        CONF["model"]["image_size"],
    )
    base_model = architecture(
        weights="imagenet",
        include_top=False,
        input_shape=(img_width, img_height, 3),
    )

    # Add custom layers at the top to adapt it to our problem
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x) #might work better on large dataset than
    # GlobalAveragePooling https://github.com/keras-team/keras/issues/8470
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(CONF["model"]["num_classes"], activation="softmax")(x)

    # Full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Add L2 reguralization for all the layers in the whole model
    if CONF["training"]["l2_reg"]:
        for layer in model.layers:
            layer.kernel_regularizer = regularizers.l2(
                CONF["training"]["l2_reg"])

    return model, base_model, {
        "source": "keras_applications",
        "model_name": architecture_name,
    }



def save_conf(conf):
    """
    Save CONF to a txt file to ease the reading and to a json file to ease the parsing.

    Parameters
    ----------
    conf : 1-level nested dict
    """
    save_dir = paths.get_conf_dir()

    # Save dict as json file
    with open(os.path.join(save_dir, "conf.json"), "w") as outfile:
        json.dump(conf, outfile, sort_keys=True, indent=4)

    # Save dict as txt file for easier redability
    txt_file = open(os.path.join(save_dir, "conf.txt"), "w")
    txt_file.write("{:<25}{:<30}{:<30} \n".format("group", "key", "value"))
    txt_file.write("=" * 75 + "\n")
    for key, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            txt_file.write("{:<25}{:<30}{:<15} \n".format(
                key, g_key, str(g_val)))
        txt_file.write("-" * 75 + "\n")
    txt_file.close()

