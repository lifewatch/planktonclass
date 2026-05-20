"""
Training runfile

Date: September 2023
Last updated: March 2026
Author: Wout Decrop (based on code from Ignacio Heredia)
Email: wout.decrop@VLIZ.be
Github: ai4os-hub / phyto-plankton-classification

Description:
This file contains the commands for training a convolutional net for image
classification for phytoplankton.

Additional notes:
* On the training routine: Preliminary tests show that using a custom lr
  multiplier for the lower layers yield to better results than freezing them at
  the beginning and unfreezing them after a few epochs like it is suggested in
  the Keras tutorials.
"""

import io
import json
import logging
import os
import sys
import time
from datetime import datetime
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# Configure warnings before importing TensorFlow/Keras.
from planktonclass import warnings_config

warnings_config.configure_warnings()

import tensorflow as tf

from planktonclass import config, model_utils, paths, utils
from planktonclass.runtime import configure_tensorflow_runtime, format_runtime_summary
from planktonclass.data_utils import (
    compute_classweights,
    compute_meanRGB,
    create_data_splits,
    data_sequence,
    json_friendly,
    k_crop_data_sequence,
    load_aphia_ids,
    load_class_names,
    load_data_splits,
    split_file_has_entries,
)
from planktonclass.optimizers import customAdam
from planktonclass import test_utils

# TODO: Add additional metrics for test time in addition to accuracy

# from planktonclass.api import load_inference_model

# Set TensorFlow verbosity logs
tf.get_logger().setLevel(logging.ERROR)

# Configure logger for training
logger = logging.getLogger("planktonclass.train_runfile")
logger.setLevel(logging.INFO)

def log_section(title):
    # line = "=" * 70
    # logger.info(line)
    logger.info("[train] %s", title)
    # logger.info(line)


def log_step(message, *args):
    logger.info("[train] " + message, *args)


def display_path(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    except ValueError:
        return path


def prediction_display_path(path):
    try:
        return os.path.relpath(path, paths.get_base_dir()).replace("\\", "/")
    except ValueError:
        return display_path(path)


def get_preferred_testing_checkpoint(conf):
    """Return the checkpoint that should be used for testing/inference outputs."""
    use_validation = conf["training"]["use_validation"]
    best_model_name = "best_model.keras"
    best_model_path = os.path.join(paths.get_checkpoints_dir(), best_model_name)

    if use_validation and os.path.exists(best_model_path):
        return best_model_name
    return "final_model.keras"


def should_save_final_model(conf):
    """Return whether a final-model export should be written after training."""
    use_validation = conf["training"]["use_validation"]
    return not use_validation


def _safe_metric(metric_fn, true_lab, pred_top1, labels, average):
    kwargs = {"average": average, "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = labels
    return float(metric_fn(true_lab, pred_top1, **kwargs))


def _build_test_metrics_summary(true_lab, pred_lab, class_names):
    true_lab = np.array(true_lab, dtype=int)
    pred_lab = np.array(pred_lab, dtype=int)
    pred_top1 = pred_lab[:, 0]
    labels = list(range(len(class_names)))
    max_k = min(5, pred_lab.shape[1])

    topk = {
        f"top{k}_accuracy": float(test_utils.topK_accuracy(true_lab, pred_lab, K=k))
        for k in range(1, max_k + 1)
    }

    recall = {
        "micro": _safe_metric(recall_score, true_lab, pred_top1, labels, "micro"),
        "macro": _safe_metric(recall_score, true_lab, pred_top1, labels, "macro"),
        "macro_no_labels": _safe_metric(
            recall_score, true_lab, pred_top1, None, "macro"
        ),
        "weighted": _safe_metric(
            recall_score, true_lab, pred_top1, labels, "weighted"
        ),
    }
    precision = {
        "micro": _safe_metric(
            precision_score, true_lab, pred_top1, labels, "micro"
        ),
        "macro": _safe_metric(
            precision_score, true_lab, pred_top1, labels, "macro"
        ),
        "macro_no_labels": _safe_metric(
            precision_score, true_lab, pred_top1, None, "macro"
        ),
        "weighted": _safe_metric(
            precision_score, true_lab, pred_top1, labels, "weighted"
        ),
    }
    f1 = {
        "micro": _safe_metric(f1_score, true_lab, pred_top1, labels, "micro"),
        "macro": _safe_metric(f1_score, true_lab, pred_top1, labels, "macro"),
        "macro_no_labels": _safe_metric(f1_score, true_lab, pred_top1, None, "macro"),
        "weighted": _safe_metric(f1_score, true_lab, pred_top1, labels, "weighted"),
    }

    return {
        "num_samples": int(len(true_lab)),
        "num_classes": int(len(class_names)),
        "topk_accuracy": topk,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
    }


def train_fn(TIMESTAMP, CONF):

    paths.timestamp = TIMESTAMP
    paths.CONF = CONF

    utils.create_dir_tree()
    run_log_path = os.path.join(paths.get_logs_dir(), "training.log")
    warnings_config.attach_file_handler(run_log_path)
    utils.backup_splits()
    log_step("Writing run log to: %s", display_path(run_log_path))
    log_step("%s", format_runtime_summary(configure_tensorflow_runtime(), purpose="training"))

    if not split_file_has_entries(paths.get_ts_splits_dir(), split_name="train"):
        if not CONF["dataset"]["split_ratios"]:
            if CONF["training"]["use_validation"] & CONF["training"]["use_test"]:
                split_ratios = [0.8, 0.1, 0.1]
            elif CONF["training"]["use_validation"] & ~CONF["training"]["use_test"]:
                split_ratios = [0.9, 0.1, 0]
            else:
                split_ratios = [1, 0, 0]
        else:
            split_ratios = CONF["dataset"]["split_ratios"]

        log_section("Preparing dataset splits")
        log_step("Split ratios: %s", split_ratios)
        create_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_ratios=split_ratios,
        )
    else:
        log_section("Using existing dataset splits")

    log_section("Loading training data")
    log_step("Splits directory: %s", display_path(paths.get_ts_splits_dir()))
    log_step("Images directory: %s", display_path(paths.get_images_dir()))
    X_train, y_train = load_data_splits(
        splits_dir=paths.get_ts_splits_dir(),
        im_dir=paths.get_images_dir(),
        split_name="train",
    )

    if (
        CONF["training"]["use_validation"]
        and split_file_has_entries(paths.get_ts_splits_dir(), split_name="val")
    ):
        X_val, y_val = load_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_name="val",
        )
    else:
        logger.warning(
            "[train] No validation split found; continuing without validation."
        )
        X_val, y_val = None, None
        CONF["training"]["use_validation"] = False

    class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())
    aphia_ids = load_aphia_ids(splits_dir=paths.get_ts_splits_dir())

    CONF["model"]["preprocess_mode"] = model_utils.model_modes[
        CONF["model"]["modelname"]
    ]
    CONF["training"]["batch_size"] = min(
        CONF["training"]["batch_size"], len(X_train)
    )

    if CONF["model"]["num_classes"] is None:
        CONF["model"]["num_classes"] = len(class_names)

    if CONF["training"]["use_class_weights"]:
        log_section("Computing class weights")
        class_weights = compute_classweights(
            y_train, max_dim=CONF["model"]["num_classes"])
    else:
        class_weights = None

    if CONF["dataset"]["mean_RGB"] is None:
        log_section("Computing dataset statistics")
        CONF["dataset"]["mean_RGB"], CONF["dataset"]["std_RGB"] = compute_meanRGB(
            X_train,
            workers=CONF.get("dataset", {}).get("num_workers", 4)
        )

    train_gen = data_sequence(
        X_train,
        y_train,
        batch_size=CONF["training"]["batch_size"],
        num_classes=CONF["model"]["num_classes"],
        im_size=CONF["model"]["image_size"],
        mean_RGB=CONF["dataset"]["mean_RGB"],
        std_RGB=CONF["dataset"]["std_RGB"],
        preprocess_mode=CONF["model"]["preprocess_mode"],
        aug_params=CONF["augmentation"]["train_mode"],
    )
    train_steps = int(np.ceil(len(X_train) / CONF["training"]["batch_size"]))

    if CONF["training"]["use_validation"]:
        val_gen = data_sequence(
            X_val,
            y_val,
            batch_size=CONF["training"]["batch_size"],
            num_classes=CONF["model"]["num_classes"],
            im_size=CONF["model"]["image_size"],
            mean_RGB=CONF["dataset"]["mean_RGB"],
            std_RGB=CONF["dataset"]["std_RGB"],
            preprocess_mode=CONF["model"]["preprocess_mode"],
            aug_params=CONF["augmentation"]["val_mode"],
        )
        val_steps = int(np.ceil(len(X_val) / CONF["training"]["batch_size"]))
    else:
        val_gen = None
        val_steps = None

    t0 = time.time()

    log_section("Building model")
    model, base_model, model_info = model_utils.create_model(CONF)

    if model_info.get("source") == "resume":
        log_step(
            "Continuing from previous run: %s (%s)",
            model_info["timestamp"],
            model_info["checkpoint_name"],
        )

    base_vars = [var.name for var in base_model.trainable_variables]
    all_vars = [var.name for var in model.trainable_variables]
    top_vars = list(set(all_vars) - set(base_vars))

    if CONF["training"]["mode"] == "fast":
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(
        optimizer=customAdam(
            learning_rate=CONF["training"]["initial_lr"],
            amsgrad=True,
            lr_mult=0.1,
            excluded_vars=top_vars,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log_section("Starting training")
    log_step(
        "Epochs: %s | Batch size: %s | Training samples: %s | Validation samples: %s",
        CONF["training"]["epochs"],
        CONF["training"]["batch_size"],
        len(X_train),
        0 if X_val is None else len(X_val),
    )
    with utils.prefixed_stdout("planktonclass.train_runfile", "[train]"):
        history = model.fit(
            x=train_gen,
            steps_per_epoch=train_steps,
            epochs=CONF["training"]["epochs"],
            class_weight=class_weights,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=utils.get_callbacks(CONF),
            verbose=1,
            initial_epoch=0,
        )

    log_section("Training complete")
    log_step("Saving to: %s", display_path(paths.get_timestamped_dir()))
    log_step("Saving training statistics")
    stats = {
        "epoch": history.epoch,
        "training time (s)": round(time.time() - t0, 2),
        "timestamp": TIMESTAMP,
    }
    stats.update(history.history)
    stats = json_friendly(stats)
    stats_dir = paths.get_stats_dir()
    with open(os.path.join(stats_dir, "stats.json"), "w") as outfile:
        json.dump(stats, outfile, sort_keys=True, indent=4)

    if should_save_final_model(CONF):
        log_step("Saving final model")
        fpath = os.path.join(paths.get_checkpoints_dir(), "final_model.keras")

        stderr_backup = sys.stderr
        sys.stderr = io.StringIO()
        try:
            model.save(fpath, include_optimizer=False)
        finally:
            sys.stderr = stderr_backup
    else:
        log_step(
            "Skipping final model export because validation is enabled and best_model.keras is used."
        )

    preferred_ckpt_name = get_preferred_testing_checkpoint(CONF)
    CONF["testing"]["timestamp"] = TIMESTAMP
    CONF["testing"]["ckpt_name"] = preferred_ckpt_name

    log_step("Saving configuration")
    model_utils.save_conf(CONF)
    log_step("Default testing checkpoint: %s", preferred_ckpt_name)

    logger.info("[train] Training finished successfully.")

    if CONF["training"]["use_test"]:
        log_section("Evaluating test split")
        if preferred_ckpt_name != "final_model.keras":
            preferred_ckpt_path = os.path.join(
                paths.get_checkpoints_dir(), preferred_ckpt_name
            )
            log_step(
                "Reloading preferred checkpoint for test evaluation: %s",
                display_path(preferred_ckpt_path),
            )
            model = tf.keras.models.load_model(
                preferred_ckpt_path,
                custom_objects=utils.get_custom_objects(),
                compile=False,
            )

        X_test, y_test = load_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_name="test",
        )
        crop_num = 10
        filemode = "local"
        test_gen = k_crop_data_sequence(
            inputs=X_test,
            im_size=CONF["model"]["image_size"],
            mean_RGB=CONF["dataset"]["mean_RGB"],
            std_RGB=CONF["dataset"]["std_RGB"],
            preprocess_mode=CONF["model"]["preprocess_mode"],
            aug_params=CONF["augmentation"]["val_mode"],
            crop_mode="random",
            crop_number=crop_num,
            filemode=filemode,
        )
        top_K = 5

        with utils.prefixed_stdout("planktonclass.train_runfile", "[train]"):
            output = model.predict(
                test_gen,
                verbose=1,
            )

        output = output.reshape(len(X_test), -1, output.shape[-1])
        output = np.mean(output, axis=1)

        lab = np.argsort(output, axis=1)[:, ::-1]
        lab = lab[:, :top_K]
        prob = output[
            np.repeat(np.arange(len(lab)), lab.shape[1]),
            lab.flatten(),
        ].reshape(lab.shape)

        pred_lab, pred_prob = lab, prob

        if aphia_ids is not None:
            pred_aphia_ids = [aphia_ids[i] for i in pred_lab]
            pred_aphia_ids = [aphia_id.tolist() for aphia_id in pred_aphia_ids]
        else:
            pred_aphia_ids = aphia_ids

        class_index_map = {
            index: class_name for index, class_name in enumerate(class_names)
        }

        pred_lab_names = [
            [class_index_map[label] for label in labels] for labels in pred_lab
        ]
        y_test_names = [class_index_map.get(index) for index in y_test]

        pred_dict = {
            "filenames": [prediction_display_path(path) for path in X_test],
            "pred_lab": pred_lab.tolist(),
            "pred_prob": pred_prob.tolist(),
            "pred_lab_names": pred_lab_names,
            "aphia_ids": pred_aphia_ids,
        }
        if y_test is not None:
            pred_dict["true_lab"] = y_test.tolist()
            pred_dict["true_lab_names"] = y_test_names

        pred_path = os.path.join(
            paths.get_predictions_dir(),
            "{}+{}+top{}.json".format(preferred_ckpt_name, "DS_split", top_K),
        )
        with open(pred_path, "w") as outfile:
            json.dump(pred_dict, outfile, sort_keys=True)
        logger.info("[train] Predictions saved to: %s", display_path(pred_path))

        metrics_summary = _build_test_metrics_summary(
            true_lab=y_test,
            pred_lab=pred_lab,
            class_names=class_names,
        )
        metrics_summary.update(
            {
                "timestamp": TIMESTAMP,
                "checkpoint": preferred_ckpt_name,
                "predictions_file": display_path(pred_path),
            }
        )
        metrics_path = os.path.join(
            paths.get_predictions_dir(),
            "{}+{}+metrics.json".format(preferred_ckpt_name, "DS_split"),
        )
        with open(metrics_path, "w") as outfile:
            json.dump(metrics_summary, outfile, sort_keys=True, indent=2)
        logger.info("[train] Test metrics saved to: %s", display_path(metrics_path))
        logger.info("[train] Test set evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phytoplankton CNN")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of multiprocessing workers (use 1 for Jupyter)"
    )
    args = parser.parse_args()

    CONF = config.get_conf_dict()
    CONF["dataset"]["num_workers"] = args.workers  # store in CONF
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    train_fn(TIMESTAMP=timestamp, CONF=CONF)
