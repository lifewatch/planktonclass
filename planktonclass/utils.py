"""
Miscellaneous utils

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification
"""

import logging
import os
import shutil
import subprocess
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from distutils.dir_util import copy_tree
from multiprocessing import Process

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.layers import BatchNormalization, Dense, InputLayer

from planktonclass import paths
from planktonclass.optimizers import customAdam, customAdamW, customSGD

# Configure logger
logger = logging.getLogger(__name__)
epoch_logger = logging.getLogger("planktonclass.epoch_metrics")


class CompatDense(Dense):
    """Dense layer compatibility shim for models saved with newer Keras configs."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


class CompatInputLayer(InputLayer):
    """InputLayer compatibility shim for older Keras loaders."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        batch_shape = config.pop("batch_shape", None)
        if batch_shape is not None and "batch_input_shape" not in config:
            config["batch_input_shape"] = batch_shape
        return super().from_config(config)


class CompatBatchNormalization(BatchNormalization):
    """BatchNormalization compatibility shim for newer Keras configs."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("synchronized", None)
        return super().from_config(config)


def create_dir_tree():
    """
    Create directory tree structure
    """
    dirs = paths.get_dirs()
    created_dirs = []
    for d in dirs.values():
        if not os.path.isdir(d):
            os.makedirs(d)
            created_dirs.append(d)

    if created_dirs:
        logger.info("[utils] Created %d dataset directories", len(created_dirs))
        return
    
    if created_dirs:
        logger.info("▌ Created %d dataset directories", len(created_dirs))
    for directory in dirs.values():
        if not os.path.isdir(directory):
            os.makedirs(directory)
            created_dirs.append(directory)

    if created_dirs:
        logger.info("[utils] Created %d dataset directories", len(created_dirs))


def progress_prefix(logger_name, tag):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp} - {logger_name} - INFO - {tag} "


def display_path(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    except ValueError:
        return path


class PrefixedProgressStream:
    """Prefix stdout progress-bar lines so tqdm/Keras output aligns with logs."""

    def __init__(self, logger_name, tag, stream):
        self.logger_name = logger_name
        self.tag = tag
        self.stream = stream
        self.at_line_start = True
        self.encoding = getattr(stream, "encoding", None)
        self.errors = getattr(stream, "errors", None)
        self.newlines = getattr(stream, "newlines", None)
        self.line_buffering = getattr(stream, "line_buffering", False)

    def write(self, text):
        if not text:
            return 0

        pieces = []
        for char in text:
            if self.at_line_start and char not in ("\n", "\r"):
                pieces.append(progress_prefix(self.logger_name, self.tag))
                self.at_line_start = False
            pieces.append(char)
            if char in ("\n", "\r"):
                self.at_line_start = True

        rendered = "".join(pieces)
        self.stream.write(rendered)
        return len(text)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def prefixed_stdout(logger_name, tag):
    original_stdout = sys.stdout
    try:
        sys.stdout = PrefixedProgressStream(logger_name, tag, original_stdout)
        yield sys.stdout
    finally:
        sys.stdout = original_stdout


def backup_splits():
    """
    Save the data splits used during training to the timestamped dir.
    """
    src = paths.get_splits_dir()
    dst = paths.get_ts_splits_dir()
    copy_tree(src, dst)


def get_custom_objects():
    return {
        "customSGD": customSGD,
        "customAdam": customAdam,
        "customAdamW": customAdamW,
        "BatchNormalization": CompatBatchNormalization,
        "Dense": CompatDense,
        "InputLayer": CompatInputLayer,
        "DTypePolicy": tf.keras.mixed_precision.Policy,
    }


class LR_scheduler(callbacks.LearningRateScheduler):
    """
    Custom callback to decay the learning rate.
    Schedule follows a 'step' decay.

    Reference
    ---------
    https://github.com/keras-team/keras/issues/898#issuecomment-285995644
    """

    def __init__(self, lr_decay=0.1, epoch_milestones=[]):
        self.lr_decay = lr_decay
        self.epoch_milestones = epoch_milestones
        super().__init__(schedule=self.schedule)

    def schedule(self, epoch):
        current_lr = K.eval(self.model.optimizer.learning_rate)
        if epoch in self.epoch_milestones:
            new_lr = current_lr * self.lr_decay
            logger.info("▌ Learning rate decayed to: %.2e", new_lr)
            logger.info("[train] Learning rate decayed to: %.2e", new_lr)
        else:
            new_lr = current_lr
        return new_lr


class LRHistory(callbacks.Callback):
    """
    Custom callback to save the learning rate history

    Reference
    ---------
    https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
    """

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs.update(
            {"lr": K.eval(self.model.optimizer.learning_rate).astype(np.float64)}
        )
        super().on_epoch_end(epoch, logs)


class EpochMetricsLogger(callbacks.Callback):
    """Write a concise epoch summary to the application log."""

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.best_monitor_name = None
        self.best_monitor_value = None

    def on_train_begin(self, logs=None):
        logger.info("[train] Epoch metrics will be written to training.log and epoch_metrics.csv.")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        duration_s = None
        if self.epoch_start_time is not None:
            duration_s = time.time() - self.epoch_start_time

        monitor_name = "val_accuracy" if "val_accuracy" in logs else "accuracy"
        monitor_value = logs.get(monitor_name)
        if monitor_value is not None and (
            self.best_monitor_value is None or monitor_value > self.best_monitor_value
        ):
            self.best_monitor_name = monitor_name
            self.best_monitor_value = monitor_value

        metric_parts = [f"epoch {epoch + 1:03d}"]
        for key in [
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
            "lr",
        ]:
            value = logs.get(key)
            if value is not None:
                metric_parts.append(f"{key}={value:.5f}")
        if duration_s is not None:
            metric_parts.append(f"time={duration_s:.1f}s")
        if self.best_monitor_name is not None:
            metric_parts.append(
                f"best_{self.best_monitor_name}={self.best_monitor_value:.5f}"
            )
        epoch_logger.info("[epoch] %s", " | ".join(metric_parts))

    def on_train_end(self, logs=None):
        if self.best_monitor_name is not None:
            epoch_logger.info(
                "[train] Best %s: %.5f",
                self.best_monitor_name,
                self.best_monitor_value,
            )


def launch_tensorboard(port, logdir, host="0.0.0.0"):  # nosec
    tensorboard_path = shutil.which("tensorboard")
    if tensorboard_path is None:
        raise RuntimeError("TensorBoard executable not found in PATH.")

    subprocess.call([
        tensorboard_path,
        "--logdir",
        "{}".format(logdir),
        "--port",
        "{}".format(port),
        "--host",
        "{}".format(host),
    ])


def get_callbacks(CONF, use_lr_decay=True):
    """
    Get a callback list to feed fit_generator.

    Parameters
    ----------
    CONF: dict

    Returns
    -------
    List of callbacks
    """
    calls = []

    calls.append(callbacks.TerminateOnNaN())
    calls.append(LRHistory())
    calls.append(EpochMetricsLogger())
    calls.append(
        callbacks.CSVLogger(
            os.path.join(paths.get_logs_dir(), "epoch_metrics.csv"),
            separator=",",
            append=False,
        )
    )

    if use_lr_decay:
        schedule_mode = CONF["training"].get("lr_schedule_mode", "step")
        if schedule_mode == "step":
            milestones = (
                np.array(CONF["training"]["lr_step_schedule"]) * CONF["training"]["epochs"]
            )
            milestones = milestones.astype(np.int64)
            calls.append(
                LR_scheduler(
                    lr_decay=CONF["training"]["lr_step_decay"],
                    epoch_milestones=milestones.tolist(),
                )
            )
        elif schedule_mode == "plateau":
            monitor_name = CONF["training"].get("lr_plateau_monitor", "val_loss")
            if monitor_name.startswith("val_") and not CONF["training"]["use_validation"]:
                monitor_name = "loss"
                logger.warning(
                    "[train] ReduceLROnPlateau monitor '%s' requires validation; falling back to '%s'.",
                    CONF["training"].get("lr_plateau_monitor", "val_loss"),
                    monitor_name,
                )

            calls.append(
                callbacks.ReduceLROnPlateau(
                    monitor=monitor_name,
                    factor=CONF["training"].get("lr_plateau_factor", 0.1),
                    patience=CONF["training"].get("lr_plateau_patience", 3),
                    min_delta=CONF["training"].get("lr_plateau_min_delta", 1e-4),
                    cooldown=CONF["training"].get("lr_plateau_cooldown", 0),
                    min_lr=CONF["training"].get("lr_plateau_min_lr", 1e-6),
                    mode=CONF["training"].get("lr_plateau_mode", "auto"),
                    verbose=1,
                )
            )

    if CONF["monitor"].get("use_tensorboard", False):
        calls.append(
            callbacks.TensorBoard(
                log_dir=paths.get_logs_dir(),
                write_graph=False,
                profile_batch=0,
            ))

        print(
            "Monitor your training in Tensorboard by executing the "
            "following comand on your console:"
        )
        print("    tensorboard --logdir={}".format(display_path(paths.get_logs_dir())))

        # Get the full path to the 'fuser' executable
        # fuser_path = shutil.which("fuser")
        port = os.getenv("monitorPORT", 6006)
        port = int(port) if len(str(port)) >= 4 else 6006


        try:
            if os.name != "nt":
                fuser_path = shutil.which("fuser")
                if fuser_path:
                    subprocess.run([fuser_path, "-k", f"{port}/tcp"])
        except Exception as e:
            print(f"Warning: Could not kill existing TensorBoard on port {port}. {e}")

        process = Process(
            target=launch_tensorboard,
            args=(port, paths.get_logs_dir()),
            daemon=True,
        )
        process.start()



    if (CONF["training"]["use_validation"]
            and CONF["training"]["use_early_stopping"]):
        calls.append(
            callbacks.EarlyStopping(patience=int(0.1 *
                                                 CONF["training"]["epochs"])))

    if CONF["training"]["ckpt_freq"] is not None:
        calls.append(
            callbacks.ModelCheckpoint(
                os.path.join(
                    paths.get_checkpoints_dir(),
                    "epoch-{epoch:02d}.hdf5",
                ),
                verbose=0,
                period=max(
                    1,
                    int(CONF["training"]["ckpt_freq"] *
                        CONF["training"]["epochs"]),
                ),
            ))

    if CONF["training"]["use_validation"]:
        best_model_path = os.path.join(paths.get_checkpoints_dir(), "best_model.keras")
        best_model_display_path = display_path(best_model_path)

        calls.append(
            callbacks.ModelCheckpoint(
                filepath=best_model_display_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                mode="max",
                verbose=1,
            )
        )

        print("Best model will be saved to:", best_model_display_path)

    if not calls:
        calls = None

    return calls
