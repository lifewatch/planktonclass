"""
TensorFlow runtime helpers shared by inference and training code.
"""

import os
import platform

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def _is_wsl():
    release = platform.release().lower()
    version = platform.version().lower()
    return "microsoft" in release or "microsoft" in version


def get_runtime_info():
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    system = platform.system()
    is_native_windows = system == "Windows" and not _is_wsl()
    tf_version = tf.__version__

    return {
        "tensorflow_version": tf_version,
        "system": system,
        "is_wsl": _is_wsl(),
        "is_native_windows": is_native_windows,
        "is_cuda_build": bool(tf.test.is_built_with_cuda()),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "gpu_available": bool(logical_gpus),
    }


def get_install_recommendation(info=None):
    if info is None:
        info = get_runtime_info()

    if info["gpu_available"]:
        return {
            "recommended_extra": "gpu",
            "platform_path": "gpu-ready",
            "message": "GPU is available for TensorFlow on this machine.",
        }

    if info["is_native_windows"]:
        return {
            "recommended_extra": "gpu",
            "platform_path": "windows-directml",
            "message": (
                "Native Windows should use Python 3.10 and "
                "`pip install \"planktonclass[gpu]\"` for the DirectML path."
            ),
        }

    if info["system"] == "Linux" or info["is_wsl"]:
        return {
            "recommended_extra": "gpu",
            "platform_path": "linux-cuda",
            "message": (
                "Linux/WSL2 should use `pip install \"planktonclass[gpu]\"` "
                "to pull the TensorFlow CUDA-enabled path."
            ),
        }

    return {
        "recommended_extra": None,
        "platform_path": "cpu-default",
        "message": "Use the default CPU install on this platform.",
    }


def configure_tensorflow_runtime():
    info = get_runtime_info()
    for gpu in info["physical_gpus"]:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            continue
    return get_runtime_info()


def _native_windows_tf_gpu_unsupported(info):
    version = info["tensorflow_version"]
    try:
        major, minor, *_ = [int(part) for part in version.split(".")]
    except ValueError:
        return False
    return (
        info["is_native_windows"]
        and not info["is_cuda_build"]
        and (major, minor) >= (2, 11)
    )


def format_runtime_summary(info=None, purpose="runtime"):
    if info is None:
        info = get_runtime_info()

    prefix = f"TensorFlow {purpose}"
    if info["gpu_available"]:
        gpu_names = ", ".join(gpu.name for gpu in info["logical_gpus"])
        return (
            f"{prefix}: GPU enabled ({gpu_names}) "
            f"[TF {info['tensorflow_version']}, CUDA build={info['is_cuda_build']}]"
        )

    if _native_windows_tf_gpu_unsupported(info):
        return (
            f"{prefix}: GPU unavailable because this is a native Windows TensorFlow "
            f"{info['tensorflow_version']} CPU-only build. "
            "TensorFlow 2.11+ does not provide native Windows GPU support."
        )

    return (
        f"{prefix}: GPU unavailable. "
        f"[TF {info['tensorflow_version']}, CUDA build={info['is_cuda_build']}]"
    )


def format_doctor_report(info=None):
    if info is None:
        info = get_runtime_info()

    recommendation = get_install_recommendation(info)
    lines = [
        f"System: {info['system']}",
        f"TensorFlow: {info['tensorflow_version']}",
        f"WSL: {info['is_wsl']}",
        f"Native Windows: {info['is_native_windows']}",
        f"CUDA build: {info['is_cuda_build']}",
        f"Physical GPUs: {len(info['physical_gpus'])}",
        f"Logical GPUs: {len(info['logical_gpus'])}",
        format_runtime_summary(info, purpose="runtime"),
        f"Recommendation: {recommendation['message']}",
    ]
    if recommendation["recommended_extra"]:
        lines.append(
            f"Suggested install: pip install \"planktonclass[{recommendation['recommended_extra']}]\""
        )
    return "\n".join(lines)
