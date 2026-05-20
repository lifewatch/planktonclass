from planktonclass import runtime


def test_runtime_sets_tensorflow_log_level_env(monkeypatch):
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "2")
    assert runtime.os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"


def test_runtime_summary_reports_native_windows_cpu_only(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.platform, "release", lambda: "10")
    monkeypatch.setattr(runtime.platform, "version", lambda: "10.0.26100")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(runtime.tf, "__version__", "2.19.0")
    monkeypatch.setattr(runtime.tf.test, "is_built_with_cuda", lambda: False)
    monkeypatch.setattr(runtime.tf.config, "list_physical_devices", lambda kind: [])
    monkeypatch.setattr(runtime.tf.config, "list_logical_devices", lambda kind: [])

    summary = runtime.format_runtime_summary(runtime.get_runtime_info(), purpose="prediction")

    assert "native Windows TensorFlow 2.19.0 CPU-only build" in summary
    assert "TensorFlow 2.11+ does not provide native Windows GPU support" in summary


def test_runtime_summary_reports_visible_gpu(monkeypatch):
    class FakeGpu:
        name = "GPU:0"

    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.setattr(runtime.platform, "version", lambda: "WSL2")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.11.9")
    monkeypatch.setattr(runtime.tf, "__version__", "2.19.0")
    monkeypatch.setattr(runtime.tf.test, "is_built_with_cuda", lambda: True)
    monkeypatch.setattr(runtime.tf.config, "list_physical_devices", lambda kind: [FakeGpu()] if kind == "GPU" else [])
    monkeypatch.setattr(runtime.tf.config, "list_logical_devices", lambda kind: [FakeGpu()] if kind == "GPU" else [])

    summary = runtime.format_runtime_summary(runtime.get_runtime_info(), purpose="prediction")

    assert "GPU enabled" in summary
    assert "GPU:0" in summary


def test_doctor_report_recommends_gpu_extra_for_linux(monkeypatch):
    class FakeGpu:
        name = "GPU:0"

    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(runtime.platform, "version", lambda: "generic")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.10.14")
    monkeypatch.setattr(runtime.tf, "__version__", "2.19.0")
    monkeypatch.setattr(runtime.tf.test, "is_built_with_cuda", lambda: True)
    monkeypatch.setattr(runtime.tf.config, "list_physical_devices", lambda kind: [FakeGpu()] if kind == "GPU" else [])
    monkeypatch.setattr(runtime.tf.config, "list_logical_devices", lambda kind: [FakeGpu()] if kind == "GPU" else [])

    report = runtime.format_doctor_report(runtime.get_runtime_info())

    assert "Python: 3.10.14" in report
    assert "Suggested install: pip install \"planktonclass[gpu]\"" in report
    assert "Supported Python versions: 3.10, 3.11, 3.12" in report


def test_doctor_report_recommends_directml_path_for_windows(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.platform, "release", lambda: "10")
    monkeypatch.setattr(runtime.platform, "version", lambda: "10.0.26100")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(runtime.tf, "__version__", "2.19.0")
    monkeypatch.setattr(runtime.tf.test, "is_built_with_cuda", lambda: False)
    monkeypatch.setattr(runtime.tf.config, "list_physical_devices", lambda kind: [])
    monkeypatch.setattr(runtime.tf.config, "list_logical_devices", lambda kind: [])

    report = runtime.format_doctor_report(runtime.get_runtime_info())

    assert "Python: 3.12.3" in report
    assert "Python 3.10" in report
    assert "planktonclass[gpu]" in report
