"""Setuptools hooks for keeping packaged resources in sync before a build."""

from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_TEMPLATE = PROJECT_ROOT / "config.yaml"
PACKAGED_CONFIG = PROJECT_ROOT / "planktonclass" / "resources" / "config.yaml"


def sync_packaged_resources():
    """Copy release templates from their repository source of truth."""
    copyfile(CONFIG_TEMPLATE, PACKAGED_CONFIG)


class build_py(_build_py):
    def run(self):
        sync_packaged_resources()
        super().run()


class sdist(_sdist):
    def run(self):
        sync_packaged_resources()
        super().run()


setup(cmdclass={"build_py": build_py, "sdist": sdist})
