import os
import sys

project = "Planktonclass"
author = "Wout Decrop"
copyright = "2026, Wout Decrop"
release = "0.2.0"

extensions = [
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = project

autosectionlabel_prefix_document = True

sys.path.insert(0, os.path.abspath(".."))
