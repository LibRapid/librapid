# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import textwrap

print("VERSION INFORMATION:")
os.system("pip show sphinx")
os.system("pip show breathe")
os.system("pip show exhale")
os.system("doxygen --version")

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "librapid"
copyright = "2021, Toby Davis"
author = "Toby Davis"

# The full version, including alpha/beta/rc tags
version_file = open("../../src/librapid/VERSION.hpp", "r")
release = version_file.readlines()[1].split()[2].replace("\"", "")
version_file.close()

# Set the master file
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "furo",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_panels",
    "breathe",
    "exhale",
    "numpydoc"
]

autosectionlabel_prefix_document = True

# Set up the breathe extension
breathe_projects = {
    "librapid": "./doxygenoutput/xml"
}

breathe_default_project = "librapid"

# Custom inputs to doxygen generator
doxygen_inputs = """

INPUT                 = ../../src/librapid
	   
ENABLE_PREPROCESSING  = YES
MACRO_EXPANSION       = YES
EXPAND_ONLY_PREDEF    = NO
PREDEFINED            += LR_INLINE=
PREDEFINED            += __restrict=
PREDEFINED            += LIBRAPID_MAX_DIMS=32
PREDEFINED            += __host__=
PREDEFINED            += __device__=
PREDEFINED            += __global__=

"""

# Set up the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder": "./api",
    "rootFileName": "index.rst",
    "rootFileTitle": "LibRapid",
    "doxygenStripFromPath": "..",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": textwrap.dedent(doxygen_inputs),
    "verboseBuild": False
}

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = "furo"
html_theme = "pydata_sphinx_theme"

html_logo = "../../branding/LibRapid small space.png"

html_context = {
    "github_user": "LibRapid",
    "github_repo": "librapid",
    "github_version": "master",
    "doc_path": "docs/source",
}

html_theme_options = {
    "logo_link": "https://librapid.readthedocs.io/en/latest/",
    "github_url": "https://github.com/LibRapid/librapid",
    "use_edit_page_button": True,
    "google_analytics_id": "G-LYHM6F499Q"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
