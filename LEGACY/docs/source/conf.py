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
import shutil
import regex

# print("VERSION INFORMATION:")
# os.system("pip show sphinx")
# os.system("pip show breathe")
# os.system("pip show exhale")
# os.system("doxygen --version")

# Update the environment file based on the requirements.txt file
# Note: requirements.txt is automatically udpated by dependabot, while
# the environment file is not. This allows the updates to actually take
# effect
template = """
name: docs # doxygen
channels:
  - conda-forge
dependencies:
  - python=3.9
  - doxygen=1.9.1
  - pip
  - pip:
"""

# try:
#	 with open("../requirements.txt", "r") as requirements:
#		 template += "\n" + requirements.readline()
# except FileNotFoundError:
#	 template = """
# name: docs # doxygen
# channels:
#   - conda-forge
# dependencies:
#   - python=3.9
#   - doxygen=1.9.1
#   - pip
#   - pip:
#	   - sphinx==4.4.0
#	   - breathe==4.33.1
#	   - exhale==0.3.1
#	   - furo==2022.2.14.1
#	   - pydata-sphinx-theme==0.8.0
#	   - numpydoc==1.2
#	   - sphinx-panels==0.6.0
#	   - regex
# 
#	 """
# 
# try:
#	 with open("../environment.txt", "w") as environment:
#		 environment.write(template)
# except FileNotFoundError:
#	 

sys.path.insert(0, os.path.abspath("../.."))

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_dirs = [
	"pybind11",
	"version2",
	"jitify",
	"fmt",
	"blas",
	"python"
]

file_match = regex.compile(".*\..*")

# -- Copy LibRapid source directory ------------------------------------------
# This is to avoid documenting Jitify, PyBind11, VectorClass and BLAS
if os.path.exists("./librapid_doc_copy"):
	shutil.rmtree("./librapid_doc_copy")

for file in os.listdir("../../src/librapid"):
	if file in exclude_dirs: continue

	print(f"../../src/librapid/{file}")
	if file_match.match(file):
		shutil.copyfile(f"../../src/librapid/{file}", f"./librapid_doc_copy/{file}")
	else:
		shutil.copytree(f"../../src/librapid/{file}", f"./librapid_doc_copy/{file}")

# -- Project information -----------------------------------------------------

project = "librapid"
copyright = "2021, Toby Davis"
author = "Toby Davis"

# The full version, including alpha/beta/rc tags
version_file = open("./librapid_doc_copy/VERSION.hpp", "r")
release = version_file.readlines()[1].split()[2].replace("\"", "")
version_file.close()

# Set the master file
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
	# "furo", <= DO NOT comment back in -- it breaks everything
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

INPUT				 = ./librapid_doc_copy
	   
ENABLE_PREPROCESSING  = YES
MACRO_EXPANSION	   = YES
EXPAND_ONLY_PREDEF	= NO
PREDEFINED			+= LIBRAPID_DOXYGEN_BUILD
PREDEFINED			+= LR_INLINE=
PREDEFINED			+= __restrict=
PREDEFINED			+= LIBRAPID_MAX_DIMS=32
PREDEFINED			+= __host__=
PREDEFINED			+= __device__=
PREDEFINED			+= __global__=

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
	"verboseBuild": True
}

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
	"*pybind*",
	"*version2*",
	"*jitify*",
	"*fmt*",
	"*blas/*"
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "furo"
# html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_book_theme"

# html_logo = "../../branding/LibRapid small space.png"

html_context = {
	"github_user": "LibRapid",
	"github_repo": "librapid",
	"github_version": "master",
	"doc_path": "docs/source",
}

html_theme_options = {
	# Show a banner at the top of the page
	# "announcement": """
	#	 <a style=\"text-decoration: none; color: white;\" 
	#		href=\"https://www.patreon.com/librapid\">
	#		<img src=\"/en/latest/_static/LR_icon.png\"
	#		width=22.5/> Support LibRapid on Patreon
	#	 </a>
	# """,
	"sidebar_hide_name": True,
	"light_logo": "LibRapid_light.png",
	"dark_logo": "LibRapid_dark.png",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

html_favicon = "/en/latest/_static/LR_icon.png"
