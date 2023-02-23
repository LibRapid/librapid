# Configuration file for the Sphinx documentation builder

import os
import sys
import textwrap
import regex
import subprocess

sys.path.insert(0, os.path.abspath("../.."))

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_dirs = [
	"src/librapid/vendor",
	"src/librapid/blas",
	"src/librapid/cxxblas"
]

file_match = regex.compile(".*\..*")

# -- Project information -----------------------------------------------------

project = "librapid"
copyright = "2023, Toby Davis"
author = "Toby Davis"

# The full version, including alpha/beta/rc tags
currentMajorVersion = None
currentMinorVersion = None
currentPatchVersion = None

try:
	with open("../../version.txt") as versionFile:
		text = versionFile.read()
		currentMajorVersion = regex.search("MAJOR [0-9]+", text).group().split()[1]
		currentMinorVersion = regex.search("MINOR [0-9]+", text).group().split()[1]
		currentPatchVersion = regex.search("PATCH [0-9]+", text).group().split()[1]
	print(f"Current Version: v{currentMajorVersion}.{currentMinorVersion}.{currentPatchVersion}")
except Exception as e:
	print("[ ERROR ] Failed to read version.txt")
	print(e)
	sys.exit(1)

release = f"v{currentMajorVersion}.{currentMinorVersion}.{currentPatchVersion}"

# Set the master file
master_doc = "index"

# -- Run Doxygen -------------------------------------------------------------

try:
	# subprocess.run(["cd ../..", "doxygen html latex xml"])
	os.system("cd ../.. && doxygen")
except Exception as e:
	print("[ ERROR ] Failed to run doxygen")
	print(e)
	print("\n\nPlease run Doxygen yourself from the source directory")
	sys.exit(1)

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
	"sphinx.ext.mathjax",
	"sphinx.ext.doctest",
	"sphinx.ext.inheritance_diagram",
	"sphinx_panels",
	"breathe",
	"exhale",
	"numpydoc",
	"sphinx-favicon"
]

autosectionlabel_prefix_document = True

# Set up the breathe extension
breathe_projects = {
	"librapid": "../xml"
}

breathe_default_project = "librapid"

# # Custom inputs to doxygen generator
# doxygen_inputs = """
#
# INPUT				 = ./librapid_doc_copy
#
# ENABLE_PREPROCESSING  = YES
# MACRO_EXPANSION	   = YES
# EXPAND_ONLY_PREDEF	= NO
# PREDEFINED			+= LIBRAPID_DOXYGEN_BUILD
# PREDEFINED			+= LR_INLINE=
# PREDEFINED			+= __restrict=
# PREDEFINED			+= LIBRAPID_MAX_DIMS=32
# PREDEFINED			+= __host__=
# PREDEFINED			+= __device__=
# PREDEFINED			+= __global__=
#
# """

# Set up the exhale extension
exhale_args = {
	# These arguments are required
	"containmentFolder": "./api",
	"rootFileName": "index.rst",
	"rootFileTitle": "LibRapid",
	"createTreeView": True,
	"doxygenStripFromPath": "../..",
	"verboseBuild": True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
	"*jitify*",
	"*fmt*",
	"*blas/*"
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_title = 'LibRapid Docs'
html_theme = "furo"
# html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_book_theme"

html_logo = "../../branding/LibRapid small space.png"

html_context = {
	"github_user": "LibRapid",
	"github_repo": "librapid",
	"github_version": "master",
	"doc_path": "docs/source",
}

html_theme_options = {
	# Show a banner at the top of the page
	"announcement": """
		 <a style=\"text-decoration: none; color: white;\" 
			href=\"https://github.com/sponsors/Pencilcaseman\">
			<img src=\"/en/latest/_static/LR_icon.png\"
			width=22.5/> If you like LibRapid, please consider giving a small donation to help keep it alive!
		 </a>
	""",
	"sidebar_hide_name": True,
	"light_logo": "LibRapid_light.png",
	"dark_logo": "LibRapid_dark.png",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]
html_css_files = ['css/s4defs-roles.css']

# html_favicon = "/en/latest/_static/LR_icon.png"
favicons = [
	{
		"rel": "icon",
		"static-file": "LR_icon_128.png",
		"type": "image/png",
	},
	{
		"rel": "icon",
		"sizes": "16x16",
		"href": "LR_icon_128.png",
		"type": "image/png",
	},
	{
		"rel": "icon",
		"sizes": "32x32",
		"href": "LR_icon_128.png",
		"type": "image/png",
	},
	{
		"rel": "icon",
		"sizes": "64x64",
		"href": "LR_icon_128.png",
		"type": "image/png",
	},
	{
		"rel": "apple-touch-icon",
		"sizes": "180x180",
		"href": "LR_icon.png",
		"type": "image/png",
	},
]

rst_prolog = """
.. include:: <s5defs.txt>
"""
