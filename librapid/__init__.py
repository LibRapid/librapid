# coding: utf8

import os
import platform
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(1, ROOT_DIR)

# If using windows, tell the os where the DLL for blas is
# If blas was not installed, this doesn't really do anything
if platform.system() == "Windows":
	import win32api

	print("Loading DLL from", os.path.join(ROOT_DIR, "blas"))
	win32api.SetDllDirectory(os.path.join(ROOT_DIR, "blas"))

try:
	print("Attempting to load '.librapidcore' from {}".format(os.listdir(ROOT_DIR)))
	from .librapidcore import *
except ImportError:
	print("Failed to load '.librapid'. Attempting to load 'librapidcore' globally"))
	from librapidcore import *
