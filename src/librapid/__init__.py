# coding: utf8

import os
import platform
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(1, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, os.pardir))

# If using windows, tell the os where the DLL for blas is
# If blas was not installed, this doesn't really do anything
if platform.system() == "Windows":
	import win32api

	print("Loading DLL from", os.path.join(ROOT_DIR, "blas"))
	win32api.SetDllDirectory(os.path.join(ROOT_DIR, "blas"))
	sys.path.append(os.path.join(ROOT_DIR, "blas"))

try:
	try:
		print("Attempting to load '.librapidcore' from {}".format(os.listdir(ROOT_DIR)))
		from ._librapid import *
	except ModuleNotFoundError:
		print("Failed to load '.librapid'. Attempting to load 'librapidcore' globally")
		from _librapid import *
except ImportError:
	print("There was an error trying to load the librapid C++ module '_librapid'.")
	
	if platform.system() == "Windows":
		print("This could be caused by a missing DLL file in the 'librapid/blas' directory")
		print("\n".join(os.listdir(ROOT_DIR, "blas")))
