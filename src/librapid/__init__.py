# -*- coding: utf-8 -*-

import os
import platform
import sys
import importlib

ROOT_FILE = importlib.util.find_spec('librapid').origin
ROOT_DIR = os.path.dirname(ROOT_FILE)
sys.path.append(ROOT_DIR)

try:
    if platform.system() == "Windows":
        # Add a load of paths to the DLL search paths
        os.add_dll_directory(ROOT_DIR)

        if "CUDA_PATH" in os.environ:
            os.environ["CUDA_PATH"]
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
except:
    pass

from python.pythonInterface import *
