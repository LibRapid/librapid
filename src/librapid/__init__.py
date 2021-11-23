# -*- coding: utf-8 -*-

import os
import platform
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try appending to the package directory

try:
    from distutils.sysconfig import get_python_lib

    sys.path.append(os.path.join(get_python_lib(), "librapid"))
except:
    print("Was not able to append to system path")

try:
    try:
        from ._librapid import *
    except ModuleNotFoundError:
        print("Failed to load '._librapid' locally from {}".format(os.listdir(ROOT_DIR)))
        print("Attempting to load '_librapid' globally")
        from _librapid import *
except ImportError:
    print("Attempting to load librapid again, but using win32api to set the DLL directory")

    # If using windows, tell the os where the DLL for blas is
    # If blas was not installed, this doesn't really do anything
    if platform.system() == "Windows":
        import win32api

        if "openblas.dll" in os.listdir(ROOT_DIR) or \
                "libopenblas.dll" in os.listdir(ROOT_DIR):
            print("Loading DLL from './'")
            win32api.SetDllDirectory(ROOT_DIR)
            sys.path.append(ROOT_DIR)
        elif os.path.exists(os.path.join(ROOT_DIR, "blas")):
            print("Loading DLL from './blas'")
            win32api.SetDllDirectory(os.path.join(ROOT_DIR, "blas"))
            sys.path.append(os.path.join(ROOT_DIR, "blas"))

    try:
        try:
            from ._librapid import *
        except ModuleNotFoundError:
            print("Failed to load '._librapid' from {}".format(os.listdir(ROOT_DIR)))
            print("Attempting to load '_librapid' globally")
            from _librapid import *
    except ImportError:
        print("There was an error trying to load the librapid C++ module '_librapid'")

        if platform.system() == "Windows":
            print("This could be caused by a missing BLAS DLL file in the '{}' directory".format(ROOT_DIR))
            print("\n".join(os.listdir(ROOT_DIR)))

        raise ImportError("Could not import '_librapid'")
