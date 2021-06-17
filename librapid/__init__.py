import os
import platform
import distutils.sysconfig
import shutil

from . import progress

if platform.system() == "Windows":
	import win32api
	print("Adding DLL path", os.path.join(distutils.sysconfig.get_python_lib(), "librapid", "blas"))
	win32api.SetDllDirectory(os.path.join(distutils.sysconfig.get_python_lib(), "librapid", "blas"))

from librapid_ import *
