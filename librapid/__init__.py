import os
import platform
import distutils.sysconfig
import shutil

from . import progress

# If using windows, tell the os where the DLL for blas is
# If blas was not installed, this doesn't really do anything
if platform.system() == "Windows":
	import win32api
	win32api.SetDllDirectory(os.path.join(distutils.sysconfig.get_python_lib(), "librapid", "blas"))

from librapid_ import *
