import os
import platform

# Add this to the PATH value on Windows to enable loading the BLAS DLL
# (if BLAS was not found, this doesn't do anything, so there's no point checking)
if platform.system() == "Windows":
    os.environ["PATH"] = os.path.join(os.getcwd(), "librapid/blas") + os.pathsep + os.environ["PATH"]

from . import progress
