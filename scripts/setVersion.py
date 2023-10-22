import sys
"""

This script accepts a version number in the form X.Y.Z, where X, Y, and Z are integers.
It will then update all LibRapid version numbers to the specified version.

"""

import sys
import re
import argparse
from datetime import datetime

# Extract current version number
currentMajorVersion = None
currentMinorVersion = None
currentPatchVersion = None

try:
    with open("../version.txt", "r") as versionFile:
        text = versionFile.read()
        currentMajorVersion = re.search("MAJOR [0-9]+", text).group().split()[1]
        currentMinorVersion = re.search("MINOR [0-9]+", text).group().split()[1]
        currentPatchVersion = re.search("PATCH [0-9]+", text).group().split()[1]
    print(f"Current Version: v{currentMajorVersion}.{currentMinorVersion}.{currentPatchVersion}")
except Exception as e:
    print("[ ERROR ] Failed to read version.txt")
    print(e)
    sys.exit(1)

argParser = argparse.ArgumentParser()
argParser.add_argument("-v", "--version", type=str, help="Full Version")
argParser.add_argument("-M", "--major", type=int, help="Major Version")
argParser.add_argument("-m", "--minor", type=int, help="Minor Version")
argParser.add_argument("-p", "--patch", type=int, help="Patch Version")

args = argParser.parse_args()

if args.version and any([args.major, args.minor, args.patch]):
    print("[ ERROR ] -v and -M options cannot be used together")
    sys.exit(1)

if args.version:
    # Validate version number
    if not re.match("[0-9]+\\.[0-9]+\\.[0-9]+", args.version):
        print("[ ERROR ] Invalid version number")
        sys.exit(1)
    newMajorVersion = args.version.split(".")[0]
    newMinorVersion = args.version.split(".")[1]
    newPatchVersion = args.version.split(".")[2]
else:
    newMajorVersion = args.major if args.major else currentMajorVersion
    newMinorVersion = args.minor if args.minor else currentMinorVersion
    newPatchVersion = args.patch if args.patch else str(int(currentPatchVersion) + 1)
    
print(f"New Version: v{newMajorVersion}.{newMinorVersion}.{newPatchVersion}\n\n")

# Write to version.txt
with open("../version.txt", "w") as versionFile:
    versionFile.write(f"MAJOR {newMajorVersion}\n")
    versionFile.write(f"MINOR {newMinorVersion}\n")
    versionFile.write(f"PATCH {newPatchVersion}\n")
    print("Written to version.txt")

# Write to Doxyfile
template = None
with open("tmp/doxyTemplate", "r") as templateFile:
    template = templateFile.read()
    print("Loaded Doxyfile template")

with open("../Doxyfile", "w") as doxyfile:
    versionString = f"PROJECT_NUMBER         = v{newMajorVersion}.{newMinorVersion}.{newPatchVersion}"
    template = template.replace("$${{ INSERT_VERSION_NUMBER_HERE }}$$", versionString)
    doxyfile.write(template)
    print("Written to Doxyfile")

# Write to CITATION.cff
with open("tmp/citationTemplate.cff", "r") as templateFile:
    template = templateFile.read()
    print("Loaded CITATION.cff template")

with open("../CITATION.cff", "w") as citationFile:
    versionString = f"version: {newMajorVersion}.{newMinorVersion}.{newPatchVersion}"
    dateString = f"date-released: \"{datetime.now().strftime('%Y-%m-%d')}\""
    template = template.replace("$${{ INSERT_VERSION_NUMBER_HERE }}$$", versionString)
    template = template.replace("$${{ INSERT_DATE_HERE }}$$", dateString)
    citationFile.write(template)
    print("Written to CITATION.cff")

# Write to pyproject.toml
with open("tmp/pyprojectTemplate.toml", "r") as templateFile:
    template = templateFile.read()
    print("Loaded pyproject.toml template")

with open("../pyproject.toml", "w") as pyprojectFile:
    versionString = f"{newMajorVersion}.{newMinorVersion}.{newPatchVersion}"
    template = template.replace("$${{ INSERT_VERSION_NUMBER_HERE }}$$", versionString)
    pyprojectFile.write(template)
    print("Written to pyproject.toml")
