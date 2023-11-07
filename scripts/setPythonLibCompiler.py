import sys
import re
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-cc", "--ccompiler", type=str, help="C Compiler", required=True)
argParser.add_argument("-cxx", "--cxxcompiler", type=str, help="C++ Compiler", required=True)
argParser.add_argument("-o", "--output", type=str, help="Output File", default="../pyproject2.toml")

args = argParser.parse_args()

text = ""
with open("../pyproject.toml", "r") as pyproj:
    for line in pyproj.readlines():
        if line.startswith("cmake.args"):
            text += f'cmake.args = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PARALLEL_LEVEL=1", "-DCMAKE_C_COMPILER={args.ccompiler}", "-DCMAKE_CXX_COMPILER={args.cxxcompiler}"]\n'
        else:
            text += line

with open("../pyproject2.toml", "w") as pyproj:
    pyproj.write(text)
