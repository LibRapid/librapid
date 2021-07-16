import sys
import os

def file_contents(path):
    with open(path, "r") as f:
        return f.readlines()

def find_append(contents, find, to_append, keep_tabs = False):
    res = contents.copy()
    inserted = 0

    for i in range(len(contents)):
        line = contents[i]

        if find in line:
            # Insert the thing
            tabs = 0
            spaces = 0
            if keep_tabs:
                for char in line:
                    if char == " ":
                        spaces += 1
                    elif char == "\t":
                        tabs += 1
                    else:
                        break
                tabs = tabs + spaces // 4

            print("Tabs:", tabs)

            for j in range(len(to_append)):
               res.insert(i + j + 1 + inserted, (" " * tabs * 4) + to_append[j] + "\n")
            inserted += len(to_append)

    return res 

args = sys.argv
if len(args) == 1:
    exit(0)
else:
    file = args[1]

# OpenBLAS command line options to include
append = ["-DNUM_THREADS=128", "-DBUFFERSIZE=25", "-DUSE_TLS=1"]

contents = file_contents(file)
new_contents = find_append(contents, "${}COMMON_OPTIONS{}".format("{", "}"), append, True)

for line in new_contents:
    print(line, end="")

print("Saving contents back to file")

with open(file, "w") as f:
    for line in new_contents:
        f.write(line)

print("File updated")
