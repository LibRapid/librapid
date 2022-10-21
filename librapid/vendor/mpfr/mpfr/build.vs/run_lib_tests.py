#
# Python script for running MPFR tests
#
# Run this from the build.vc11 directory

import sys, os, shutil, subprocess, filecmp

test = 'lib'
test_dir = test + '_mpfr_tests\\'
lib_loc  = '..\\' + test + '\\'
lib_name = '\\mpfr.' + test

cw, f = os.path.split(__file__)
os.chdir(cw)

def write_f(ipath, opath):
  if os.path.exists(ipath) and not os.path.isdir(ipath):
    if os.path.exists(opath) and os.path.isfile(opath) and filecmp.cmp(ipath, opath):
      return
    dp, f = os.path.split(opath)
    try:
      os.mkdir(dp)
    except FileExistsError:
      pass
    shutil.copy2(ipath, opath)

def copy_dir(src, dst):
  os.makedirs(dst, exist_ok=True)
  for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
      copy_dir(s, d)
    else:
      write_f(s, d)

# get a list of tests from the user
def get_input(n):
  li = []
  while True:
    s = "list the numbers of those you wish to test [1 - " + str(n) +"] ? "
    if sys.version[0] == '3':
      r = input(s)
    else:
      r = raw_input(s)
    tl = []
    if len(r) == 0:
      return [i for i in range(len(li))]
    if r.find(',') != -1:
      tl = [u for u in r.split(',')]
    elif r.find(' ') != -1:
      tl = [u for u in r.split(' ')]
    else:
      tl = [r]
    for it in tl:
      try:
        ind = int(it)
      except:
        print("you can only input the numbers of listed items")
        break
      if ind > 0 and ind <= n:
        li += [ind - 1]
      else:
        print(ind, "is not in list")
        break
    if len(li):
      return li

test_ext = ["Win32\\Release", "x64\\Release", "Win32\\Debug", "x64\\Debug" ]

# find libraries that have been built and list them in order of creation date
# for selection by the user

li = []
for x in test_ext:
  p = lib_loc + x + lib_name
  print(p)
  if os.path.exists(p):
    s = os.stat(p)
    li += [[s.st_mtime, x]]

if len(li) > 1:
  li.sort(key = lambda x : x[0])
  for i in range(len(li)):
    print("  ", i + 1, end='')
    if not i:
      print(": (old)", end='')
    elif i == len(li) - 1:
      print(": (new)", end='')
    else:
      print(":      ", end='')
    print(li[i][1])
  nl = get_input(len(li))
else:
  nl = [0]

# copy any required data into the test directory

shutil.copy( "..\\tests\\inp_str.dat", test_dir )
shutil.copy( "..\\tests\\tfpif_r1.dat", test_dir )
shutil.copy( "..\\tests\\tfpif_r2.dat", test_dir )
shutil.copy( "..\\tests\\tmul.dat", test_dir )
if os.path.exists("..\\tests\\data\\"):
  if not os.path.exists(test_dir + "\\data\\"):
    shutil.copytree("..\\tests\\data\\", test_dir + "\\data\\")
  else:
    copy_dir("..\\tests\\data\\", test_dir + "\\data\\")

# generate list of projects from *.vcproj files

prj_list = []
for p, d, f in os.walk(test_dir):
  for ff in f:
    x = os.path.splitext(ff)
    if x[1] == '.vcxproj' and x[0] != 'lib_tests':
      prj_list += [x[0]]
prj_list.sort()

# go through each library to be tested

for k in nl:
  tpos = test_dir + li[k][1] + "\\"
  print("Testing",  lib_name, "in", li[k][1], ":")

  if "dll" in lib_loc:
    # copy gmp.dll and mpfr.dll into the test directory
    gpos = "..\\..\\mpir\\dll\\" + li[k][1] + "\\mpir.dll"
    mpos = lib_loc + li[k][1] + lib_name
    shutil.copy(gpos, tpos)
    shutil.copy(mpos, tpos)

  # list the *.exe files in the test directory
  exe_list = []
  try:
    l = os.listdir(tpos)
  except:
    print("Tests have not been built for this configuration")
    os._exit(-1)

  for f in l:
    x = os.path.splitext(f)
    if x[1] == '.exe':
      exe_list += [x[0]]
  if len(exe_list) == 0:
    print("No executable test files for this configuration")
    os._exit(-1)

  # now test projects that have been built and the build failures
  build_fail = 0
  run_ok = 0
  run_fail = 0
  skipped = 0
  for i in prj_list:
    if i in exe_list:
      ef = test_dir + li[k][1] + '\\' + i + '.exe'
      prc = subprocess.Popen( ef, stdout = subprocess.PIPE, cwd = test_dir,
        stderr = subprocess.STDOUT, creationflags = 0x08000000 )
      output = prc.communicate()[0]
      if prc.returncode:
        if prc.returncode == 77:
          print(i, ': test skipped')
          skipped += 1
        else:
          print(i, ': ERROR (', prc.returncode, ' )')
          run_fail += 1
          if output:
            print('    ', output.decode(), end = '')
      else:
        print(i, ': success')
        run_ok += 1
    else:
      print("Build failure for {0}".format(i))
      build_fail += 1
  print(build_fail + run_ok + run_fail + skipped, "tests:")
  if build_fail > 0:
    print("\t{0} failed to build".format(build_fail))
  if run_ok > 0:
    print("\t{0} ran correctly".format(run_ok))
  if run_fail > 0:
    print("\t{0} failed".format(run_fail))
  if skipped > 0:
    print("\t{0} skipped".format(skipped))
if len(sys.argv) == 1:
  try:
    input(".. completed - press ENTER")
  except:
    pass
else:
  sys.exit(build_fail + run_fail)
