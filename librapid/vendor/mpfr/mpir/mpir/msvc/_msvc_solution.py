# add a project file to the solution

from collections import defaultdict, OrderedDict
from os.path import join, exists, relpath, split
from uuid import uuid4
from re import compile

folder_guid = "{2150E333-8FDC-42A3-9474-1A3956D46DE8}"
vcxproj_guid = "{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}"
pyproj_guid =  "{888888A0-9F3D-457C-B088-3A5042F75D52}"

sol_11 = r'''
Microsoft Visual Studio Solution File, Format Version {0}.00
# Visual Studio {1}
'''

sol_12 = r'''VisualStudioVersion = {0}
MinimumVisualStudioVersion = 10.0.40219.1
'''

sol_2 = '''Project("{}") = "{}", "{}", "{}"
EndProject
'''

sol_3 = r'''Global
'''

sol_4 = r'''	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|Win32 = Debug|Win32
		Debug|x64 = Debug|x64
		Release|Win32 = Release|Win32
		Release|x64 = Release|x64
	EndGlobalSection
'''

sol_5 = r'''	GlobalSection(ProjectConfigurationPlatforms) = postSolution
'''

sol_6 = r'''	GlobalSection(NestedProjects) = preSolution
'''

sol_7 = '''		{} = {}
'''

sol_8 = r'''	EndGlobalSection
'''

sol_9 = r'''	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
'''

sol_10 = r'''EndGlobal
'''

s_guid = r'\s*(\{\w{8}-\w{4}-\w{4}-\w{4}-\w{12}\})\s*'
s_name = r'\s*\"([a-zA-Z][-.\\_a-zA-Z0-9]*\s*)\"\s*'
re_guid = compile(r'\s*\"\s*' + s_guid + r'\"\s*')
re_proj = compile(r'Project\s*\(\s*\"' + s_guid + r'\"\)\s*=\s*'
                  + s_name + r'\s*,\s*' + s_name + r'\s*,\s*\"' + s_guid + r'\"')
re_fmap = compile(r'\s*' + s_guid + r'\s*=\s*' + s_guid)
acfg = r'(Debug|Release)\|(WIn32|x64)'
re_acfg = compile(s_guid + r'\.' + acfg + r'\.ActiveCfg\s*=\s*' + acfg)

class msvc_solution(object):

  def __init__(self, path):
    self.soln_path, self.soln_file = split(path)

    self.g2fldr = {}
    self.g2proj = OrderedDict()
    self.gf2gpl = defaultdict(list)
    self.g2acfg = OrderedDict()

    if exists(path):
      lines = open(path).readlines()
      for i, ln in enumerate(lines):
        m = re_proj.search(ln)
        if m:
          if m.group(1) == folder_guid and m.group(2) == m.group(3):
            # folder guid -> folder name
            self.g2fldr[m.group(4)] = m.group(2)
          elif (m.group(1) == vcxproj_guid and m.group(3).endswith('.vcxproj') or
                m.group(1) == pyproj_guid and m.group(3).endswith('.pyproj')):
            # project guid -> proj_type_guid, proj_name, vcx_path
            self.g2proj[m.group(4)] = (m.group(1), m.group(2), m.group(3))

      for i, ln in enumerate(lines):
        m = re_fmap.search(ln)
        if m:
          if m.group(1) in self.g2proj and m.group(2) in self.g2fldr:
            self.gf2gpl[m.group(2)].append(m.group(1))
        m = re_acfg.search(ln)
        if m:
          g = m.group(1)
          if not g in self.g2acfg.keys():
            self.g2acfg[g] = [0, 1, 2, 3]
          ix = 2 * (m.group(2) == 'Release') + (m.group(3) == 'x64')
          ac = 2 * (m.group(4) == 'Release') + (m.group(5) == 'x64')
          self.g2acfg[g][ix] = ac

    for g in self.g2proj:
      for _, gpl in self.gf2gpl.items():
        if g in gpl:
          break
      else:
        self.gf2gpl[''].append(g)

    assert len(self.g2fldr.keys()) == len(self.gf2gpl.keys()) - (1 if '' in self.gf2gpl.keys() else 0)
    assert sum(len(gpl) for gf, gpl in self.gf2gpl.items()) == len(self.g2proj.keys())

  def write_solution(self, vs_info):
    if len(self.g2fldr.keys()) > len(self.gf2gpl.keys()):
      for g in list(self.g2fldr.keys()):
        if g not in self.gf2gpl.keys():
          del self.g2fldr[g]

    assert len(self.g2fldr.keys()) == len(self.gf2gpl.keys()) - (1 if '' in self.gf2gpl.keys() else 0)
    assert sum(len(gpl) for gf, gpl in self.gf2gpl.items()) == len(self.g2proj.keys())

    vs_str = sol_11.format(vs_info['solution'], vs_info['visual studio'])
    if int(vs_info['msvc']) > 12:
      vs_str += sol_12.format(vs_info['msvc_long'])

    with open(join(self.soln_path, self.soln_file), 'w') as outf:
      outf.write(vs_str)

      for g, f in self.g2fldr.items():
        outf.write(sol_2.format(folder_guid, f, f, g))

      for g, (gg, f, n) in self.g2proj.items():
        outf.write(sol_2.format(gg, f, n, g))

      outf.write(sol_3)
      outf.write(sol_4)

      outf.write(sol_5)
      wx, dr = ['Win32', 'x64'], ['Debug', 'Release']
      for g in self.g2proj.keys():
        if g in self.g2acfg.keys():
          ac4 = self.g2acfg.get(g, [0, 1, 2, 3])
          for cf in (0, 1):
            for pl in (0, 1):
              ac = ac4[2 * cf + pl]
              awx = 'AnyCPU' if gg == pyproj_guid else wx[ac & 1]
              adr = dr[(ac > 1) & 1]
              outf.write('		{0:s}.{1:s}|{2:s}.ActiveCfg = {3:s}|{4:s}\n'.format(g, dr[cf], wx[pl], adr, awx))
      outf.write(sol_8)

      del self.gf2gpl['']
      if len(self.gf2gpl.keys()):
        outf.write(sol_6)
        for gf, gpl in self.gf2gpl.items():
          for gp in gpl:
            outf.write(sol_7.format(gp, gf))
        outf.write(sol_8)

      outf.write(sol_9)
      outf.write(sol_10)

  def get_project_guid(self, proj_name, file_path):
    relp = relpath(file_path, self.soln_path)
    p_guid = None
    for g in list(self.g2proj.keys()):
      if self.g2proj[g] == (vcxproj_guid, proj_name, relp):
        p_guid = g
        break
    if not p_guid:
        p_guid = '{' + str(uuid4()).upper() + '}'
    return p_guid

  def set_acfg(self, g, mode):
    if g not in self.g2acfg.keys():
      proj_name = self.g2proj[g][1]
      if len(mode) == 2 or proj_name.endswith('gc') or proj_name.endswith('cxx'):
        cfg = [0, 1, 2, 3]
      else:
        cfg = [0, 0, 2, 2] if mode[0] == 'Win32' else [1, 1, 3, 3]
      self.g2acfg[g] = cfg

  def add_project(self, soln_folder, proj_name, file_path, p_guid, mode):
    relp = relpath(file_path, self.soln_path)
    if soln_folder:
      for g, f in self.g2fldr.items():
        if f == soln_folder:
          f_guid = g
          break
      else:
        f_guid = '{' + str(uuid4()).upper() + '}'
        self.g2fldr[f_guid] = soln_folder

    for g in list(self.g2proj.keys()):
      if self.g2proj[g] == (vcxproj_guid, proj_name, relp):
        break
    else:
      self.g2proj[p_guid.upper()] = (vcxproj_guid, proj_name, relp)
      self.gf2gpl[f_guid if soln_folder else ''].append(p_guid.upper())
      self.set_acfg(p_guid.upper(), mode)
