# convert from Gonnet's FPAccuracy data sets to mpfr format
# http://www.inf.ethz.ch/personal/gonnet/FPAccuracy/all.tar.Z

# 1 - cut the lines from (say) C/acos.c, remove the 3rd (eps) field,
#     replace the commata ',' by spaces, and remove the final '};'
#     (hint: cut -d" " -f1,2,4,5 /tmp/acos.c > /tmp/acos2.c)
# 2 - edit the infile and outfile lines below, and run
#     maple -q < gonnet.mpl 

infile := "/tmp/acos2.c":
outfile := "acos":

###################### don't edit below this line #############################

foo := proc(arg_m, val_m, arg_e, val_e, fp)
   fprintf (fp, "53 53 n ", 53);
   to_hex(arg_m, arg_e, fp);
   fprintf (fp, " ");
   # warning: Gonnet stores -val_e
   to_hex(val_m, -val_e, fp);
   fprintf (fp, "\n");
end:

to_hex := proc(m, e, fp)
   if m<0 then fprintf (fp, "-") fi;
   fprintf (fp, "0x%sp%d", convert(abs(m),hex), e);
end:

copyright := proc(fp)
   fprintf (fp, "# This file was generated from the FPAccuracy package\n# http://www.inf.ethz.ch/personal/gonnet/FPAccuracy/all.tar.Z:\n# Copyright (C) Gaston H. Gonnet\n# This program is free software; you can redistribute it and/or\n# modify it under the terms of the GNU General Public License\n# as published by the Free Software Foundation; either version 2\n# of the License, or (at your option) any later version.\n# This program is distributed in the hope that it will be useful,\n# but WITHOUT ANY WARRANTY; without even the implied warranty of\n# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n# GNU General Public License for more details.\n# You should have received a copy of the GNU General Public License\n# along with this program; if not, write to the Free Software\n# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.\n")
end:

fp := fopen (outfile, WRITE):

l := readdata(infile, integer, 4):
copyright(fp):
for e in l do foo(op(e), fp) od:

fclose (fp);

quit;

