# Copyright 2005-2022 Free Software Foundation, Inc.
# Contributed by Patrick Pelissier, INRIA.
#
# This file is part of the MPFR Library.
#
# The MPFR Library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# The MPFR Library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with the MPFR Library; see the file COPYING.LESSER.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,
# MA 02110-1301, USA.

AR=ar
CC=gcc
CXX=g++
RANLIB=ranlib
# added -march=native to CFLAGS to properly detect cpus
# (the -march=native option is not supported on all architectures,
# but if this is an issue, this could be solved by a test of the
# architecture with an ifeq).
CFLAGS=-O2 -fomit-frame-pointer -Wall -g -static -march=native
LDFLAGS=
RM=rm -f
CP=cp -f
MV=mv -f

GMP=/usr
MPFR=$(PWD)/../../src
PARI=$(GMP)
NTL=$(GMP)
CLN=$(GMP)
MAPM=$(GMP)
ARPREC=$(GMP)
CRLIBM=$(GMP)
LIDIA=/localdisk/lidia/

GMP_INCLUDE=$(GMP)/include

MPFR_INCLUDE=$(MPFR)/include
MPFR_LIB=$(MPFR)/lib

########################################################
VERSION=0.6.2
NAME=mbench

#########################################################
.SUFFIXES: .c .o

# add -I$(MPFR) and -I$(GMP) if GMP/MPFR build directories are given
# Warning! If $(MPFR) and/or $(GMP) is regarded as a "system directory",
# this may give unexpected results[*]; in particular, with GCC, the option
# is ignored in such a case.
# [*] https://austingroupbugs.net/view.php?id=187
ifeq	($(CC),gcc)
INCLUDES=`test -f $(GMP_INCLUDE)/gmp.h && echo "-include $(GMP_INCLUDE)/gmp.h"` `test -f $(GMP)/gmp.h && echo "-include $(GMP)/gmp.h"` `test -f $(MPFR_INCLUDE)/mpfr.h && echo "-include $(MPFR_INCLUDE)/mpfr.h"` `test -f $(MPFR)/mpfr.h && echo "-include $(MPFR)/mpfr.h"` `test -f $(MPFR)/src/mpfr.h && echo "-include $(MPFR)/src/mpfr.h"` -I. -I$(MPFR_INCLUDE) -I$(MPFR) -I$(GMP)/include/ -I$(GMP)
else
INCLUDES=-I. -I$(MPFR_INCLUDE) -I$(MPFR) -I$(GMP)/include/ -I$(GMP)
endif

# search first for real install, then for build directory
LIBS=`(test -f $(MPFR_LIB)/libmpfr.a && echo $(MPFR_LIB)/libmpfr.a)` `(test -f $(MPFR)/.libs/libmpfr.a && echo $(MPFR)/.libs/libmpfr.a)` `(test -f $(MPFR)/src/.libs/libmpfr.a && echo $(MPFR)/src/.libs/libmpfr.a)` `(test -f $(GMP)/lib/libgmp.a && echo $(GMP)/lib/libgmp.a)` `(test -f $(GMP)/lib/$$(gcc -print-multiarch 2>/dev/null)/libgmp.a && echo $(GMP)/lib/$$(gcc -print-multiarch 2>/dev/null)/libgmp.a)` `(test -f $(GMP)/.libs/libgmp.a && echo $(GMP)/.libs/libgmp.a)`

XLIBS=`test -f $(PARI)/lib/libpari.a && echo $(PARI)/lib/libpari.a` \
      `test -f $(NTL)/lib/libntl.a && echo $(NTL)/lib/libntl.a`     \
      `test -f $(CLN)/lib/libcln.a && echo $(CLN)/lib/libcln.a`     \
      `test -f $(ARPREC)/lib/libmp.a && echo $(ARPREC)/lib/libmp.a` \
      `test -f $(CRLIBM)/lib/libcrlibm.a && echo $(CRLIBM)/lib/libcrlibm.a` \
      -lm

TARGETS=mpfr-v4 mpfr-v6 mpfr-gfx mfv5
DIST=Makefile mpfr-v4.c mpfr-v6.c mpfr-gfx.c generate.c timp.h mfv5-arprec.cc mfv5-cln.cc mfv5-mpfr.cc mfv5-pari.cc mfv5.cc mfv5-mpf.cc mfv5-ntl.cc mfv5-lidia.cc mfv5.h mfv5-void.cc mfv5-libc.cc mfv5-crlibm.cc

##############################################################

.c.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $<

.cc.o:
	$(CXX) $(INCLUDES) $(CFLAGS) -c $<

all: mfv5

clean:
	$(RM) *.o *~ $(TARGETS) float.data

mpfr-v4: mpfr-v4.o
	$(CC) $(CFLAGS) mpfr-v4.o $(LIBS) -o mpfr-v4

mpfr-v6: mpfr-v6.o float.data
	$(CC) $(CFLAGS) mpfr-v6.o $(LIBS) -lm -o mpfr-v6

mpfr-gfx: mpfr-gfx.o
	$(CC) $(CFLAGS) mpfr-gfx.o $(LIBS) -o mpfr-gfx

float.data: generate.o
	$(CC) $(CFLAGS) generate.o $(LIBS) -o generate-float-data
	./generate-float-data
	$(RM) generate-float-data

mfv5.o: timp.h

mfv5-libc.o: timp.h

mfv5-mpf.o: timp.h

mfv5-mpfr.o: timp.h

mfv5-ntl.o: mfv5-ntl.cc mfv5.h timp.h
	@(test -f $(NTL)/lib/libntl.a && $(CXX) $(INCLUDES) -I$(NTL)/include/ $(CFLAGS) -c $< && echo "Using NTL.") || ($(CXX) -c mfv5-void.cc -o mfv5-ntl.o && echo "NTL not found.")

mfv5-arprec.o: mfv5-arprec.cc mfv5.h timp.h
	@(test -f $(ARPREC)/lib/libmp.a && $(CXX) $(INCLUDES) -I$(ARPREC)/include/ $(CFLAGS) -c $< && echo "Using ARPREC.") || ($(CXX) -c mfv5-void.cc -o mfv5-arprec.o && echo "Arprec not found.")

mfv5-pari.o: mfv5-pari.cc mfv5.h timp.h
	@(test -f $(PARI)/lib/libpari.a && $(CXX) $(INCLUDES) -I$(PARI)/include/ $(CFLAGS) -c $< && echo "Using PARI.") || ($(CXX) -c mfv5-void.cc -o mfv5-pari.o && echo "PARI not found.")

mfv5-cln.o: mfv5-cln.cc mfv5.h timp.h
	@(test -f $(CLN)/lib/libcln.a && $(CXX) $(INCLUDES) -I$(CLN)/include/ $(CFLAGS) -c $< && echo "Using CLN.") || ($(CXX) -c mfv5-void.cc -o mfv5-cln.o && echo "CLN not found.")

mfv5-lidia.o: mfv5-lidia.cc mfv5.h timp.h
	@(test -f $(LIDIA)/lib/liblidia.a && $(CXX) $(INCLUDES) -I$(LIDIA)/include/ $(CFLAGS) -c $< && echo "Using LIDIA.") || ($(CXX) -c mfv5-void.cc -o mfv5-lidia.o && echo "LIDIA not found.")

mfv5-crlibm.o: mfv5-crlibm.cc mfv5.h timp.h
	@(test -f $(CRLIBM)/lib/libcrlibm.a && $(CXX) $(INCLUDES) -I$(CRLIBM)/include/ $(CFLAGS) -c $< && echo "Using CRLIBM.") || ($(CXX) -c mfv5-void.cc -o mfv5-crlibm.o && echo "CRLIBM not found.")

mfv5:	mfv5.o mfv5-mpfr.o mfv5-mpf.o mfv5-ntl.o mfv5-arprec.o mfv5-pari.o mfv5-cln.o mfv5-libc.o mfv5-crlibm.o
	$(CXX) $(CFLAGS) mfv5.o mfv5-mpfr.o mfv5-mpf.o mfv5-ntl.o mfv5-arprec.o mfv5-pari.o mfv5-cln.o mfv5-libc.o mfv5-crlibm.o $(XLIBS) $(LIBS) $(LDFLAGS) -o mfv5

dist: $(DIST)
	rm -fr $(NAME)-$(VERSION) $(NAME)-$(VERSION).tar $(NAME)-$(VERSION).tar.bz2
	mkdir $(NAME)-$(VERSION)
	cp $(DIST) $(NAME)-$(VERSION)
	tar cf $(NAME)-$(VERSION).tar $(NAME)-$(VERSION)
	bzip2 $(NAME)-$(VERSION).tar
	rm -fr $(NAME)-$(VERSION)

rt:
	make clean all CFLAGS="-O2 -fomit-frame-pointer -Wall -g -DUSE_CLOCK_MONOTONIC" LDFLAGS="-lrt"
