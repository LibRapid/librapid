#!/bin/bash
# this script helps to check MPFR on the GCC compile farm
# 1) update the GMP version if needed
# 2) update the MPFR release candidate
# 3) ssh gcc10 < cfarm.sh
GMP=gmp-6.1.2
MPFR=mpfr-4.0.1
RC=rc2
/bin/rm -fr gmp*
if [ ! -d $GMP ]; then
   wget https://gmplib.org/download/gmp/$GMP.tar.bz2
   bunzip2 $GMP.tar.bz2
   tar xf $GMP.tar
   cd $GMP
   ./configure --prefix=$HOME
   make -j4
   make install
   cd $HOME
fi
/bin/rm -fr mpfr*
wget https://www.mpfr.org/$MPFR/$MPFR-$RC.tar.gz
gunzip $MPFR-$RC.tar.gz
tar xf $MPFR-$RC.tar
cd $MPFR-$RC
if [ "`hostname`" = "power-aix" ]; then # gcc111
   export OBJECT_MODE=64
   # or ./configure AR="ar -X64" NM="nm -B -X64"
fi
./configure --with-gmp=$HOME
make -j4
make check -j4

# results with mpfr-4.0.1-rc2.tar.gz (180 tests)
# gcc10 No route to host
# gcc11 Connection refused (asks for a password)
# gcc12 # PASS:  180
# gcc13 # PASS:  180
# gcc14 # PASS:  180
# gcc15 # PASS:  180
# gcc16 # PASS:  180
# gcc17 Connection timed out
# gcc20 # PASS:  180
# gcc21 # PASS:  180
# gcc22 # PASS:  178 # SKIP:  2
# gcc23 # PASS:  178 # SKIP:  2
# gcc24 # PASS:  178 # SKIP:  2
# gcc33 Connection timed out
# gcc34 Connection timed out
# gcc35 Connection timed out
# gcc36 Connection timed out
# gcc37 Connection timed out
# gcc38 Connection timed out
# gcc40 Connection timed out
# gcc41 Connection timed out
# gcc42 Connection timed out
# gcc43 Connection timed out
# gcc45 Connection timed out
# gcc46 Connection timed out
# gcc47 Connection timed out
# gcc49 Name or service not known
# gcc50 Connection timed out
# gcc51 Connection timed out
# gcc52 Connection timed out
# gcc53 Connection refused
# gcc54 Connection timed out
# gcc55 Connection timed out
# gcc56 Connection timed out
# gcc57 Connection timed out
# gcc60 Connection timed out
# gcc61 Connection timed out
# gcc62 Connection timed out
# gcc63 Connection timed out
# gcc64 Connection timed out
# gcc66 Connection timed out
# gcc67 # PASS:  180
# gcc68 Network is unreachable
# gcc70 # PASS:  178 # SKIP:  2
# gcc75 # PASS:  180
# gcc76 # PASS:  180
# gcc100 Name or service not known
# gcc101 Name or service not known
# gcc110 # PASS:  179 # SKIP:  1
# gcc111 # PASS:  178 # SKIP:  2
# gcc112 # PASS:  179 # SKIP:  1
# gcc113 # PASS:  178 # SKIP:  2
# gcc114 # PASS:  178 # SKIP:  2
# gcc115 # PASS:  178 # SKIP:  2
# gcc116 # PASS:  178 # SKIP:  2
# gcc117 # PASS:  178 # SKIP:  2
# gcc118 # PASS:  178 # SKIP:  2
# gcc119 ???
# gcc200 Connection timed out
# gcc201 Connection timed out
# gcc202 # PASS:  159 # SKIP:  1 (gmp-6.1.2 configured with --disable-assembly)
