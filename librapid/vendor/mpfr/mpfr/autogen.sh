#!/bin/sh

# "autoreconf -f" will clobber some of our files with generic ones if we
# don't move them out of the way (see $files below).

# EXIT and signals that correspond to SIGHUP, SIGINT, SIGQUIT and SIGTERM.
signals="0 1 2 3 15"

files="INSTALL doc/texinfo.tex"

cleanup()
{
  trap '' $signals
  for i in $files
  do
    if [ -f "$i.$$.tmp" ]; then
      echo "$0: restoring the $i file" >&2
      mv -f "$i.$$.tmp" "$i"
    fi
  done
}

for i in $files; do rm -f -- "$i.$$.tmp"; done
trap cleanup $signals
for i in $files; do mv -f -- "$i" "$i.$$.tmp"; done

autoreconf -v -f -i --warnings=all,error
status=$?

rm -rf autom4te.cache

exit $status
