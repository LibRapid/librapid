/* Return user CPU time measured in milliseconds. Thanks to Torbjorn. */

#if defined (ANSIONLY) || defined (USG) || defined (__SVR4) || defined (_UNICOS) || defined(__hpux)
#include <time.h>

static int
cputime ()
{
  return (int) ((double) clock () * 1000 / CLOCKS_PER_SEC);
}
#else
#include <sys/types.h>
#include <sys/resource.h>

static int
cputime ()
{
  struct rusage rus;

  getrusage (0, &rus);
  return rus.ru_utime.tv_sec * 1000 + rus.ru_utime.tv_usec / 1000;
}
#endif
