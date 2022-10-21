/* Example file to test mpfr addition against NTL.
   Usage:
   0) compile this file with NTL
   1) compile tadd.c with -DCHECK_EXTERNAL
   2) ./tadd | egrep -v 'Seed|Inf|NaN' > /tmp/log
      (Warning, this produces a large file.)
   3) ./RRTest < /tmp/add.log
*/

#include <NTL/mat_RR.h>

NTL_CLIENT

void
ReadBinary (ZZ &x)
{
  int s = 1;
  ZZ b, y;

  cin >> b;

  if (b < 0)
    {
      s = -1;
      b = -b;
    }

  x = 0;
  y = 1;
  while (b != 0)
    {
      x += (b % 10) * y;
      y *= 2;
      b /= 10;
    }
  if (s < 0)
    x = -x;
}

long
ReadRR (RR &a)
{
  long p;
  ZZ x;
  long e;

  cin >> p;
  ReadBinary (x);
  cin >> e;
  MakeRRPrec (a, x, e, p);
  return p;
}

void
Output (RR a, long p)
{
  cout << a.mantissa() << "*2^(" << a.exponent() << ") [" << p << "]" << endl;
}

// ulp difference between a and b
long
ulp (RR a, RR b, long p)
{
  ZZ ma, mb;
  long ea, eb;

  ma = a.x;
  ea = a.e;
  while (NumBits (ma) < p)
    {
      ma *= 2;
      ea --;
    }
  mb = b.x;
  eb = b.e;
  while (NumBits (mb) < p)
    {
      mb *= 2;
      eb --;
    }
  if (ea != eb) abort ();
  return to_long (ma - mb);
}

// #define TWO_ARGS /* for functions of two arguments like add, sub, pow */

int
main (void)
{
  RR a, b, c, d;
  long line = 0, errors = 0;
  long pa, pb, pc;

  while (!feof(stdin))
    {
      if (++line % 10 == 0)
        cout << "line " << line << endl;
      pb = ReadRR (b);
#ifdef TWO_ARGS
      pc = ReadRR (c);
#endif
      pa = ReadRR (a);
      RR::SetPrecision(pa);
      cos (d, b
#ifdef TWO_ARGS
           , c
#endif
           );
      if (d != a)
        {
          cerr << "error at line " << line << endl;
          cerr << "b="; Output(b, pb);
#ifdef TWO_ARGS
          cerr << " c="; Output(c, pc);
#endif
          cerr << "expected "; Output(a, pa);
          cerr << "got      "; Output(d, pa);
          cerr << "difference is " << ulp (a, d, pa) << " ulps" << endl;
          cerr << ++errors << " errors" << endl;
        }
    }
}
