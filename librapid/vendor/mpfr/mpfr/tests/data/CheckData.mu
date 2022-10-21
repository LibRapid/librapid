file := "acos":  // data file to be tested
f := arccos:     // corresponding MuPAD function

// ------------------------ do not edit below this line -----------------------

// auxiliary function for file log10
log10 := proc(x)
begin
   if testtype(x, Type::Numeric) then return (ln(x)/ln(10))
   else return (procname (x))
   end_if
end_proc:
log10 := funcenv(log10):
log10::type := "log10":
log10::print := "log10":

// auxiliary function for file mulpi
mulpi := proc(x)
begin
   if testtype(x, Type::Numeric) then return (x * 884279719003555/2^48)
   else return (procname (x))
   end_if
end_proc:
mulpi := funcenv(mulpi):
mulpi::type := "mulpi":
mulpi::print := "mulpi":

// auxiliary function for file exp2
exp2 := proc(x)
begin
   if testtype(x, Type::Numeric) then return(2^x)
   else return (procname (x))
   end_if
end_proc:
exp2 := funcenv(exp2):
exp2::type := "exp2":
exp2::print := "exp2":

// auxiliary function for file pow275
pow275 := proc(x)
begin
   if testtype(x, Type::Numeric) then return(x^(11/4))
   else return (procname (x))
   end_if
end_proc:
pow275 := funcenv(pow275):
pow275::type := "pow275":
pow275::print := "pow275":

print (Unquoted, _concat ("Checking file ", file, " with MuPAD function ", f));

prec := 20: // default working precision

// tries to parse an unsigned integer x from l[i]
// return [FAIL, i] if an error occurred
// otherwise return [x, j] where j is the next position (after x)
scanf_uint := proc (l, i)
local x, c, n;
begin
   n := length (l);
   if i > n then return ([FAIL, i])
   else
      c := stringlib::contains ("0123456789", l[i], IndexList);
      if c = [] then return ([FAIL, i]) end_if;
      x := c[1] - 1;
      i := i + 1;
      while i <= n do
         c := stringlib::contains ("0123456789", l[i], IndexList);
         if c = [] then return ([x, i]) end_if;
         x := 10*x + (c[1] - 1);
         i := i + 1;
      end_while;
      return ([x, i]);
   end_if
end_proc:

scanf_hex := proc (l, i)
local x, c, n, s, se, e;
begin
   n := length (l);
   if i > n then return ([FAIL, i]) end_if;
   if l[i] = "-" then s := -1; i := i + 1 else s := 1 end_if;
   // we need to read at least 0xd, thus l[i..i+2]
   if i + 2 > n or l[i..i+1] <> "0x" then return ([FAIL, i]) end_if;
   i := i + 2;
   c := stringlib::contains ("0123456789ABCDEF", l[i], IndexList);
   if c = [] then return ([FAIL, i]) end_if;
   x := c[1] - 1;
   i := i + 1;
   while i <= n do
      c := stringlib::contains ("0123456789ABCDEF", l[i], IndexList);
      if c = [] then break end_if;
      x := 16*x + (c[1] - 1);
      i := i + 1;
   end_while;
   // now read p<exp> (if any)
   if i <= n and l[i] = "p" then
      i := i + 1;
      if i > n then return ([FAIL, i]) end_if;
      if l[i] = "-" then se := -1; i := i + 1 else se := 1 end_if;
      e := scanf_uint (l, i);
      if e[1] = FAIL then return (e) end_if;
      i := e[2];
      e := se * e[1];
   else e := 0
   end_if;
   return ([s*x*2^e, i])
end_proc:

scanf_rnd := proc (l, i)
local n, r;
begin
   n := length (l);
   if i > n then return ([FAIL, i]) end_if;
   r := l[i];
   if contains (["n", "z", "u", "d"], r) = 0 then return ([FAIL, i])
   else return ([r, i+1])
   end_if
end_proc:

// return new i
skip_blanks := proc (l, i)
local n;
begin
   n := length (l);
   while i <= n and l[i] = " " do
      i := i + 1;
   end_while;
   return (i)
end_proc:

// round z with mode r and precision p bits
Round := proc (z, r, p)
local e, s, m;
begin
   if iszero(z) then return(0) end_if;
   // first compute exponent
   if z < 0 then s := -1; m := -z else s := 1; m := z end_if;
   e := ceil (log(2, m));
   m := m * 2 ^ (p - e);
   if r = "n" then m := round (m)
   elif r = "z" then m := floor (m)
   elif r = "u" then m := ceil (s * m); s := 1
   elif r = "d" then m := floor (s * m); s := 1
   else error (_concat("unknown rounding mode: ", r))
   end_if;
   return (s * m * 2^(e-p))
end_proc:

foo := fopen (file, Read, Text):
l := ftextinput (foo):
checked := 0: // number of checked lines
maxprec := 0: // maximal precision used
while type(l) <> DOM_NULL do
   if l[1] <> "#" then
      i := 1; // current index in l
      xprec := scanf_uint (l, i);
      i := xprec[2];
      xprec := xprec[1];
      if xprec = FAIL then
         error ("cannot read input precision")
      end_if;
      i := skip_blanks (l, i);
      yprec := scanf_uint (l, i);
      i := yprec[2];
      yprec := yprec[1];
      if yprec = FAIL then
         error ("cannot read output precision")
      end_if;
      i := skip_blanks (l, i);
      rnd := scanf_rnd (l, i);
      i := rnd[2];
      rnd := rnd[1];
      if rnd = FAIL then
         error ("cannot read rounding mode")
      end_if;
      i := skip_blanks (l, i);
      x := scanf_hex (l, i);
      i := x[2];
      x := x[1];
      if x = FAIL then
         error ("cannot read input number")
      end_if;
      i := skip_blanks (l, i);
      y := scanf_hex (l, i);
      i := y[2];
      y := y[1];
      if y = FAIL then
         error ("cannot read output number")
      end_if;
      DIGITS := prec;
      exact := bool(type(f(x)) <> type(f(xxx)));
      repeat
         if exact then z := f(x)...f(x) else z := f(x...x) end_if;
         z1 := Round (op(z, 1), rnd, yprec);
         z2 := Round (op(z, 2), rnd, yprec);
         if z1 = z2 then break end_if;
         DIGITS := DIGITS + iquo (DIGITS, 2);
      until z1 = z2 end_repeat;
      if DIGITS > maxprec then maxprec := DIGITS end_if;
      if y <> z1 then
         print (Unquoted, _concat ("x=", x));
         print (Unquoted, _concat ("file contains ", y));
         print (Unquoted, _concat ("MuPAD claims ", z1));
         error (_concat("wrong data line: ", l))
      end_if;
      checked := checked + 1;
   end_if;
   l := ftextinput (foo); // read next line
end_while;
print (Unquoted, _concat ("checked lines: ", checked));
print (Unquoted, _concat ("maximal precision: ", maxprec));
print (Unquoted, _concat ("total time: ", time ()));
fclose (foo);
quit;
