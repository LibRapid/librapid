#
# These procedures come from the source code of GFun.
#
getname:=proc(yofz::function(name), y, z)
  y:=op(0,yofz);
  if type(y,'procedure') then error `not an unassigned name`,y fi;
  z:=op(yofz)
end proc:


#
# returns the smallest i such that u(n+i) appears in a recurrence
#
minindex := proc(rec,u,n)
  min(op(map(op,indets(rec,'specfunc'('linear'(n),u)))))-n
end proc:


#
# returns the largest i such that u(n+i) appears in a recurrence
#
maxindex := proc(rec,u,n)
  max(op(map(op,indets(rec,'specfunc'('linear'(n),u)))))-n
end proc:


#
# A recurrence of the form a(n+d) = p(n)/q(n) a(n) is represented through a record:
# OneTermRecurrence : record(order, numerator, denominator)
#
`type/OneTermRecurrence` := 'record(order, numerator, denominator)':


#
#checkOneTermRecurrence
# Input: a recurrence rec (either with or without initial conditions).
#          If it has initial conditions, they are ignored.
#        a(n): the name of the sequence and the name of the variable.
#
# Output:
# This procedure checks that rec is a recurrence of the form a(n+d) = p(n)/q(n) a(n)
# If the check succeeds, it returns the corresponding record. If it fails, an error is
# returned.
#
checkOneTermRecurrence := proc(rec, aofn)::OneTermRecurrence;
  local r, d, a, n, term1, term2, res;

  getname(aofn, a, n):
  if type(rec, 'set') then
    r:=select(has, rec, n);
    if nops(r)>1
      then error `invalid recurrence`, rec
    fi:
    if nops(r)=0
      then error "%1 does not appear in the recurrence", n
    fi:
    r := op(r):
  else r:=rec:
  fi:
  if type(r,'`=`')
    then r:=op(1,r)-op(2,r)
  fi:
  if indets(r,'specfunc'('anything',a)) <> indets(r,'specfunc'('linear'(n),a))
    then error "the recurrence contains elements that are not linear in %1", n
  fi:
  if nops(r) <> 2
    then error "the recurrence contains %1 terms (expected 2)", nops(r)
  fi:
  r := subs(n=n-minindex(r, a, n), r):
  d := maxindex(r, a, n):

  term1 := select(has, r, a(n)):
  term2 := select(has, r, a(n+d)):

  res := factor( -(term1/a(n)) / (term2/a(n+d)) ):

  Record( 'order'=d, 'numerator' = numer(res), 'denominator' = denom(res) )
end proc:


#
# my_factors factorizes p the same way as factors(p) would do except that the constant part is computed
# differently. We assume here that p has integer coefficients, and we want to factorize it over polynomials
# with integer coefficients. my_factors ensures that the factors have integer coefficients.
#
my_factors := proc(p)
  local L, c, fact, i, my_c, my_fact, q:
  L := factors(p):
  c := L[1]: fact := L[2]:
  my_c := c: my_fact := []:
  for i from 1 to nops(fact) do
    q := denom(fact[i][1]):
    my_fact := [ op(my_fact), [ fact[i][1]*q, fact[i][2] ] ]:
    my_c := my_c / (q^fact[i][2]):
  od:
  [ my_c, my_fact]:
end proc:


#
# This procedure decomposes a one-term recurrence with the following form:
# a(n+d) = c * s1(n)/s1(n+d) * s2(n+d)/s2(n) * p(n)/q(n) * a(n)
#
# Known issue: this procedure assumes that the only variables involved are n and x with their usual meaning.
#
decomposeOneTermRecurrence := proc(formalRec::OneTermRecurrence, res_cste, res_s1, res_s2, res_p, res_q)
  local p, q, cste, s1, s2, d, L, i, tmp, exponent, r, polyring;
  p := formalRec:-numerator:
  q := formalRec:-denominator:
  d := formalRec:-order:
  s1 := 1:
  L := op(2,my_factors(p)): # L contains the non trivial factors of p
  for i from 1 to nops(L) do
    tmp := L[i][1]: exponent := L[i][2]:
    r := gcd(tmp^exponent, subs(n=n-d, q)):
    p := quo(p,r,n): q := quo(q, subs(n=n+d, r),n): s1 := s1 * r:
  od:

  s2 := 1:
  L := op(2,my_factors(p)): # L contains the *remaining* non trivial factors of p
  for i from 1 to nops(L) do
    tmp := L[i][1]: exponent := L[i][2]:
    r := gcd(tmp^exponent, subs(n=n+d, q)):
    p := quo(p, r, n): q := quo(q, subs(n=n-d, r), n): s2 := s2 * r:
  od:

  # Finally we look for the constant part (with respect to n) of p/q
  cste := op(1, my_factors(p))/op(1, my_factors(q)):
  p := p/op(1, my_factors(p)): q := q/op(1, my_factors(q)):
  polyring := RegularChains[PolynomialRing]([n,x]):
  L := op(2, my_factors(p)):
  for i from 1 to nops(L) do
    if RegularChains[MainVariable](L[i][1], polyring) = x
      then cste := cste * L[i][1]^L[i][2]: p := quo(p,L[i][1]^L[i][2],x):
    fi:
  od:
  L := op(2, my_factors(q)):
  for i from 1 to nops(L) do
    if RegularChains[MainVariable](L[i][1], polyring) = x
      then cste := cste / L[i][1]^L[i][2]: q := quo(q,L[i][1]^L[i][2],x):
    fi:
  od:

  res_cste := cste;
  res_s1 := s1;
  res_s2 := s2;
  res_p := simplify(p);
  res_q := simplify(q);
end proc:


#
#coeffrecToTermsrec
# Input: a linear recurrence rec (either with or without initial conditions).
#           a(n): the name of the sequence and the name of the variable.
#           x: a value or symbolic name
#
# Output:
# The recurrence satisfied by a(n)*x^n. Note that this recurrence is also denoted by a(n).
# If initial conditions were provided, corresponding initial conditions are computed.
#
coeffrecToTermsrec := proc(rec, aofn, x)
  local a,n,L,r,cond,d,i,tmp,c,res;
  getname(aofn, a, n):
  if type(rec, 'set') then
    L := selectremove(has, rec, n):
    r := L[1]:
    if nops(r)>1
      then error `invalid recurrence`, rec
    fi:
    if nops(r)=0
      then error "%1 does not appear in the recurrence", n
    fi:
    r := op(r):
    cond := L[2]:
  else r := rec:
  fi:
  d := maxindex(r, a, n):
  L := indets(r,'specfunc'('linear'(n),a)):
  if indets(r,'specfunc'('anything',a)) <> L
    then error "the recurrence contains elements that are not linear in %1", n
  fi:
  L := map(op, L):
  for i from 1 to nops(L) do
    r := subs(a(op(i,L))=a(op(i,L))*x^(d-op(i,L)+n), r):
  od:
  if cond<>'cond' then
    c := {}:
    for i from 1 to nops(cond) do
      tmp := op(i, cond): # tmp should have the form 'a(k) = cste'
      if not type(tmp,'`=`') then error "Invalid initial condition: %1", tmp: fi:
      L := selectremove(has, {op(tmp)}, a):
      if (nops(L[1]) <> 1) or (nops(L[2])<>1)
        then error "Invalid initial condition: %1", tmp:
      fi:
      tmp := op(1, L[1]): # tmp has the form 'a(k)'
      c := {op(c), tmp = op(1, L[2])*x^op(tmp)}:
    od:
    res := {r, op(c)}:
  else res := r:
  fi:
  res:
end proc:


#
# This procedure removes the conditions of the form a(k)=0 from the initial conditions of rec
# It returns a list L = [L1, L2, ...] where Li = [k, expr] representing the condition a(k)=expr.
# Moreover, it asserts that the Li are ordered by increasing k.
#
removeTrivialConditions := proc(rec, aofn)
  local a,n,i,L,tmp,c,cond,k:
  getname(aofn, a, n):
  if not type(rec, 'set') then
    error "%1 is not a recurrence with initial conditions", rec
  else
    L := selectremove(has, rec, n):
    cond := L[2]:
    if nops(cond)=0
      then error "%1 does not contain initial conditions", rec
    fi:
  fi:  
  c := []:
  for i from 1 to nops(cond) do
    tmp := op(i, cond): # tmp should have the form 'a(k) = cste'
    if not type(tmp,'`=`') then error "Invalid initial condition: %1", tmp: fi:
    L := selectremove(has, {op(tmp)}, a):
    if (nops(L[1]) <> 1) or (nops(L[2])<>1)
      then error "Invalid initial condition: %1", tmp:
    fi:
    if op(1, L[2])<>0 then c := [op(c), [op(op(1, L[1])), op(1, L[2])]]: fi:
  od:
  # We check that the conditions are ordered by increasing k.
  if (nops(c)=0) then return c: fi:
  k := c[1][1]:
  for i from 2 to nops(c) do
    if (c[i][1]<=k)
    then error "Unexpected error in removeTrivialConditions: the conditions are not correctly ordered (%1)\n", c
    else k := c[i][1]
    fi:
  od:
  c:
end proc:


#
# findFixpointOfDifferences: takes a set L of integer and returns the smallest set S
# containing L and such that for each i, S[i]-S[i-1] \in S
findFixpointOfDifferences := proc(L)
  local res, i:
  res := L:
  for i from 2 to nops(L) do
    res := { op(res), L[i]-L[i-1] }:
  od:
  if (res=L) then return res else return findFixpointOfDifferences(res) fi:
end proc:


#
# error_counter functions allows one to follow the accumulation of errors in each variable.
#   an error_counter is a list of the form [[var1, c1], [var2, c2], ... ]
#   where the vari are variable names and the ci indicate how many approximation errors
#   are accumulated in vari.
#

#
# This procedure initializes the counter associated with variable var to 1 (and creates it if needed.)
# It returns an up-to-date error_counter.
init_error_counter := proc (var, error_counter)
  local i, res:
  res  := error_counter:
  for i from 1 to nops(res) do
    if (res[i][1]=var) 
    then res[i][2] := 1:
         return res:
    fi
  od:
  res := [op(res), [var, 1]]:
end:


#
# This procedure adds a given number to the counter associated with variable var.
# It returns an up-to-date error_counter.
add_to_error_counter := proc (var, n, error_counter)
  local i, res:
  res  := error_counter:
  for i from 1 to nops(res) do
    if (res[i][1]=var) 
    then res[i][2] := res[i][2]+n:
         return res:
    fi
  od:
  res := [op(res), [var, n]]:
end proc:

#
# This procedure sets the value of the counter associated with variable var.
# It returns an up-to-date error_counter.
set_error_counter := proc(var, n, error_counter)
  local i,err:
  err  := error_counter:
  for i from 1 to nops(err) do
    if (err[i][1]=var) 
    then err[i][2] := n:
         return err:
    fi
  od:
  err := [op(err), [var, n]]:  
end proc:

#
# This procedure initializes the counter associated to the multiplication of var2 and var3,
# putting the result in variable var1. 
# It returns an up-to-date error_counter.
error_counter_of_a_multiplication := proc (var1, var2, var3, error_counter)
  local i, res, c2, c3:
  c2 := 0: c3 := 0:
  for i from 1 to nops(error_counter) do
    if (error_counter[i][1]=var2) then c2 := error_counter[i][2] fi:
    if (error_counter[i][1]=var3) then c3 := error_counter[i][2] fi:
    if (error_counter[i][1]=var1)
    then
      res := [ op(error_counter[1..i-1]), op(error_counter[i+1..nops(error_counter)]) ]
    fi:
  od:
  if (res = 'res') then res := error_counter fi:
  res := [op(res), [var1, c2+c3+1]]:
end:

#
# Copies the error counter of var2 into var1
error_counter_on_copy := proc(var1, var2, error_counter)
  local i, err, c2:
  c2 := 0:
  for i from 1 to nops(error_counter) do
    if (error_counter[i][1] = var2) then c2 := error_counter[i][2] fi:
    if (error_counter[i][1] = var1)
    then
      err := [ op(error_counter[1..i-1]), op(error_counter[i+1..nops(error_counter)]) ]
    fi:
  od:
  if (err = 'err') then err := error_counter fi:
  if (c2 <> 0) then err := [op(res), [var1, c2]] fi:
end proc:


#
# Returns the value of the error counter associated to a variable
find_in_error_counter := proc(var, error_counter) 
  local i:
  for i from 1 to nops(error_counter) do
    if (error_counter[i][1] = var) then return error_counter[i][2] fi:
  od:
  return 0:
end proc:

#
# generate_multiply_rational(fd, var1, var2, r, error_counter, indent) generates code for performing
# var1 = var2*r in MPFR
#   fd is the file descriptor in which the code shall be produced.
#   var1 and var2 are strings representing variable names. r is a Maple rational number.
#   error_counter is an error_counter (as described above).
#   indent is an optional argument. It is a string used to correctly indent the code. It is prefixed to any
#   generated line. Hence, if indent="  ", the generated code will be indented by 2 spaces. 
# An up-to-date error_counter is returned.
generate_multiply_rational := proc(fd, var1, var2, r, error_counter, indent:="")
  local p,q,err:
  err := error_counter:
  if (whattype(r)<>'fraction') and (whattype(r)<>'integer')
  then error "generate_multiply_rational used with non rational number %1", r: fi:
  if (abs(r)=1)
  then
    if (var1=var2)
    then
      if (r<>1) then fprintf(fd, "%sMPFR_CHANGE_SIGN (%s);\n", indent, var1) fi:
      return err:
    else 
      if (r=1)
        then fprintf(fd, "%smpfr_set (%s, %s, MPFR_RNDN);\n", indent, var1, var2):
        else fprintf(fd, "%smpfr_neg (%s, %s, MPFR_RNDN);\n", indent, var1, var2):
      fi:
      return error_counter_on_copy(var1, var2, err):
    fi
  fi:
  # Now, r is a rational number different from 1.
  p := numer(r): q := denom(r):
  if (abs(p)<>1)
  then
    fprintf(fd, "%smpfr_mul_si (%s, %s, %d, MPFR_RNDN);\n", indent, var1, var2, p):
    err := error_counter_of_a_multiplication(var1, var2, "", err):
    if(q<>1)
    then
      fprintf(fd, "%smpfr_div_si (%s, %s, %d, MPFR_RNDN);\n", indent, var1, var1, q):
      err := error_counter_of_a_multiplication(var1, var1, "", err):
    fi:
  else
    fprintf(fd, "%smpfr_div_si (%s, %s, %d, MPFR_RNDN);\n", indent, var1, var2, p*q):
    err := error_counter_of_a_multiplication(var1, var2, "", err):
  fi:
  return err:
end proc:


#
# generate_multiply_poly is the same as generate_multiply_rational but when r is a rational fraction.
# The fraction r must have the form p/q where p and q are polynomials with integer coefficients.
# Moreover, the gcd of the coefficients of p must be 1. Idem for q.
# The procedure returned a list [m, d, err] where m is the set of indices k such that
# a mpfr_mul_sik function is needed and idem for d with mpfr_div_sik.
# err is an up-to-date error counter.
generate_multiply_poly := proc(fd, var1, var2, r, error_counter, indent:="")
  local p,q,Lp,Lq,n,i,j,var, required_mulsi, required_divsi, err:
  err := error_counter:
  required_mulsi := {}:
  required_divsi := {}:
  p := numer(r): q := denom(r):
  Lp := my_factors(p): Lq := my_factors(q):
  if (Lp[1] <> 1)
    then error "generate_multiply_poly: an integer can be factored out of %1", p:
  fi:
  if (Lq[1] <> 1)
    then error "generate_multiply_poly: an integer can be factored out of %1", q:
  fi:
  Lp := Lp[2]: Lq := Lq[2]:
  var := var2:
  if (nops(Lp) <> 0)
  then
    n := 0:
    for i from 1 to nops(Lp) do n := n + Lp[i][2] od:
    if (n=1)
    then
      fprintf(fd, "%smpfr_mul_si (%s, %s", indent, var1, var):
    else
      required_mulsi := { op(required_mulsi), n }:
      fprintf(fd, "%smpfr_mul_si%d (%s, %s", indent, n, var1, var):
    fi:
    for i from 1 to nops(Lp) do
      for j from 1 to Lp[i][2] do
        fprintf(fd, ", %a", Lp[i][1]):
      od:
    od:
    fprintf(fd, ", MPFR_RNDN);\n"):
    err := set_error_counter(var1, n+find_in_error_counter(var, err) , err):
    var := var1:
  fi:
  if (nops(Lq) <> 0)
  then
    n := 0:
    for i from 1 to nops(Lq) do n := n + Lq[i][2] od:
    if (n=1)
    then
      fprintf(fd, "%smpfr_div_si (%s, %s", indent, var1, var):
    else
      required_divsi := { op(required_divsi), n }:
      fprintf(fd, "%smpfr_div_si%d (%s, %s", indent, n, var1, var)
    fi:
    for i from 1 to nops(Lq) do
      for j from 1 to Lq[i][2] do
        fprintf(fd, ", %a", Lq[i][1])
      od:
    od:
    fprintf(fd, ", MPFR_RNDN);\n"):
    err := set_error_counter(var1, n+find_in_error_counter(var, err) , err):
    var := var1:
  fi:
  if (var1 <> var) then
    fprintf(fd, "%smpfr_set (%s, %s, MPFR_RNDN);\n", indent, var1, var):
    err := set_error_counter(var1, find_in_error_counter(var, err) , err):
  fi:
  return [required_mulsi, required_divsi, err]:
end proc:


#
# This function generates the code of a procedure mpfr_mul_uin or mpfr_div_uin
#
generate_muldivsin := proc(op, n)
  local i, var:
  if ((op <> "mul") and (op <> "div"))
    then error "Invalid argument to generate_muldivuin (%1). Must be 'mul' or 'div'", op
  fi:
  if (whattype(n) <> 'integer')
  then error "Invalid argument to generate_muldivuin (%1). Must be an integer.", n
  fi:

  if (op="mul") then var := "MUL" else var := "DIV" fi:

  printf("__MPFR_DECLSPEC void mpfr_div_si%d _MPFR_PROTO((mpfr_ptr, mpfr_srcptr,\n", n):
  for i from n to 2 by -2 do
    printf("                                               long int, long int,\n"):
  od:
  if (i=1)
  then
    printf("                                               long int, mpfr_rnd_t));\n"):
  else
    printf("                                               mpfr_rnd_t));\n")
  fi:

  printf("\n\n\n"):
  printf("void\n"):
  printf("mpfr_%s_si%d (mpfr_ptr y, mpfr_srcptr x,\n", op, n):
  for i from n to 2 by -2 do
    printf("              long int v%d, long int v%d,\n", n-i+1, n-i+2):
  od:
  if (i=1)
  then
    printf("              long int v%d, mpfr_rnd_t mode)\n", n):
  else
    printf("              mpfr_rnd_t mode)\n")
  fi:
  printf("{\n"):
  printf("  long int acc = v1;\n"):
  printf("  mpfr_set (y, x, mode);\n"):
   for i from 2 to n do
    printf("  MPFR_ACC_OR_%s (v%d);\n", var, i):
  od:
  printf("  mpfr_%s_si (y, y, acc, mode);\n", op):
  printf("}\n"):
  return:
end proc:
