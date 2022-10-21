read ("metaMPFR_straightforwardAlgo.mpl"):

f := AiryAi(x):
deq := holexprtodiffeq(f, y(x)):
rec := diffeqtorec(deq, y(x), a(n)):
name_of_function := op(0,f):
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

f := erf(x):
deq := holexprtodiffeq(f, y(x)):
rec := diffeqtorec(deq, y(x), a(n)):
name_of_function := op(0,f):
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = -(6*n+1)*(6*n+2)*(6*n+3)*(6*n+4)*(6*n+5)*(6*n+6)*a(n)/( (n+1)^3*(3*n+1)*(3*n+2)*(3*n+3)*12288000 ), a(0)=1 }:
name_of_function := alpha:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):


rec := { a(n+1) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1 }:
name_of_function := test0a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=Pi }:
name_of_function := test1a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=1 }:
name_of_function := test2a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi }:
name_of_function := test3a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1, a(1)=2 }:
name_of_function := test4a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=Pi, a(1)=0}:
name_of_function := test5a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=1, a(1)=Pi }:
name_of_function := test6a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi, a(1)=0}:
name_of_function := test7a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+3) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1, a(1)=0, a(2)=0 }:
name_of_function := test8a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+3) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=0, a(1)=Pi, a(2)=2 }:
name_of_function := test9a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+3) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=0, a(1)=0, a(2)=1 }:
name_of_function := test10a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):

rec := { a(n+7) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi, a(4)=1, a(6)=2  }:
name_of_function := test11a:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), CONSTANT_SERIES, name_of_function, name_of_file, f):


rec := { a(n+1) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1 }:
name_of_function := test0b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=Pi }:
name_of_function := test1b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=1 }:
name_of_function := test2b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+1) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi }:
name_of_function := test3b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1, a(1)=2 }:
name_of_function := test4b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=Pi, a(1)=0}:
name_of_function := test5b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=1, a(1)=Pi }:
name_of_function := test6b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+2) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi, a(1)=0}:
name_of_function := test7b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+3) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1, a(1)=0, a(2)=0 }:
name_of_function := test8b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+3) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=0, a(1)=Pi, a(2)=2 }:
name_of_function := test9b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+3) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=0, a(1)=0, a(2)=1 }:
name_of_function := test10b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):

rec := { a(n+7) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi, a(4)=1, a(6)=2  }:
name_of_function := test11b:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES, name_of_function, name_of_file, f):


rec := { a(n+1) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1 }:
name_of_function := test0c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+1) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=Pi }:
name_of_function := test1c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+1) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=1 }:
name_of_function := test2c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+1) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi }:
name_of_function := test3c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+2) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1, a(1)=2 }:
name_of_function := test4c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+2) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=Pi, a(1)=0}:
name_of_function := test5c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+2) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=1, a(1)=Pi }:
name_of_function := test6c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+2) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi, a(1)=0}:
name_of_function := test7c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+3) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=1, a(1)=0, a(2)=0 }:
name_of_function := test8c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+3) = -(6*n+1)*a(n)/( (n+1)^3 ), a(0)=0, a(1)=Pi, a(2)=2 }:
name_of_function := test9c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+3) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=0, a(1)=0, a(2)=1 }:
name_of_function := test10c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

rec := { a(n+7) = (1/Pi)*(3*n+1)*a(n)/( (n+2) ), a(0)=Pi, a(4)=1, a(6)=2  }:
name_of_function := test11c:
name_of_file := sprintf("%a.c", name_of_function):
printf("\n\n\n/************************** Implémentation de %s ******************************/\n", name_of_file):
generateStraightforwardAlgo(rec, a(n), FUNCTION_SERIES_RATIONAL, name_of_function, name_of_file, f):

