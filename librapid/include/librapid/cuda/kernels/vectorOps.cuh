#ifndef LIBRAPID_CUDA_VECTOR_OPS_HELPER
#define LIBRAPID_CUDA_VECTOR_OPS_HELPER

#define FL float
#define DL double
#define CO const

#define F2 float2
#define F3 float3
#define F4 float4
#define D2 double2
#define D3 double3
#define D4 double4

#define MF2 return make_float2
#define MF3 return make_float3
#define MF4 return make_float4
#define MD2 return make_double2
#define MD3 return make_double3
#define MD4 return make_double4

#define MF2N return a = make_float2
#define MF3N return a = make_float3
#define MF4N return a = make_float4
#define MD2N return a = make_double2
#define MD3N return a = make_double3
#define MD4N return a = make_double4

#define IOF(R, O) inline R operator O

IOF(F2,+)(CO F2 &a,CO F2 &b){MF2(a.x+b.x,a.y+b.y);}
IOF(F2,+)(CO F2 &a,CO FL &b){MF2(a.x + b,a.y + b);}
IOF(F2,+)(CO FL &a,CO F2 &b){MF2(a + b.x,a + b.y);}
IOF(F3,+)(CO F3 &a,CO F3 &b){MF3(a.x+b.x,a.y+b.y,a.z+b.z);}
IOF(F3,+)(CO F3 &a,CO FL &b){MF3(a.x + b,a.y + b,a.z + b);}
IOF(F3,+)(CO FL &a,CO F3 &b){MF3(a + b.x,a + b.y,a + b.z);}
IOF(F4,+)(CO F4 &a,CO F4 &b){MF4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
IOF(F4,+)(CO F4 &a,CO FL &b){MF4(a.x + b,a.y + b,a.z + b,a.w + b);}
IOF(F4,+)(CO FL &a,CO F4 &b){MF4(a + b.x,a + b.y,a + b.z,a + b.w);}
IOF(F2,-)(CO F2 &a,CO F2 &b){MF2(a.x-b.x,a.y-b.y);}
IOF(F2,-)(CO F2 &a,CO FL &b){MF2(a.x - b,a.y - b);}
IOF(F2,-)(CO FL &a,CO F2 &b){MF2(a - b.x,a - b.y);}
IOF(F3,-)(CO F3 &a,CO F3 &b){MF3(a.x-b.x,a.y-b.y,a.z-b.z);}
IOF(F3,-)(CO F3 &a,CO FL &b){MF3(a.x - b,a.y - b,a.z - b);}
IOF(F3,-)(CO FL &a,CO F3 &b){MF3(a - b.x,a - b.y,a - b.z);}
IOF(F4,-)(CO F4 &a,CO F4 &b){MF4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);}
IOF(F4,-)(CO F4 &a,CO FL &b){MF4(a.x - b,a.y - b,a.z - b,a.w - b);}
IOF(F4,-)(CO FL &a,CO F4 &b){MF4(a - b.x,a - b.y,a - b.z,a - b.w);}
IOF(F2,*)(CO F2 &a,CO F2 &b){MF2(a.x*b.x,a.y*b.y);}
IOF(F2,*)(CO F2 &a,CO FL &b){MF2(a.x * b,a.y * b);}
IOF(F2,*)(CO FL &a,CO F2 &b){MF2(a * b.x,a * b.y);}
IOF(F3,*)(CO F3 &a,CO F3 &b){MF3(a.x*b.x,a.y*b.y,a.z*b.z);}
IOF(F3,*)(CO F3 &a,CO FL &b){MF3(a.x * b,a.y * b,a.z * b);}
IOF(F3,*)(CO FL &a,CO F3 &b){MF3(a * b.x,a * b.y,a * b.z);}
IOF(F4,*)(CO F4 &a,CO F4 &b){MF4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
IOF(F4,*)(CO F4 &a,CO FL &b){MF4(a.x * b,a.y * b,a.z * b,a.w * b);}
IOF(F4,*)(CO FL &a,CO F4 &b){MF4(a * b.x,a * b.y,a * b.z,a * b.w);}
IOF(F2,/)(CO F2 &a,CO F2 &b){MF2(a.x/b.x,a.y/b.y);}
IOF(F2,/)(CO F2 &a,CO FL &b){MF2(a.x / b,a.y / b);}
IOF(F2,/)(CO FL &a,CO F2 &b){MF2(a / b.x,a / b.y);}
IOF(F3,/)(CO F3 &a,CO F3 &b){MF3(a.x/b.x,a.y/b.y,a.z/b.z);}
IOF(F3,/)(CO F3 &a,CO FL &b){MF3(a.x / b,a.y / b,a.z / b);}
IOF(F3,/)(CO FL &a,CO F3 &b){MF3(a / b.x,a / b.y,a / b.z);}
IOF(F4,/)(CO F4 &a,CO F4 &b){MF4(a.x/b.x,a.y/b.y,a.z/b.z,a.w/b.w);}
IOF(F4,/)(CO F4 &a,CO FL &b){MF4(a.x / b,a.y / b,a.z / b,a.w / b);}
IOF(F4,/)(CO FL &a,CO F4 &b){MF4(a / b.x,a / b.y,a / b.z,a / b.w);}
IOF(F2,+=)(F2 &a,CO F2 &b){MF2N(a.x+b.x,a.y+b.y);}
IOF(F2,+=)(F2 &a,CO FL &b){MF2N(a.x + b,a.y + b);}
IOF(F3,+=)(F3 &a,CO F3 &b){MF3N(a.x+b.x,a.y+b.y,a.z+b.z);}
IOF(F3,+=)(F3 &a,CO FL &b){MF3N(a.x + b,a.y + b,a.z + b);}
IOF(F4,+=)(F4 &a,CO F4 &b){MF4N(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
IOF(F4,+=)(F4 &a,CO FL &b){MF4N(a.x + b,a.y + b,a.z + b,a.w + b);}
IOF(F2,-=)(F2 &a,CO F2 &b){MF2N(a.x-b.x,a.y-b.y);}
IOF(F2,-=)(F2 &a,CO FL &b){MF2N(a.x - b,a.y - b);}
IOF(F3,-=)(F3 &a,CO F3 &b){MF3N(a.x-b.x,a.y-b.y,a.z-b.z);}
IOF(F3,-=)(F3 &a,CO FL &b){MF3N(a.x - b,a.y - b,a.z - b);}
IOF(F4,-=)(F4 &a,CO F4 &b){MF4N(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);}
IOF(F4,-=)(F4 &a,CO FL &b){MF4N(a.x - b,a.y - b,a.z - b,a.w - b);}
IOF(F2,*=)(F2 &a,CO F2 &b){MF2N(a.x*b.x,a.y*b.y);}
IOF(F2,*=)(F2 &a,CO FL &b){MF2N(a.x * b,a.y * b);}
IOF(F3,*=)(F3 &a,CO F3 &b){MF3N(a.x*b.x,a.y*b.y,a.z*b.z);}
IOF(F3,*=)(F3 &a,CO FL &b){MF3N(a.x * b,a.y * b,a.z * b);}
IOF(F4,*=)(F4 &a,CO F4 &b){MF4N(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
IOF(F4,*=)(F4 &a,CO FL &b){MF4N(a.x * b,a.y * b,a.z * b,a.w * b);}
IOF(F2,/=)(F2 &a,CO F2 &b){MF2N(a.x/b.x,a.y/b.y);}
IOF(F2,/=)(F2 &a,CO FL &b){MF2N(a.x / b,a.y / b);}
IOF(F3,/=)(F3 &a,CO F3 &b){MF3N(a.x/b.x,a.y/b.y,a.z/b.z);}
IOF(F3,/=)(F3 &a,CO FL &b){MF3N(a.x / b,a.y / b,a.z / b);}
IOF(F4,/=)(F4 &a,CO F4 &b){MF4N(a.x/b.x,a.y/b.y,a.z/b.z,a.w/b.w);}
IOF(F4,/=)(F4 &a,CO FL &b){MF4N(a.x / b,a.y / b,a.z / b,a.w / b);}
IOF(F2,>)(CO F2 &a,CO F2 &b){MF2(a.x>b.x,a.y>b.y);}
IOF(F2,>)(CO F2 &a,CO FL &b){MF2(a.x > b,a.y > b);}
IOF(F2,>)(CO FL &a,CO F2 &b){MF2(a > b.x,a > b.y);}
IOF(F3,>)(CO F3 &a,CO F3 &b){MF3(a.x>b.x,a.y>b.y,a.z>b.z);}
IOF(F3,>)(CO F3 &a,CO FL &b){MF3(a.x > b,a.y > b,a.z > b);}
IOF(F3,>)(CO FL &a,CO F3 &b){MF3(a > b.x,a > b.y,a > b.z);}
IOF(F4,>)(CO F4 &a,CO F4 &b){MF4(a.x>b.x,a.y>b.y,a.z>b.z,a.w>b.w);}
IOF(F4,>)(CO F4 &a,CO FL &b){MF4(a.x > b,a.y > b,a.z > b,a.w > b);}
IOF(F4,>)(CO FL &a,CO F4 &b){MF4(a > b.x,a > b.y,a > b.z,a > b.w);}
IOF(F2,<)(CO F2 &a,CO F2 &b){MF2(a.x<b.x,a.y<b.y);}
IOF(F2,<)(CO F2 &a,CO FL &b){MF2(a.x < b,a.y < b);}
IOF(F2,<)(CO FL &a,CO F2 &b){MF2(a < b.x,a < b.y);}
IOF(F3,<)(CO F3 &a,CO F3 &b){MF3(a.x<b.x,a.y<b.y,a.z<b.z);}
IOF(F3,<)(CO F3 &a,CO FL &b){MF3(a.x < b,a.y < b,a.z < b);}
IOF(F3,<)(CO FL &a,CO F3 &b){MF3(a < b.x,a < b.y,a < b.z);}
IOF(F4,<)(CO F4 &a,CO F4 &b){MF4(a.x<b.x,a.y<b.y,a.z<b.z,a.w<b.w);}
IOF(F4,<)(CO F4 &a,CO FL &b){MF4(a.x < b,a.y < b,a.z < b,a.w < b);}
IOF(F4,<)(CO FL &a,CO F4 &b){MF4(a < b.x,a < b.y,a < b.z,a < b.w);}
IOF(F2,>=)(CO F2 &a,CO F2 &b){MF2(a.x>=b.x,a.y>=b.y);}
IOF(F2,>=)(CO F2 &a,CO FL &b){MF2(a.x >= b,a.y >= b);}
IOF(F2,>=)(CO FL &a,CO F2 &b){MF2(a >= b.x,a >= b.y);}
IOF(F3,>=)(CO F3 &a,CO F3 &b){MF3(a.x>=b.x,a.y>=b.y,a.z>=b.z);}
IOF(F3,>=)(CO F3 &a,CO FL &b){MF3(a.x >= b,a.y >= b,a.z >= b);}
IOF(F3,>=)(CO FL &a,CO F3 &b){MF3(a >= b.x,a >= b.y,a >= b.z);}
IOF(F4,>=)(CO F4 &a,CO F4 &b){MF4(a.x>=b.x,a.y>=b.y,a.z>=b.z,a.w>=b.w);}
IOF(F4,>=)(CO F4 &a,CO FL &b){MF4(a.x >= b,a.y >= b,a.z >= b,a.w >= b);}
IOF(F4,>=)(CO FL &a,CO F4 &b){MF4(a >= b.x,a >= b.y,a >= b.z,a >= b.w);}
IOF(F2,<=)(CO F2 &a,CO F2 &b){MF2(a.x<=b.x,a.y<=b.y);}
IOF(F2,<=)(CO F2 &a,CO FL &b){MF2(a.x <= b,a.y <= b);}
IOF(F2,<=)(CO FL &a,CO F2 &b){MF2(a <= b.x,a <= b.y);}
IOF(F3,<=)(CO F3 &a,CO F3 &b){MF3(a.x<=b.x,a.y<=b.y,a.z<=b.z);}
IOF(F3,<=)(CO F3 &a,CO FL &b){MF3(a.x <= b,a.y <= b,a.z <= b);}
IOF(F3,<=)(CO FL &a,CO F3 &b){MF3(a <= b.x,a <= b.y,a <= b.z);}
IOF(F4,<=)(CO F4 &a,CO F4 &b){MF4(a.x<=b.x,a.y<=b.y,a.z<=b.z,a.w<=b.w);}
IOF(F4,<=)(CO F4 &a,CO FL &b){MF4(a.x <= b,a.y <= b,a.z <= b,a.w <= b);}
IOF(F4,<=)(CO FL &a,CO F4 &b){MF4(a <= b.x,a <= b.y,a <= b.z,a <= b.w);}
IOF(F2,==)(CO F2 &a,CO F2 &b){MF2(a.x==b.x,a.y==b.y);}
IOF(F2,==)(CO F2 &a,CO FL &b){MF2(a.x == b,a.y == b);}
IOF(F2,==)(CO FL &a,CO F2 &b){MF2(a == b.x,a == b.y);}
IOF(F3,==)(CO F3 &a,CO F3 &b){MF3(a.x==b.x,a.y==b.y,a.z==b.z);}
IOF(F3,==)(CO F3 &a,CO FL &b){MF3(a.x == b,a.y == b,a.z == b);}
IOF(F3,==)(CO FL &a,CO F3 &b){MF3(a == b.x,a == b.y,a == b.z);}
IOF(F4,==)(CO F4 &a,CO F4 &b){MF4(a.x==b.x,a.y==b.y,a.z==b.z,a.w==b.w);}
IOF(F4,==)(CO F4 &a,CO FL &b){MF4(a.x == b,a.y == b,a.z == b,a.w == b);}
IOF(F4,==)(CO FL &a,CO F4 &b){MF4(a == b.x,a == b.y,a == b.z,a == b.w);}
IOF(F2,!=)(CO F2 &a,CO F2 &b){MF2(a.x!=b.x,a.y!=b.y);}
IOF(F2,!=)(CO F2 &a,CO FL &b){MF2(a.x != b,a.y != b);}
IOF(F2,!=)(CO FL &a,CO F2 &b){MF2(a != b.x,a != b.y);}
IOF(F3,!=)(CO F3 &a,CO F3 &b){MF3(a.x!=b.x,a.y!=b.y,a.z!=b.z);}
IOF(F3,!=)(CO F3 &a,CO FL &b){MF3(a.x != b,a.y != b,a.z != b);}
IOF(F3,!=)(CO FL &a,CO F3 &b){MF3(a != b.x,a != b.y,a != b.z);}
IOF(F4,!=)(CO F4 &a,CO F4 &b){MF4(a.x!=b.x,a.y!=b.y,a.z!=b.z,a.w!=b.w);}
IOF(F4,!=)(CO F4 &a,CO FL &b){MF4(a.x != b,a.y != b,a.z != b,a.w != b);}
IOF(F4,!=)(CO FL &a,CO F4 &b){MF4(a != b.x,a != b.y,a != b.z,a != b.w);}
inline F2 sin(CO F2 &a){MF2(sin(a.x),sin(a.y));}
inline F3 sin(CO F3 &a){MF3(sin(a.x),sin(a.y),sin(a.z));}
inline F4 sin(CO F4 &a){MF4(sin(a.x),sin(a.y),sin(a.z),sin(a.w));}
inline F2 cos(CO F2 &a){MF2(cos(a.x),cos(a.y));}
inline F3 cos(CO F3 &a){MF3(cos(a.x),cos(a.y),cos(a.z));}
inline F4 cos(CO F4 &a){MF4(cos(a.x),cos(a.y),cos(a.z),cos(a.w));}
inline F2 tan(CO F2 &a){MF2(tan(a.x),tan(a.y));}
inline F3 tan(CO F3 &a){MF3(tan(a.x),tan(a.y),tan(a.z));}
inline F4 tan(CO F4 &a){MF4(tan(a.x),tan(a.y),tan(a.z),tan(a.w));}
inline F2 asin(CO F2 &a){MF2(asin(a.x),asin(a.y));}
inline F3 asin(CO F3 &a){MF3(asin(a.x),asin(a.y),asin(a.z));}
inline F4 asin(CO F4 &a){MF4(asin(a.x),asin(a.y),asin(a.z),asin(a.w));}
inline F2 acos(CO F2 &a){MF2(acos(a.x),acos(a.y));}
inline F3 acos(CO F3 &a){MF3(acos(a.x),acos(a.y),acos(a.z));}
inline F4 acos(CO F4 &a){MF4(acos(a.x),acos(a.y),acos(a.z),acos(a.w));}
inline F2 atan(CO F2 &a){MF2(atan(a.x),atan(a.y));}
inline F3 atan(CO F3 &a){MF3(atan(a.x),atan(a.y),atan(a.z));}
inline F4 atan(CO F4 &a){MF4(atan(a.x),atan(a.y),atan(a.z),atan(a.w));}
inline F2 sinh(CO F2 &a){MF2(sinh(a.x),sinh(a.y));}
inline F3 sinh(CO F3 &a){MF3(sinh(a.x),sinh(a.y),sinh(a.z));}
inline F4 sinh(CO F4 &a){MF4(sinh(a.x),sinh(a.y),sinh(a.z),sinh(a.w));}
inline F2 cosh(CO F2 &a){MF2(cosh(a.x),cosh(a.y));}
inline F3 cosh(CO F3 &a){MF3(cosh(a.x),cosh(a.y),cosh(a.z));}
inline F4 cosh(CO F4 &a){MF4(cosh(a.x),cosh(a.y),cosh(a.z),cosh(a.w));}
inline F2 tanh(CO F2 &a){MF2(tanh(a.x),tanh(a.y));}
inline F3 tanh(CO F3 &a){MF3(tanh(a.x),tanh(a.y),tanh(a.z));}
inline F4 tanh(CO F4 &a){MF4(tanh(a.x),tanh(a.y),tanh(a.z),tanh(a.w));}
inline F2 asinh(CO F2 &a){MF2(asinh(a.x),asinh(a.y));}
inline F3 asinh(CO F3 &a){MF3(asinh(a.x),asinh(a.y),asinh(a.z));}
inline F4 asinh(CO F4 &a){MF4(asinh(a.x),asinh(a.y),asinh(a.z),asinh(a.w));}
inline F2 acosh(CO F2 &a){MF2(acosh(a.x),acosh(a.y));}
inline F3 acosh(CO F3 &a){MF3(acosh(a.x),acosh(a.y),acosh(a.z));}
inline F4 acosh(CO F4 &a){MF4(acosh(a.x),acosh(a.y),acosh(a.z),acosh(a.w));}
inline F2 atanh(CO F2 &a){MF2(atanh(a.x),atanh(a.y));}
inline F3 atanh(CO F3 &a){MF3(atanh(a.x),atanh(a.y),atanh(a.z));}
inline F4 atanh(CO F4 &a){MF4(atanh(a.x),atanh(a.y),atanh(a.z),atanh(a.w));}
inline F2 exp(CO F2 &a){MF2(exp(a.x),exp(a.y));}
inline F3 exp(CO F3 &a){MF3(exp(a.x),exp(a.y),exp(a.z));}
inline F4 exp(CO F4 &a){MF4(exp(a.x),exp(a.y),exp(a.z),exp(a.w));}
inline F2 log(CO F2 &a){MF2(log(a.x),log(a.y));}
inline F3 log(CO F3 &a){MF3(log(a.x),log(a.y),log(a.z));}
inline F4 log(CO F4 &a){MF4(log(a.x),log(a.y),log(a.z),log(a.w));}
inline F2 sqrt(CO F2 &a){MF2(sqrt(a.x),sqrt(a.y));}
inline F3 sqrt(CO F3 &a){MF3(sqrt(a.x),sqrt(a.y),sqrt(a.z));}
inline F4 sqrt(CO F4 &a){MF4(sqrt(a.x),sqrt(a.y),sqrt(a.z),sqrt(a.w));}
inline F2 cbrt(CO F2 &a){MF2(cbrt(a.x),cbrt(a.y));}
inline F3 cbrt(CO F3 &a){MF3(cbrt(a.x),cbrt(a.y),cbrt(a.z));}
inline F4 cbrt(CO F4 &a){MF4(cbrt(a.x),cbrt(a.y),cbrt(a.z),cbrt(a.w));}
inline F2 abs(CO F2 &a){MF2(abs(a.x),abs(a.y));}
inline F3 abs(CO F3 &a){MF3(abs(a.x),abs(a.y),abs(a.z));}
inline F4 abs(CO F4 &a){MF4(abs(a.x),abs(a.y),abs(a.z),abs(a.w));}
inline F2 ceil(CO F2 &a){MF2(ceil(a.x),ceil(a.y));}
inline F3 ceil(CO F3 &a){MF3(ceil(a.x),ceil(a.y),ceil(a.z));}
inline F4 ceil(CO F4 &a){MF4(ceil(a.x),ceil(a.y),ceil(a.z),ceil(a.w));}
inline F2 floor(CO F2 &a){MF2(floor(a.x),floor(a.y));}
inline F3 floor(CO F3 &a){MF3(floor(a.x),floor(a.y),floor(a.z));}
inline F4 floor(CO F4 &a){MF4(floor(a.x),floor(a.y),floor(a.z),floor(a.w));}
IOF(D2,+)(CO D2 &a,CO D2 &b){MD2(a.x+b.x,a.y+b.y);}
IOF(D2,+)(CO D2 &a,CO DL &b){MD2(a.x + b,a.y + b);}
IOF(D2,+)(CO DL &a,CO D2 &b){MD2(a + b.x,a + b.y);}
IOF(D3,+)(CO D3 &a,CO D3 &b){MD3(a.x+b.x,a.y+b.y,a.z+b.z);}
IOF(D3,+)(CO D3 &a,CO DL &b){MD3(a.x + b,a.y + b,a.z + b);}
IOF(D3,+)(CO DL &a,CO D3 &b){MD3(a + b.x,a + b.y,a + b.z);}
IOF(D4,+)(CO D4 &a,CO D4 &b){MD4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
IOF(D4,+)(CO D4 &a,CO DL &b){MD4(a.x + b,a.y + b,a.z + b,a.w + b);}
IOF(D4,+)(CO DL &a,CO D4 &b){MD4(a + b.x,a + b.y,a + b.z,a + b.w);}
IOF(D2,-)(CO D2 &a,CO D2 &b){MD2(a.x-b.x,a.y-b.y);}
IOF(D2,-)(CO D2 &a,CO DL &b){MD2(a.x - b,a.y - b);}
IOF(D2,-)(CO DL &a,CO D2 &b){MD2(a - b.x,a - b.y);}
IOF(D3,-)(CO D3 &a,CO D3 &b){MD3(a.x-b.x,a.y-b.y,a.z-b.z);}
IOF(D3,-)(CO D3 &a,CO DL &b){MD3(a.x - b,a.y - b,a.z - b);}
IOF(D3,-)(CO DL &a,CO D3 &b){MD3(a - b.x,a - b.y,a - b.z);}
IOF(D4,-)(CO D4 &a,CO D4 &b){MD4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);}
IOF(D4,-)(CO D4 &a,CO DL &b){MD4(a.x - b,a.y - b,a.z - b,a.w - b);}
IOF(D4,-)(CO DL &a,CO D4 &b){MD4(a - b.x,a - b.y,a - b.z,a - b.w);}
IOF(D2,*)(CO D2 &a,CO D2 &b){MD2(a.x*b.x,a.y*b.y);}
IOF(D2,*)(CO D2 &a,CO DL &b){MD2(a.x * b,a.y * b);}
IOF(D2,*)(CO DL &a,CO D2 &b){MD2(a * b.x,a * b.y);}
IOF(D3,*)(CO D3 &a,CO D3 &b){MD3(a.x*b.x,a.y*b.y,a.z*b.z);}
IOF(D3,*)(CO D3 &a,CO DL &b){MD3(a.x * b,a.y * b,a.z * b);}
IOF(D3,*)(CO DL &a,CO D3 &b){MD3(a * b.x,a * b.y,a * b.z);}
IOF(D4,*)(CO D4 &a,CO D4 &b){MD4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
IOF(D4,*)(CO D4 &a,CO DL &b){MD4(a.x * b,a.y * b,a.z * b,a.w * b);}
IOF(D4,*)(CO DL &a,CO D4 &b){MD4(a * b.x,a * b.y,a * b.z,a * b.w);}
IOF(D2,/)(CO D2 &a,CO D2 &b){MD2(a.x/b.x,a.y/b.y);}
IOF(D2,/)(CO D2 &a,CO DL &b){MD2(a.x / b,a.y / b);}
IOF(D2,/)(CO DL &a,CO D2 &b){MD2(a / b.x,a / b.y);}
IOF(D3,/)(CO D3 &a,CO D3 &b){MD3(a.x/b.x,a.y/b.y,a.z/b.z);}
IOF(D3,/)(CO D3 &a,CO DL &b){MD3(a.x / b,a.y / b,a.z / b);}
IOF(D3,/)(CO DL &a,CO D3 &b){MD3(a / b.x,a / b.y,a / b.z);}
IOF(D4,/)(CO D4 &a,CO D4 &b){MD4(a.x/b.x,a.y/b.y,a.z/b.z,a.w/b.w);}
IOF(D4,/)(CO D4 &a,CO DL &b){MD4(a.x / b,a.y / b,a.z / b,a.w / b);}
IOF(D4,/)(CO DL &a,CO D4 &b){MD4(a / b.x,a / b.y,a / b.z,a / b.w);}
IOF(D2,+=)(D2 &a,CO D2 &b){MD2N(a.x+b.x,a.y+b.y);}
IOF(D2,+=)(D2 &a,CO DL &b){MD2N(a.x + b,a.y + b);}
IOF(D3,+=)(D3 &a,CO D3 &b){MD3N(a.x+b.x,a.y+b.y,a.z+b.z);}
IOF(D3,+=)(D3 &a,CO DL &b){MD3N(a.x + b,a.y + b,a.z + b);}
IOF(D4,+=)(D4 &a,CO D4 &b){MD4N(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
IOF(D4,+=)(D4 &a,CO DL &b){MD4N(a.x + b,a.y + b,a.z + b,a.w + b);}
IOF(D2,-=)(D2 &a,CO D2 &b){MD2N(a.x-b.x,a.y-b.y);}
IOF(D2,-=)(D2 &a,CO DL &b){MD2N(a.x - b,a.y - b);}
IOF(D3,-=)(D3 &a,CO D3 &b){MD3N(a.x-b.x,a.y-b.y,a.z-b.z);}
IOF(D3,-=)(D3 &a,CO DL &b){MD3N(a.x - b,a.y - b,a.z - b);}
IOF(D4,-=)(D4 &a,CO D4 &b){MD4N(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);}
IOF(D4,-=)(D4 &a,CO DL &b){MD4N(a.x - b,a.y - b,a.z - b,a.w - b);}
IOF(D2,*=)(D2 &a,CO D2 &b){MD2N(a.x*b.x,a.y*b.y);}
IOF(D2,*=)(D2 &a,CO DL &b){MD2N(a.x * b,a.y * b);}
IOF(D3,*=)(D3 &a,CO D3 &b){MD3N(a.x*b.x,a.y*b.y,a.z*b.z);}
IOF(D3,*=)(D3 &a,CO DL &b){MD3N(a.x * b,a.y * b,a.z * b);}
IOF(D4,*=)(D4 &a,CO D4 &b){MD4N(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
IOF(D4,*=)(D4 &a,CO DL &b){MD4N(a.x * b,a.y * b,a.z * b,a.w * b);}
IOF(D2,/=)(D2 &a,CO D2 &b){MD2N(a.x/b.x,a.y/b.y);}
IOF(D2,/=)(D2 &a,CO DL &b){MD2N(a.x / b,a.y / b);}
IOF(D3,/=)(D3 &a,CO D3 &b){MD3N(a.x/b.x,a.y/b.y,a.z/b.z);}
IOF(D3,/=)(D3 &a,CO DL &b){MD3N(a.x / b,a.y / b,a.z / b);}
IOF(D4,/=)(D4 &a,CO D4 &b){MD4N(a.x/b.x,a.y/b.y,a.z/b.z,a.w/b.w);}
IOF(D4,/=)(D4 &a,CO DL &b){MD4N(a.x / b,a.y / b,a.z / b,a.w / b);}
IOF(D2,>)(CO D2 &a,CO D2 &b){MD2(a.x>b.x,a.y>b.y);}
IOF(D2,>)(CO D2 &a,CO DL &b){MD2(a.x > b,a.y > b);}
IOF(D2,>)(CO DL &a,CO D2 &b){MD2(a > b.x,a > b.y);}
IOF(D3,>)(CO D3 &a,CO D3 &b){MD3(a.x>b.x,a.y>b.y,a.z>b.z);}
IOF(D3,>)(CO D3 &a,CO DL &b){MD3(a.x > b,a.y > b,a.z > b);}
IOF(D3,>)(CO DL &a,CO D3 &b){MD3(a > b.x,a > b.y,a > b.z);}
IOF(D4,>)(CO D4 &a,CO D4 &b){MD4(a.x>b.x,a.y>b.y,a.z>b.z,a.w>b.w);}
IOF(D4,>)(CO D4 &a,CO DL &b){MD4(a.x > b,a.y > b,a.z > b,a.w > b);}
IOF(D4,>)(CO DL &a,CO D4 &b){MD4(a > b.x,a > b.y,a > b.z,a > b.w);}
IOF(D2,<)(CO D2 &a,CO D2 &b){MD2(a.x<b.x,a.y<b.y);}
IOF(D2,<)(CO D2 &a,CO DL &b){MD2(a.x < b,a.y < b);}
IOF(D2,<)(CO DL &a,CO D2 &b){MD2(a < b.x,a < b.y);}
IOF(D3,<)(CO D3 &a,CO D3 &b){MD3(a.x<b.x,a.y<b.y,a.z<b.z);}
IOF(D3,<)(CO D3 &a,CO DL &b){MD3(a.x < b,a.y < b,a.z < b);}
IOF(D3,<)(CO DL &a,CO D3 &b){MD3(a < b.x,a < b.y,a < b.z);}
IOF(D4,<)(CO D4 &a,CO D4 &b){MD4(a.x<b.x,a.y<b.y,a.z<b.z,a.w<b.w);}
IOF(D4,<)(CO D4 &a,CO DL &b){MD4(a.x < b,a.y < b,a.z < b,a.w < b);}
IOF(D4,<)(CO DL &a,CO D4 &b){MD4(a < b.x,a < b.y,a < b.z,a < b.w);}
IOF(D2,>=)(CO D2 &a,CO D2 &b){MD2(a.x>=b.x,a.y>=b.y);}
IOF(D2,>=)(CO D2 &a,CO DL &b){MD2(a.x >= b,a.y >= b);}
IOF(D2,>=)(CO DL &a,CO D2 &b){MD2(a >= b.x,a >= b.y);}
IOF(D3,>=)(CO D3 &a,CO D3 &b){MD3(a.x>=b.x,a.y>=b.y,a.z>=b.z);}
IOF(D3,>=)(CO D3 &a,CO DL &b){MD3(a.x >= b,a.y >= b,a.z >= b);}
IOF(D3,>=)(CO DL &a,CO D3 &b){MD3(a >= b.x,a >= b.y,a >= b.z);}
IOF(D4,>=)(CO D4 &a,CO D4 &b){MD4(a.x>=b.x,a.y>=b.y,a.z>=b.z,a.w>=b.w);}
IOF(D4,>=)(CO D4 &a,CO DL &b){MD4(a.x >= b,a.y >= b,a.z >= b,a.w >= b);}
IOF(D4,>=)(CO DL &a,CO D4 &b){MD4(a >= b.x,a >= b.y,a >= b.z,a >= b.w);}
IOF(D2,<=)(CO D2 &a,CO D2 &b){MD2(a.x<=b.x,a.y<=b.y);}
IOF(D2,<=)(CO D2 &a,CO DL &b){MD2(a.x <= b,a.y <= b);}
IOF(D2,<=)(CO DL &a,CO D2 &b){MD2(a <= b.x,a <= b.y);}
IOF(D3,<=)(CO D3 &a,CO D3 &b){MD3(a.x<=b.x,a.y<=b.y,a.z<=b.z);}
IOF(D3,<=)(CO D3 &a,CO DL &b){MD3(a.x <= b,a.y <= b,a.z <= b);}
IOF(D3,<=)(CO DL &a,CO D3 &b){MD3(a <= b.x,a <= b.y,a <= b.z);}
IOF(D4,<=)(CO D4 &a,CO D4 &b){MD4(a.x<=b.x,a.y<=b.y,a.z<=b.z,a.w<=b.w);}
IOF(D4,<=)(CO D4 &a,CO DL &b){MD4(a.x <= b,a.y <= b,a.z <= b,a.w <= b);}
IOF(D4,<=)(CO DL &a,CO D4 &b){MD4(a <= b.x,a <= b.y,a <= b.z,a <= b.w);}
IOF(D2,==)(CO D2 &a,CO D2 &b){MD2(a.x==b.x,a.y==b.y);}
IOF(D2,==)(CO D2 &a,CO DL &b){MD2(a.x == b,a.y == b);}
IOF(D2,==)(CO DL &a,CO D2 &b){MD2(a == b.x,a == b.y);}
IOF(D3,==)(CO D3 &a,CO D3 &b){MD3(a.x==b.x,a.y==b.y,a.z==b.z);}
IOF(D3,==)(CO D3 &a,CO DL &b){MD3(a.x == b,a.y == b,a.z == b);}
IOF(D3,==)(CO DL &a,CO D3 &b){MD3(a == b.x,a == b.y,a == b.z);}
IOF(D4,==)(CO D4 &a,CO D4 &b){MD4(a.x==b.x,a.y==b.y,a.z==b.z,a.w==b.w);}
IOF(D4,==)(CO D4 &a,CO DL &b){MD4(a.x == b,a.y == b,a.z == b,a.w == b);}
IOF(D4,==)(CO DL &a,CO D4 &b){MD4(a == b.x,a == b.y,a == b.z,a == b.w);}
IOF(D2,!=)(CO D2 &a,CO D2 &b){MD2(a.x!=b.x,a.y!=b.y);}
IOF(D2,!=)(CO D2 &a,CO DL &b){MD2(a.x != b,a.y != b);}
IOF(D2,!=)(CO DL &a,CO D2 &b){MD2(a != b.x,a != b.y);}
IOF(D3,!=)(CO D3 &a,CO D3 &b){MD3(a.x!=b.x,a.y!=b.y,a.z!=b.z);}
IOF(D3,!=)(CO D3 &a,CO DL &b){MD3(a.x != b,a.y != b,a.z != b);}
IOF(D3,!=)(CO DL &a,CO D3 &b){MD3(a != b.x,a != b.y,a != b.z);}
IOF(D4,!=)(CO D4 &a,CO D4 &b){MD4(a.x!=b.x,a.y!=b.y,a.z!=b.z,a.w!=b.w);}
IOF(D4,!=)(CO D4 &a,CO DL &b){MD4(a.x != b,a.y != b,a.z != b,a.w != b);}
IOF(D4,!=)(CO DL &a,CO D4 &b){MD4(a != b.x,a != b.y,a != b.z,a != b.w);}
inline D2 sin(CO D2 &a){MD2(sin(a.x),sin(a.y));}
inline D3 sin(CO D3 &a){MD3(sin(a.x),sin(a.y),sin(a.z));}
inline D4 sin(CO D4 &a){MD4(sin(a.x),sin(a.y),sin(a.z),sin(a.w));}
inline D2 cos(CO D2 &a){MD2(cos(a.x),cos(a.y));}
inline D3 cos(CO D3 &a){MD3(cos(a.x),cos(a.y),cos(a.z));}
inline D4 cos(CO D4 &a){MD4(cos(a.x),cos(a.y),cos(a.z),cos(a.w));}
inline D2 tan(CO D2 &a){MD2(tan(a.x),tan(a.y));}
inline D3 tan(CO D3 &a){MD3(tan(a.x),tan(a.y),tan(a.z));}
inline D4 tan(CO D4 &a){MD4(tan(a.x),tan(a.y),tan(a.z),tan(a.w));}
inline D2 asin(CO D2 &a){MD2(asin(a.x),asin(a.y));}
inline D3 asin(CO D3 &a){MD3(asin(a.x),asin(a.y),asin(a.z));}
inline D4 asin(CO D4 &a){MD4(asin(a.x),asin(a.y),asin(a.z),asin(a.w));}
inline D2 acos(CO D2 &a){MD2(acos(a.x),acos(a.y));}
inline D3 acos(CO D3 &a){MD3(acos(a.x),acos(a.y),acos(a.z));}
inline D4 acos(CO D4 &a){MD4(acos(a.x),acos(a.y),acos(a.z),acos(a.w));}
inline D2 atan(CO D2 &a){MD2(atan(a.x),atan(a.y));}
inline D3 atan(CO D3 &a){MD3(atan(a.x),atan(a.y),atan(a.z));}
inline D4 atan(CO D4 &a){MD4(atan(a.x),atan(a.y),atan(a.z),atan(a.w));}
inline D2 sinh(CO D2 &a){MD2(sinh(a.x),sinh(a.y));}
inline D3 sinh(CO D3 &a){MD3(sinh(a.x),sinh(a.y),sinh(a.z));}
inline D4 sinh(CO D4 &a){MD4(sinh(a.x),sinh(a.y),sinh(a.z),sinh(a.w));}
inline D2 cosh(CO D2 &a){MD2(cosh(a.x),cosh(a.y));}
inline D3 cosh(CO D3 &a){MD3(cosh(a.x),cosh(a.y),cosh(a.z));}
inline D4 cosh(CO D4 &a){MD4(cosh(a.x),cosh(a.y),cosh(a.z),cosh(a.w));}
inline D2 tanh(CO D2 &a){MD2(tanh(a.x),tanh(a.y));}
inline D3 tanh(CO D3 &a){MD3(tanh(a.x),tanh(a.y),tanh(a.z));}
inline D4 tanh(CO D4 &a){MD4(tanh(a.x),tanh(a.y),tanh(a.z),tanh(a.w));}
inline D2 asinh(CO D2 &a){MD2(asinh(a.x),asinh(a.y));}
inline D3 asinh(CO D3 &a){MD3(asinh(a.x),asinh(a.y),asinh(a.z));}
inline D4 asinh(CO D4 &a){MD4(asinh(a.x),asinh(a.y),asinh(a.z),asinh(a.w));}
inline D2 acosh(CO D2 &a){MD2(acosh(a.x),acosh(a.y));}
inline D3 acosh(CO D3 &a){MD3(acosh(a.x),acosh(a.y),acosh(a.z));}
inline D4 acosh(CO D4 &a){MD4(acosh(a.x),acosh(a.y),acosh(a.z),acosh(a.w));}
inline D2 atanh(CO D2 &a){MD2(atanh(a.x),atanh(a.y));}
inline D3 atanh(CO D3 &a){MD3(atanh(a.x),atanh(a.y),atanh(a.z));}
inline D4 atanh(CO D4 &a){MD4(atanh(a.x),atanh(a.y),atanh(a.z),atanh(a.w));}
inline D2 exp(CO D2 &a){MD2(exp(a.x),exp(a.y));}
inline D3 exp(CO D3 &a){MD3(exp(a.x),exp(a.y),exp(a.z));}
inline D4 exp(CO D4 &a){MD4(exp(a.x),exp(a.y),exp(a.z),exp(a.w));}
inline D2 log(CO D2 &a){MD2(log(a.x),log(a.y));}
inline D3 log(CO D3 &a){MD3(log(a.x),log(a.y),log(a.z));}
inline D4 log(CO D4 &a){MD4(log(a.x),log(a.y),log(a.z),log(a.w));}
inline D2 sqrt(CO D2 &a){MD2(sqrt(a.x),sqrt(a.y));}
inline D3 sqrt(CO D3 &a){MD3(sqrt(a.x),sqrt(a.y),sqrt(a.z));}
inline D4 sqrt(CO D4 &a){MD4(sqrt(a.x),sqrt(a.y),sqrt(a.z),sqrt(a.w));}
inline D2 cbrt(CO D2 &a){MD2(cbrt(a.x),cbrt(a.y));}
inline D3 cbrt(CO D3 &a){MD3(cbrt(a.x),cbrt(a.y),cbrt(a.z));}
inline D4 cbrt(CO D4 &a){MD4(cbrt(a.x),cbrt(a.y),cbrt(a.z),cbrt(a.w));}
inline D2 abs(CO D2 &a){MD2(abs(a.x),abs(a.y));}
inline D3 abs(CO D3 &a){MD3(abs(a.x),abs(a.y),abs(a.z));}
inline D4 abs(CO D4 &a){MD4(abs(a.x),abs(a.y),abs(a.z),abs(a.w));}
inline D2 ceil(CO D2 &a){MD2(ceil(a.x),ceil(a.y));}
inline D3 ceil(CO D3 &a){MD3(ceil(a.x),ceil(a.y),ceil(a.z));}
inline D4 ceil(CO D4 &a){MD4(ceil(a.x),ceil(a.y),ceil(a.z),ceil(a.w));}
inline D2 floor(CO D2 &a){MD2(floor(a.x),floor(a.y));}
inline D3 floor(CO D3 &a){MD3(floor(a.x),floor(a.y),floor(a.z));}
inline D4 floor(CO D4 &a){MD4(floor(a.x),floor(a.y),floor(a.z),floor(a.w));}

#endif // LIBRAPID_CUDA_VECTOR_OPS_HELPER