#!/bin/zsh

cd "${0%/*}"

octave <<EOF
n=32 * 97 * 89;
n2=floor(n/11);

n-=1;
n2-=1;

fd=fopen("reference-acos-dp.dat", "w");
fs=fopen("reference-acos-sp.dat", "w");
printf("Generating %d dp & sp acos values [0, 1]     ", n + 1);
for i = 0:n
  x=i/n;
  fwrite(fd, x, "double");
  fwrite(fd, acos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, acos(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

fd=fopen("reference-asin-dp.dat", "w");
fs=fopen("reference-asin-sp.dat", "w");
printf("\nGenerating %d dp & sp asin values [0, 1]     ", n + 1);
for i = 0:n
  x=i/n;
  fwrite(fd, x, "double");
  fwrite(fd, asin(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, asin(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

fd=fopen("reference-atan-dp.dat", "w");
fs=fopen("reference-atan-sp.dat", "w");
printf("\nGenerating %d dp & sp atan values [0, 10]     ", n + 1);
for i = 0:n
  x=i*10/n;
  fwrite(fd, x, "double");
  fwrite(fd, atan(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, atan(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

printf("\nGenerating %d dp & sp atan values [10, 10000]     ", n + 1);
for i = 0:n
  x=10+i*9990/n;
  fwrite(fd, x, "double");
  fwrite(fd, atan(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, atan(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

fd=fopen("reference-sincos-dp.dat", "w");
fs=fopen("reference-sincos-sp.dat", "w");

printf("\nGenerating %d dp & sp sincos values [0, 2π]     ", n + 1);
for i = 0:n
  x=(i/n) * 2 * pi;
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, sin(x), "single");
  fwrite(fs, cos(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

printf("\nGenerating %d dp & sp sincos values {π/4 ± small value}     ", n2 + 1);
for i = 0:n2
  x=pi/4 - 2^-20 + (i/n2) * 2^-19;
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, sin(x), "single");
  fwrite(fs, cos(x), "single");
  printf("\033[4D%3i%%", floor(i/n2*100));
endfor

printf("\nGenerating %d dp & sp sincos values {2π ± small value}     ", n2 + 1);
for i = 0:n2
  x=2*pi - 2^-20 + (i/n2) * 2^-19;
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, sin(x), "single");
  fwrite(fs, cos(x), "single");
  printf("\033[4D%3i%%", floor(i/n2*100));
endfor

printf("\nGenerating %d dp & sp sincos values {4π ± small value}     ", n2 + 1);
for i = 0:n2
  x=4*pi - 2^-20 + (i/n2) * 2^-19;
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, sin(x), "single");
  fwrite(fs, cos(x), "single");
  printf("\033[4D%3i%%", floor(i/n2*100));
endfor

printf("\nGenerating %d dp & sp sincos values [0, 0x1.0p10 ≈ 10³]     ", n2 + 1);
for i = 0:n2
  x=(i/n2) * 2^10;
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, sin(x), "single");
  fwrite(fs, cos(x), "single");
  printf("\033[4D%3i%%", floor(i/n2*100));
endfor

printf("\nGenerating %d dp & sp sincos values [10³, 0x1.0p20 ≈ 10⁶]     ", n2 + 1);
for i = 0:n2
  x=2^10 + (i/n2) * (2^20 - 2^10);
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, sin(x), "single");
  fwrite(fs, cos(x), "single");
  printf("\033[4D%3i%%", floor(i/n2*100));
endfor

printf("\nGenerating %d dp sincos values [10⁶, 10¹⁴ ≈ 0x1.7p46]     ", n + 1);
for i = 0:n
  x=2^20 + (i/n) * (10^8 - 2^20);
  fwrite(fd, x, "double");
  fwrite(fd, sin(x), "double");
  fwrite(fd, cos(x), "double");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

fd=fopen("reference-tan-dp.dat", "w");
fs=fopen("reference-tan-sp.dat", "w");
printf("\nGenerating %d dp & sp tan values [0, pi/2]     ", n + 1);
for i = 0:n
  x=i*pi/(2*n);
  fwrite(fd, x, "double");
  fwrite(fd, tan(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, tan(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

printf("\nGenerating %d dp & sp tan values [0, 10¹⁴]     ", n + 1);
for i = 0:n
  x=i*10^14/n;
  fwrite(fd, x, "double");
  fwrite(fd, tan(x), "double");
  x=round_to_float(x);
  fwrite(fs, x, "single");
  fwrite(fs, tan(x), "single");
  printf("\033[4D%3i%%", floor(i/n*100));
endfor

printf("\n");
EOF

# vim: sw=2 et
