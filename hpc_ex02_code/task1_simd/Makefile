all: force_vectorized force1d_optimized

force1d_vectorized: force_vectorized.c
	gcc -o force1d_vectorized force1d_vectorized.c -mavx

force1d_optimized: force_vectorized.c
	gcc -Ofast -mavx -o force1d_optimized force1d_vectorized.c

.PHONY:
clean:
	rm -f force1d_vectorized force1d_optimized
