all: forces_cpu forces_symmetrical forces_gpu

forces_symmetrical: forces_gpu.c
	gcc forces_gpu.c -o -lm

forces_cpu: forces_cpu.c
	gcc forces_cpu.c -g -O2 -Wall -o $@ -lm

forces_gpu: forces_gpu.cu
	nvcc forces_gpu.cu -o -lm -arch=sm_61

.PHONY: clean
clean:
	rm -f forces_cpu forces_symmetrical forces_gpu
