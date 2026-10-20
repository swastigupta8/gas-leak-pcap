# core2d/Makefile - builds every 2D variant into core2d/build/.
#
#   make            CPU targets: gas2d (serial + OpenMP), gas2d_mpi, gas2d_emu, libgas2d_cpu.so
#   make cuda       GPU targets: gas2d_cuda, libgas2d_cuda.so          (needs nvcc)
#   make clean
#
# -ffp-contract=off stops the compiler from fusing a*b+c into one FMA instruction on its own.
# Without it the serial, OpenMP and MPI builds could round differently and the bit-for-bit
# equality tests would fail for reasons unrelated to parallelism.

CC      ?= gcc
MPICC   ?= mpicc
NVCC    ?= nvcc
# Default sm_75 (Tesla T4, used by CI). `make build` at the repo root passes CUDA_ARCH=native,
# which compiles for whichever NVIDIA GPU is installed.
CUDA_ARCH ?= sm_75

# ARCH is separate so CI can pick a portable target without replacing the flags below (a
# command-line CFLAGS=... would silently discard every += in this file).
ARCH    ?= -march=native
CFLAGS   = -O3 $(ARCH) -std=gnu11 -Wall -Wextra -ffp-contract=off -Iinclude
OMP      = -fopenmp
LDLIBS   = -lm
NVFLAGS ?= -O3 -arch=$(CUDA_ARCH) -lineinfo
NVFLAGS += -Iinclude

B = build
COMMON = src/common.c src/scenario.c

all: $(B)/gas2d $(B)/gas2d_mpi $(B)/gas2d_emu $(B)/libgas2d_cpu.so

$(B):
	mkdir -p $(B)

$(B)/gas2d: src/cpu_main.c src/openmp.c $(COMMON) | $(B)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

$(B)/gas2d_mpi: src/mpi_main.c $(COMMON) | $(B)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# The CUDA kernels compiled for the CPU: checks their indexing and tiling without a GPU.
$(B)/gas2d_emu: src/emu_main.c src/common.c src/scenario.c include/kernels.h include/cuda_emu.h | $(B)
	$(CC) $(CFLAGS) src/emu_main.c src/common.c src/scenario.c -o $@ $(LDLIBS)

$(B)/libgas2d_cpu.so: src/lib_cpu.c src/openmp.c src/common.c | $(B)
	$(CC) $(CFLAGS) $(OMP) -fPIC -shared $^ -o $@ $(LDLIBS)

cuda: $(B)/gas2d_cuda $(B)/libgas2d_cuda.so

# The plain-C helpers are compiled by the C compiler and linked in: nvcc's -x option applies to
# every input file, so mixing .c and .cu in one nvcc call would compile the C files as CUDA C++.
$(B)/%.pic.o: src/%.c | $(B)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(B)/gas2d_cuda: src/cuda_main.cu src/cuda_backend.cu $(B)/common.pic.o $(B)/scenario.pic.o include/kernels.h include/cuda_backend.h
	$(NVCC) $(NVFLAGS) src/cuda_main.cu src/cuda_backend.cu $(B)/common.pic.o $(B)/scenario.pic.o -o $@ -lm

$(B)/libgas2d_cuda.so: src/lib_cuda.cu src/cuda_backend.cu $(B)/common.pic.o include/kernels.h include/cuda_backend.h
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared src/lib_cuda.cu src/cuda_backend.cu $(B)/common.pic.o -o $@ -lm

clean:
	rm -rf $(B)

.PHONY: all cuda clean
