# sim3d/Makefile - builds the 3D simulator (Lattice Boltzmann air + gas transport).
#
#   make        CPU builds: libsim_cpu.so, bench_cpu, mpi_cpu   (g++, OpenMP; compat.h maps the
#               GPU kernels onto CPU loops, so the physics is identical)
#   make cuda   GPU builds: libsim_cuda.so, bench_cuda, mpi_cuda (needs nvcc; mpi_cuda needs MPI)
#
# These are the same commands the original Colab notebook ran, collected in one place.

CXX      ?= g++
MPICXX   ?= mpicxx
NVCC     ?= nvcc
CUDA_ARCH ?= native
ARCH     ?= -march=native
CXXFLAGS  = -O3 $(ARCH) -fopenmp
B = build

all: $(B)/libsim_cpu.so $(B)/bench_cpu $(B)/mpi_cpu

$(B):
	mkdir -p $(B)

$(B)/libsim_cpu.so: core/sim.cu core/kernels.cuh core/compat.h | $(B)
	$(CXX) $(CXXFLAGS) -fPIC -shared -x c++ core/sim.cu -o $@

$(B)/bench_cpu: core/bench.cu core/testcase.h core/kernels.cuh core/compat.h | $(B)
	$(CXX) $(CXXFLAGS) -x c++ core/bench.cu -o $@

$(B)/mpi_cpu: core/mpi_sim.cu core/testcase.h core/kernels.cuh core/compat.h | $(B)
	$(MPICXX) $(CXXFLAGS) -x c++ core/mpi_sim.cu -o $@

cuda: | $(B)
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Xcompiler -fPIC -shared core/sim.cu -o $(B)/libsim_cuda.so
	$(NVCC) -O3 -arch=$(CUDA_ARCH) core/bench.cu -o $(B)/bench_cuda
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -ccbin $(MPICXX) core/mpi_sim.cu -o $(B)/mpi_cuda

clean:
	rm -rf $(B)

.PHONY: all cuda clean
