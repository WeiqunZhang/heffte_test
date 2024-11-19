ifndef HEFFTE_HOME
  $(error Must define HEFFTE_HOME)
endif

ifndef CUDA_ARCH
  cuda_device_query_result := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
  ifeq ($(cuda_device_query_result),)
    $(error Please provide CUDA_ARCH)
  else
    CUDA_ARCH = $(subst .,,$(cuda_device_query_result))
  endif
endif

CXX = mpicxx
export OMPI_CXX := nvcc
export MPICH_CXX := nvcc
export MPICXX_CXX := nvcc

CUDA_ARCH_FLAGS = -arch=compute_$(CUDA_ARCH) -code=sm_$(CUDA_ARCH)
CXXFLAGS = -ccbin=g++ -Xcompiler='-g -O3 -std=c++17' --std=c++17 -m64 $(CUDA_ARCH_FLAGS) \
           --forward-unknown-to-host-compiler --expt-extended-lambda --ptxas-options=-O3 \
           -I$(HEFFTE_HOME)/include
LIBRARIES = -L$(HEFFTE_HOME)/lib -lheffte -lcuda -lcufft -lmpi

a.out: main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBRARIES)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -x cu -dc -c -o $@ $<

clean:
	${RM} *.o a.out

FORCE:

.PHONY: clean
