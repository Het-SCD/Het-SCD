.PHONY: all clean

include config.mk

MGPU_INCLUDE=-I$(MGPU_PATH)/include
METIS_LIB=$(METIS_PATH)/libmetis/libmetis.a
METIS_INCLUDE=-I$(METIS_PATH)/include

CC=nvcc
CFLAGS=-Xcompiler "-Wall -Wconversion -Wsign-conversion -Wextra -Wshadow -fopenmp" $(NVCC_ARCH) -Xptxas="-v"
LDFLAGS=-lm $(METIS_INCLUDE) $(MGPU_INCLUDE)

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS +=-O2 -pg -g -G
else
	CFLAGS +=-O3 -Xcompiler "-march=native -fomit-frame-pointer"
endif


BINS=main \
	 main-gpu \
	 main-dist \
	 convert \
	 partition

all: $(BINS) Makefile config.mk

main convert: graph.o scd.o wcc.o common.o
main-gpu: graph.o scd.o wcc.o common.o cuda.o mgpucontext.o mgpuutil.o
main-dist: graph.o scd.o wcc.o common.o dist.o mgpucontext.o mgpuutil.o
partition: graph.o

main main-gpu main-dist convert: %: %.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

partition: %: %.cpp
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(METIS_LIB)

cuda.o: kernels.cuh cuda-common.cuh
dist.o: kernels.cuh cuda-common.cuh

mgpucontext.o: $(MGPU_PATH)/src/mgpucontext.cu
mgpuutil.o: $(MGPU_PATH)/src/mgpuutil.cpp

mgpucontext.o mgpuutil.o: %.o:
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

%.o: %.cpp %.hpp
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

%.o: %.cu %.hpp
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf $(BINS) *.o

