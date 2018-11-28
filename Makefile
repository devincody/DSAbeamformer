CC=nvcc
CXXFLAGS= -std=c++11 -O3 -lpsrdada
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets -use_fast_math -arch=sm_61 -restrict

BINDIR = bin
SRCDIR = src


all: beam

debug: CXXFLAGS += -D DEBUG=1 -g -lineinfo
debug: beam

verbose: CXXFLAGS += -DVERBOSE
verbose: beam

beam: $(SRCDIR)/beamformer.cu
	$(CC) -o $(BINDIR)/$@ $^ $(CXXFLAGS) $(NVFLAGS)

clean:
	rm beam
