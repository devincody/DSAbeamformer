CC=nvcc
CXXFLAGS= -std=c++11 -O3 -lpsrdada
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets

BINDIR = bin
SRCDIR = src


all: beam

debug: CXXFLAGS += -D DEBUG=1 -g -lineinfo
debug: beam

beam: $(SRCDIR)/beamformer.cu
	$(CC) -o $(BINDIR)/$@ $^ $(CXXFLAGS) $(NVFLAGS)

clean:
	rm beam
