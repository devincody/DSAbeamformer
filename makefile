CC=nvcc
CXXFLAGS= -O3
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets -use_fast_math -arch=sm_61 -restrict

BINDIR = bin
SRCDIR = src

all: CXXFLAGS += -lpsrdada
all: beam

debug: CXXFLAGS += -DDEBUG=1 -g -lineinfo

verbose: CXXFLAGS += -DVERBOSE

debug: beam
verbose: beam

beam: $(SRCDIR)/beamformer.cu
	$(CC) -o $(BINDIR)/$@ $^ $(CXXFLAGS) $(NVFLAGS)

clean:
	rm beam
