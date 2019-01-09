CC=/usr/local/cuda/bin/nvcc
CXXFLAGS= -O3 -std=c++11
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets -use_fast_math -arch=sm_61 -restrict

DADA_INCLUDE= /home/dcody/psrdada/src
DADA_LIB=/home/dcody/psrdada/src/.libs

BINDIR = bin
SRCDIR = src

all: CXXFLAGS += -lpsrdada
all: beam

debug: CXXFLAGS += -DDEBUG=1 -g -lineinfo -DVERBOSE

verbose: CXXFLAGS += -DVERBOSE

debug: beam
verbose: beam

beam: $(SRCDIR)/beamformer.cu
	$(CC) -o $(BINDIR)/$@ $^ -I$(DADA_INCLUDE) -L$(DADA_LIB) $(CXXFLAGS) $(NVFLAGS)

clean:
	rm beam
