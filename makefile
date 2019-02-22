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

verbose: CXXFLAGS += -DVERBOSE -lpsrdada

debug: beam
verbose: beam

.PHONY: junk
junk: 
	dada_junkdb -c 0 -z -k baab -r 4000 -t 10 lib/correlator_header_dsaX.txt

beam: $(SRCDIR)/beamformer.cu
	$(CC) -o $(BINDIR)/$@ $^ -I$(DADA_INCLUDE) -L$(DADA_LIB) $(CXXFLAGS) $(NVFLAGS)

.PHONY: clean
clean:
	rm bin/beam;
	dada_db -k baab -d;
	dada_db -d;
	echo "killing: "
	echo $(ps aux | grep -e bin/beam | awk '{print $2}')
	echo "\n"
	kill -9 $(ps aux | grep -e bin/beam | awk '{print $2}');

