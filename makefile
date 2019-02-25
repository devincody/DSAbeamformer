CC=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++11
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets

DADA_INCLUDE=/home/dcody/psrdada/src
DADA_LIB=/home/dcody/psrdada/src/.libs

BINDIR = bin
SRCDIR = src

all: CXXFLAGS += -lpsrdada -O3 -use_fast_math -arch=sm_61 -restrict
all: beam

fast_debug: CXXFLAGS += -DDEBUG -DVERBOSE -g -lineinfo -Xcompiler -fopenmp -O3
fast_debug: beam

debug: CXXFLAGS += -DDEBUG -DVERBOSE -g -lineinfo -O0
debug: beam

verbose: CXXFLAGS += -DVERBOSE -lpsrdada
verbose: beam

beam: $(SRCDIR)/beamformer.cu
	$(CC) -o $(BINDIR)/$@ $^ -I$(DADA_INCLUDE) -L$(DADA_LIB) $(CXXFLAGS) $(NVFLAGS)

.PHONY: junk
junk: 
	dada_junkdb -c 0 -z -k baab -r 4050 -t 25 lib/correlator_header_dsaX.txt

.PHONY: clean
clean:
	rm bin/beam;
	dada_db -k baab -d;
	dada_db -d;
	echo "killing: "
	echo $(ps aux | grep -e bin/beam | awk '{print $2}')
	echo "\n"
	kill -9 $(ps aux | grep -e bin/beam | awk '{print $2}');

