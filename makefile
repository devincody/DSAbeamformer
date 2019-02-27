CC=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++11
NVFLAGS=-lcublas -Wno-deprecated-gpu-targets

DADA_INCLUDE=/home/dcody/psrdada/src
DADA_LIB=/home/dcody/psrdada/src/.libs

BINDIR = bin
SRCDIR = src

.PHONY: all fast_debug debug verbose junk dada_buffers clean

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

junk: 
	dada_junkdb -c 0 -z -k baab -r 4050 -t 25 config/correlator_header_dsaX.txt

dada_buffers:
	-dada_db -d -k baab
	dada_db -k baab -n 25 -b 134217728 -l -p

clean:
	-rm bin/beam
	-dada_db -k baab -d
	echo $$(ps aux | grep -e bin/beam | awk '{print $$2}')
	-kill -9 $$(ps aux | grep -e bin/beam | awk '{print $$2}')

