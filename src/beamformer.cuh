#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// #include <pthread.h>
// #include <thread>
// #include <mutex>
// #include <barrier>

#include <cmath>
// #include <thrust::constant_iterator.h>
#include <fstream>
// #include "../lib/bitmap_image.hpp" // colorscheme
#include <cstdint>



/* dada includes */
#ifndef DEBUG
	#include <algorithm>
	#include <stdlib.h>
	#include <math.h>
	#include <string.h>
	#include <unistd.h>
	#include <netdb.h>
	#include <sys/socket.h>
	#include <sys/types.h>
	#include <netinet/in.h>
	#include <time.h>

	#include "dada_client.h"
	#include "dada_def.h"
	#include "dada_hdu.h"
	#include "multilog.h"
	#include "ipcio.h"
	#include "ipcbuf.h"
	#include "dada_affinity.h"
	#include "ascii_header.h"
#endif



// DSA CONSTANTS
#define N_BEAMS 256
#define N_ANTENNAS 64
#define N_FREQUENCIES 256

#define N_POL 2				//Number of polarizations
#define N_CX 2				//Number of real numbers in a complex number, namely 2

#if DEBUG
	#define N_AVERAGING 1
#else
	#define N_AVERAGING 16
#endif

// DATA constants

/* How many matrix multiplications could be executed based on the amount of data on the GPU*/
#define N_GEMMS_PER_GPU 256

/* How many output tensors are generated by each GEMM. This parameter helps improve throughput*/
#define N_OUTPUTS_PER_GEMM 8

/* Based on the size of a dada blocks: How many matrix-matrix multiplacations are need */
#define N_GEMMS_PER_BLOCK 64

/* For each output, we need to average over 16 iterations and 2 polarizations*/
#define N_INPUTS_PER_OUTPUT (N_POL*N_AVERAGING)

/* This is the number of columns processed in each matrix multiplication (includes 2 pol)*/
#define N_TIMESTEPS_PER_GEMM (N_OUTPUTS_PER_GEMM*N_INPUTS_PER_OUTPUT)

/* Calculates the number of blocks on the GPU given the number of GEMMMs possible on the GPU
   and the number of gemms contained in each block*/
#define N_BLOCKS_on_GPU (N_GEMMS_PER_GPU/N_GEMMS_PER_BLOCK)

/* Number of complex numbers of input data are needed for each GEMM */
#define N_CX_IN_PER_GEMM  (N_ANTENNAS*N_FREQUENCIES*N_TIMESTEPS_PER_GEMM)

/* Number of Complex numbers of output data are produced in each GEMM */
#define N_CX_OUT_PER_GEMM (N_BEAMS*N_FREQUENCIES*N_TIMESTEPS_PER_GEMM)

/* The detection step averages over N_INPUTS_PER_OUTPUT (16) numbers */
#define N_F_PER_DETECT (N_CX_OUT_PER_GEMM/N_INPUTS_PER_OUTPUT)

/* Number of Bytes of input data are needed for each GEMM, the real part and imaginary parts
   of each complex number use 1 Byte after expansion */
#define N_BYTES_POST_EXPANSION_PER_GEMM  (N_CX_IN_PER_GEMM*N_CX)

/* Number of Bytes before expansion. Each complex number uses half a Byte */
#define N_BYTES_PRE_EXPANSION_PER_GEMM  N_CX_IN_PER_GEMM*N_CX/2

/* Number of Bytes (before expansion) for input array */
#define N_BYTES_PER_BLOCK N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK


#define INPUT_DATA_SIZE N_BYTES_PRE_EXPANSION_PER_GEMM*N_DIRS

// Data Indexing, Offsets
#define N_GPUS 8
#define TOT_CHANNELS 2048
#define START_F 1.28
#define END_F 1.53
#define ZERO_PT 0
#define BW_PER_CHANNEL ((END_F - START_F)/TOT_CHANNELS)

// Numerical Constants
#define C_SPEED 299792458.0
#define PI 3.14159265358979


// Type Constants
#define N_BITS 8
#define MAX_VAL 127

#define SIG_BITS 4
#define SIG_MAX_VAL 7

// Solving Constants
#define N_STREAMS 8
#define N_DIRS  1024



/***********************************
 *				Types			   *
 ***********************************/

typedef char2 CxInt8_t;
typedef char char4_t[4]; //four chars = 32-bit so global memory bandwidth usage is optimal
typedef char char8_t[8]; //eight chars = 64-bit so global memory bandwidth usage is optimal
typedef CxInt8_t cuChar4_t[4];


/***********************************
 *		Defined Functions		   *
 ***********************************/

#define DEG2RAD(x) ((x)*PI/180.0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}


void gpuBLASchk(int errval){
	if (errval != CUBLAS_STATUS_SUCCESS){
		std::cerr << "Failed BLAS call, error code " << errval << std::endl;
	}
}

cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
	gpuErrchk(cudaEventCreate(&start));       \
	gpuErrchk(cudaEventCreate(&stop));        \
	gpuErrchk(cudaEventRecord(start));        \
}

#define STOP_RECORD_TIMER(name) {                           \
	gpuErrchk(cudaEventRecord(stop));                     \
	gpuErrchk(cudaEventSynchronize(stop));                \
	gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
	gpuErrchk(cudaEventDestroy(start));                   \
	gpuErrchk(cudaEventDestroy(stop));                    \
}


/***********************************
 *			DADA				   *
 ***********************************/
#ifndef DEBUG
/* Usage as defined by dada example code */
void usage()
{
  fprintf (stdout,
	   "dsaX_imager [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -k key [default dada]\n"
	   " -h        print usage\n");
}

/*cleanup as defined by dada example code */
void dsaX_dbgpu_cleanup (dada_hdu_t * in,  multilog_t * log) {
	if (dada_hdu_unlock_read (in) < 0){
		multilog(log, LOG_ERR, "could not unlock read on hdu_in\n");
	}
	dada_hdu_destroy (in);
}

#endif
/***********************************
 *			Kernels				   *
 ***********************************/


__global__
void expand_input(char const * __restrict__ input, char *output, int input_size){
	/*
	This code takes in an array of 4-bit integers and returns an array of 8-bit integers.
	To maximize global memory bandwidth and symplicity, two special char types are 
	defined: char4_t and char8_t. The size of these types are 32-bit and 64-bit respectively. These
	enable coalesced memory accesses, but then require the gpu to handle the 4-bit to 8-bit
	conversion simultaneously for 8 numbers (4 real/imaginary pairs). 
	*/

	__shared__ float shmem_in[32];
	__shared__ double shmem_out[32];

	char4_t *char_shmem_in;
	cuChar4_t *char_shmem_out;

	char_shmem_in = reinterpret_cast<char4_t *>(shmem_in);
	char_shmem_out = reinterpret_cast<cuChar4_t *>(shmem_out);

	int local_idx = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (global_idx < input_size/sizeof(float)){
		shmem_in[local_idx] = ((float *) input)[global_idx]; // read eight pieces of 4-bit memory into shared memory

		//#pragma unroll
		for (int i = 0; i < 4; i++){
			char temp = char_shmem_in[local_idx][i];

			// break the char into two 4-bit chunks, then convert to 8-bit
			char high = (temp >> 4); // roll the 		most  significant 4 bits over the least significant 4 bits
			char low = (temp << 4);  // roll the 		least significant 4 bits over the most  significant 4 bits
			low = (low >> 4); 	 	 // roll the *new* 	most  significant 4 bits over the least significant 4 bits

			// store the two 8-bit numbers to the output shared memory array
			char_shmem_out[local_idx][i].x = high; 
			char_shmem_out[local_idx][i].y = low;
		}

		((double *) output)[global_idx] = shmem_out[local_idx];	// write eight pieces of 8-bit memory out to global memory

		global_idx += gridDim.x * blockDim.x;
	}
}


dim3 detect_dimGrid(N_OUTPUTS_PER_GEMM, N_FREQUENCIES, 1);
dim3 detect_dimBlock(N_BEAMS, 1,1);

__global__
void detect_sum(cuComplex const * __restrict__ input, int n_avg,  float * __restrict__ output){
	/*
	Sum over the power in each N_INPUTS_PER_OUTPUT (32) time samples for each beam.
	number of threads = N_BEAMS (blockDim.x)
	number of blocks = N_OUTPUTS_PER_GEMM (blockIdx.x), N_FREQUENCIES (blockIdx.y)

	input indicies (slow to fast): freq, time batching, time, pol, beam, r/i
									256,            ~2,   16,   2,  256,   2

	out indicies (slow to fast):  time batching, freq, beam, r/i
								             ~2,  256,  256,   1

	*/

	// __shared__ cuComplex in[N_INPUTS_PER_OUTPUT*32];

	const int n_beams = blockDim.x;	 	// should be = N_BEAMS
	const int n_freq = gridDim.y;   		// should be = N_FREQUENCIES
	const int n_batch = gridDim.x;		// should be = N_OUTPUTS_PER_GEMM = N_AVERAGING * N_POL

	const int batching_idx = blockIdx.x;	// index along batching dimension
	const int freq_idx = blockIdx.y;		// index along frequency dimension
	const int beam_idx = threadIdx.x;		// index along beam dimension

	__shared__ float shmem[N_BEAMS];// cannot use n_beams here since shmem needs a constant value
	shmem[beam_idx] = 0;			// initialize shared memory to zero

	const int input_idx  	 = freq_idx * n_batch * n_avg * n_beams 	// deal with frequency component first
								+ batching_idx * n_avg * n_beams  		// deal with the indexing for each output next
																		// Each thread starts on the first time-pol index 
																		// (b/c we iterate over all time-pol indicies in for loop below)
								+ beam_idx;							   	// start each thread on the first element of each beam

	const int output_idx = batching_idx * n_freq * n_beams + freq_idx * n_beams + beam_idx;

	// cuComplex in;

	#pragma unroll
	for (int i = input_idx; i < input_idx + n_avg*n_beams; i += n_beams){
		// in = input[input_idx]; // coallesced memory access
		shmem[beam_idx] += input[i].x*input[i].x + input[i].y*input[i].y;
		 // go to the next time step
	}

	output[output_idx] = shmem[beam_idx]; // slowest to fastest indicies: freq, beam
}

__global__
void print_data(float* data){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("data[%d] = %f\n", idx, data[idx]);
}

__global__
void print_data_scalar(float* data){
	// int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("*data = %f\n", *data);
}

void print_all_defines(void){
	std::cout << "N_BEAMS:" << N_BEAMS << "\n";
	std::cout << "N_ANTENNAS:" << N_ANTENNAS << "\n";
	std::cout << "N_FREQUENCIES:" << N_FREQUENCIES << "\n";
	std::cout << "N_AVERAGING:" << N_AVERAGING << "\n";
	std::cout << "N_POL:" << N_POL << "\n";
	std::cout << "N_CX:" << N_CX << "\n";
	std::cout << "N_GEMMS_PER_GPU:" << N_GEMMS_PER_GPU << "\n";
	std::cout << "N_OUTPUTS_PER_GEMM:" << N_OUTPUTS_PER_GEMM << "\n";
	std::cout << "N_GEMMS_PER_BLOCK:" << N_GEMMS_PER_BLOCK << "\n";
	std::cout << "N_INPUTS_PER_OUTPUT:" << N_INPUTS_PER_OUTPUT << "\n";
	std::cout << "N_TIMESTEPS_PER_GEMM:" << N_TIMESTEPS_PER_GEMM << "\n";
	std::cout << "N_BLOCKS_on_GPU:" << N_BLOCKS_on_GPU << "\n";
	std::cout << "N_CX_IN_PER_GEMM:" << N_CX_IN_PER_GEMM << "\n";
	std::cout << "N_CX_OUT_PER_GEMM:" << N_CX_OUT_PER_GEMM << "\n";
	std::cout << "N_BYTES_POST_EXPANSION_PER_GEMM:" << N_BYTES_POST_EXPANSION_PER_GEMM << "\n";
	std::cout << "N_BYTES_PRE_EXPANSION_PER_GEMM:" << N_BYTES_PRE_EXPANSION_PER_GEMM << "\n";
	std::cout << "N_BYTES_PER_BLOCK:" << N_BYTES_PER_BLOCK << "\n";
	std::cout << "N_GPUS:" << N_GPUS << "\n";
	std::cout << "TOT_CHANNELS:" << TOT_CHANNELS << "\n";
	std::cout << "START_F:" << START_F << "\n";
	std::cout << "END_F:" << END_F << "\n";
	std::cout << "ZERO_PT:" << ZERO_PT << "\n";
	std::cout << "BW_PER_CHANNEL:" << BW_PER_CHANNEL << "\n";
	std::cout << "C_SPEED:" << C_SPEED << "\n";
	std::cout << "PI:" << PI <<"\n";
	std::cout << "N_BITS:" << N_BITS << "\n";
	std::cout << "MAX_VAL:" << MAX_VAL << "\n";
	std::cout << "SIG_BITS:" << SIG_BITS << "\n";
	std::cout << "SIG_MAX_VAL:" << SIG_MAX_VAL << "\n";
	std::cout << "N_STREAMS:" << N_STREAMS << "\n";
	std::cout << "N_DIRS:" << N_DIRS << "\n";

	std::cout << std::endl;
}

















