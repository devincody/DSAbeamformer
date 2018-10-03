#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cublas_v2.h>
#include <cmath>
// #include <thrust::constant_iterator.h>
#include <fstream>
#include "../lib/bitmap_image.hpp" // colorscheme

// DSA CONSTANTS
#define N_BEAMS 256
#define N_ANTENNAS 64
#define N_FREQUENCIES 256
#define N_AVERAGING 16
#define N_POL 2
#define N_TIMESTEPS_PER_CALL 1*N_AVERAGING*N_POL

#define N_CX 2
#define N_BLOCKS_on_GPU 4
#define BYTES_PER_GEMM  N_ANTENNAS*N_FREQUENCIES*N_TIMESTEPS_PER_CALL

// Data Indexing, Offsets
#define N_GPUS 8
#define TOT_CHANNELS 2048
#define START_F 1.28
#define END_F 1.53
#define ZERO_PT 0

// Numerical Constants
#define C_SPEED 299792458.0
#define PI 3.14159265358979


// Type Constants
#define N_BITS 8
#define MAX_VAL 127

#define SIG_BITS 4
#define SIG_MAX_VAL 7


// nvcc beamformer.cu -o bin/beam -lcublas


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



typedef char2 CxInt8_t;
typedef char char4_t[4]; //32-bit so global memory bandwidth usage is optimal
typedef char char8_t[8]; //64-bit so global memory bandwidth usage is optimal
typedef CxInt8_t cuChar4_t[4];



__global__
void detect_sum(cuComplex *input, float *output){
	/*
	Sum over N_TIMESTEPS_PER_CALL
	number of threads = N_BEAMS = blockDim.x
	number of blocks = N_FREQUENCIES
	*/
	__shared__ float shmem[N_BEAMS];

	int input_idx  = blockIdx.x * N_BEAMS * N_TIMESTEPS_PER_CALL + threadIdx.x;
	int local_idx  = threadIdx.x; // which beam
	int output_idx = blockIdx.x * N_BEAMS + threadIdx.x;

	cuComplex in;

	// #pragma unroll
	for (int i = 0; i < N_TIMESTEPS_PER_CALL; i++){
		in = input[input_idx];
		shmem[local_idx] += in.x*in.x;// + in.y*in.y;
		input_idx += N_BEAMS; // go to the next time step
	}

	output[output_idx] = shmem[local_idx]; // slowest to fastest indicies: freq, beam
}



__global__
void expand_input(char *input, char *output, int input_size){
	/*
	This code takes in an array of 4-bit integers and returns an array of 8-bit integers.
	To maximize global memory bandwidth and symplicity, two special char types are 
	defined: char4_t and char8_t. The size of these types are 32-bit and 64-bit respectively. These
	enable coalesced memory accesses, but then require the gpu to handle the 4 to 8-bit
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
