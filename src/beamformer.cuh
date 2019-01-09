/* beamformer.cuh

	Contains all functions and defines related to CUDA

*/

#include <cublas_v2.h>
#include <cuda_runtime.h>


/* Helper Macro for gpu error checking */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* Function for gpu error checking */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

/* Error checking for cuBLAS */
void gpuBLASchk(int errval){
	if (errval != CUBLAS_STATUS_SUCCESS){
		std::cerr << "Failed BLAS call, error code " << errval << std::endl;
	}
}

/* Variables and Functions for timing analysis */
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

	#pragma unroll
	for (int i = input_idx; i < input_idx + n_avg*n_beams; i += n_beams){
		shmem[beam_idx] += input[i].x*input[i].x + input[i].y*input[i].y;
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



void CUDA_select_GPU(char * prefered_dev_name){
	/***********************************
	 GPU Card selection			   
	 ***********************************/
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
	    cudaDeviceProp deviceProperties;
	    cudaGetDeviceProperties(&deviceProperties, deviceIndex);

	    if (!strcmp(prefered_dev_name, deviceProperties.name)){
		    std::cout <<  "Selected: " << deviceProperties.name << std::endl;
		    gpuErrchk(cudaSetDevice(deviceIndex));
		}
	}
}













