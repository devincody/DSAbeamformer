#include <iostream>

#include <cublas_v2.h>
#include <cuComplex.h>
// #include <>

#define N_BEAMS 256
#define N_AVERAGING 16
#define N_TIMESTEPS_PER_CALL 64
#define N_POL 2
#define N 1000000

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
void detect_sum(cuComplex *input, cuComplex *output){
	__shared__ float shmem[N_BEAMS];

	int tid = blockIdx.x * blockDim.x*N_POL*N_AVERAGING + threadIdx.x;

	cuComplex in;

	#pragma unroll
	for (int i = 0; i < N_POL*N_AVERAGING; i++){
		in = input[tid];
		shmem[tid] += in.x*in.x + in.y*in.y;
		tid += blockDim.x;
	}

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








int main (){
	std::cout << "hello " << std::endl;



	char *in = new char[N];
	char *out = new char[2*N];

	for (int i = 0; i < N; i ++){
		in[i] = 0xD7;
	}

	char *d_in;
	char *d_out;

	gpuErrchk(cudaMalloc(&d_in, N*sizeof(char)));
	gpuErrchk(cudaMalloc(&d_out, 2*N*sizeof(char)));

	gpuErrchk(cudaMemcpy(d_in, in, N*sizeof(char), cudaMemcpyHostToDevice));

	float gpu_time_ms_solve;
	START_TIMER();
	expand_input<<<1000,32>>>(d_in, d_out, N);
	STOP_RECORD_TIMER(gpu_time_ms_solve);

	cudaMemcpy(out, d_out, 2*N*sizeof(char), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i ++){
		std::cout << (int) out[i] << std::endl;
	}

	std::cout << "time was: " << gpu_time_ms_solve << std::endl;




	return 0;
}



