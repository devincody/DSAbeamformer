#include <iostream>
#include <cublas_v2.h>
#include <cmath>

// DSA CONSTANTS
#define N_BEAMS 256
#define N_ANTENNAS 64
#define N_FREQUENCIES 256
#define N_AVERAGING 16
#define N_TIMESTEPS_PER_CALL 32
#define N_POL 2
#define N_CX 2
#define N_BLOCKS_on_GPU 4
#define BYTES_PER_BLOCK  N_ANTENNAS*N_FREQUENCIES*N_TIMESTEPS_PER_CALL*N_AVERAGING*N_POL

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




int main(){
	std::cout << "hello" << std::endl;

	/* Variables */
	CxInt8_t *d_A; 				// Weight matrix (N_BEAMS X N_ANTENNAS, for N_FREQUENCIES)
	CxInt8_t *d_B; 				// Data Matrix (N_ANTENNAS X N_TIMESTEPS_PER_CALL, for N_FREQUENCIES)
	char *d_data;			// Raw input data (Before data massaging)
	cuComplex *d_C;				// Beamformed output (N_BEAMS X N_TIMESTEPS_PER_CALL, for N_FREQUENCIES)

	int A_rows	 = N_BEAMS;
	int A_cols 	 = N_ANTENNAS;
	int A_stride = A_rows*A_cols;
	int B_cols	 = N_TIMESTEPS_PER_CALL;
	int B_rows	 = A_cols;
	int B_stride = B_rows*B_cols;
	int C_rows	 = A_rows;
	int C_cols	 = B_cols;
	int C_stride = C_rows*C_cols;
	float bw_per_channel = (END_F - START_F)/TOT_CHANNELS;

	CxInt8_t *A = new CxInt8_t[A_cols*A_rows*N_FREQUENCIES];
	CxInt8_t *B = new CxInt8_t[B_cols*B_rows*N_FREQUENCIES];
	char *data = new char[BYTES_PER_BLOCK]; //should be the size of one "dada block", data is 4-bit so real/imag is packed into one 8-bit char
	cuComplex *C = new cuComplex[C_cols*C_rows*N_FREQUENCIES];

	float* pos = new float[N_ANTENNAS];		// Locations of antennas
	float* dir = new float[N_BEAMS];		// Direction of beams
	int gpu = 0;							// Unique identifier for each GPU


	/* Populate location/direction Matricies */
	for (int i = 0; i < N_ANTENNAS; i++){
		pos[i] = i*500.0/N_ANTENNAS - 250.0;
	}

	for (int i = 0; i < N_BEAMS; i++){
		dir[i] = i*DEG2RAD(7.0)/N_BEAMS - DEG2RAD(3.5);
	}


	// Fourier Coefficient Matrix
	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/N_GPUS + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_ANTENNAS; j++){
			for (int k = 0; k < N_BEAMS; k++){
				A[i*A_stride + j*N_BEAMS + k].x = round(MAX_VAL*cos(-2*PI*pos[j]*sin(dir[k])/wavelength));
				A[i*A_stride + j*N_BEAMS + k].y = round(MAX_VAL*sin(-2*PI*pos[j]*sin(dir[k])/wavelength));
			}
		}
		// std::cout << "A[] = " << (int) A[i*N_ANTENNAS*N_BEAMS].x << "+"<< (int) A[i*N_ANTENNAS*N_BEAMS].y << "j" << std::endl;
	}

	// Signal Matrix
	int test_frequency = 10;
	float test_direction = DEG2RAD(-3.4);

	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/N_GPUS + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);

		for (int j = 0; j < N_TIMESTEPS_PER_CALL; j++){
			for (int k = 0; k < N_ANTENNAS; k++){
				if (i == test_frequency){
					B[i*N_TIMESTEPS_PER_CALL*N_ANTENNAS + j*N_ANTENNAS + k].x = round(MAX_VAL*cos(2*PI*pos[k]*sin(test_direction)/wavelength));
					B[i*N_TIMESTEPS_PER_CALL*N_ANTENNAS + j*N_ANTENNAS + k].y = round(MAX_VAL*sin(2*PI*pos[k]*sin(test_direction)/wavelength));
				}
			}
		}

	}
 	
	int simulated_direction = 100;
	int current_block = 0;
	int tot_avging = N_POL*N_AVERAGING;

	char high, low;

	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/N_GPUS + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_TIMESTEPS_PER_CALL; j++){
			for (int k = 0; k < N_ANTENNAS; k++){
				for (int l = 0; l < tot_avging; l ++){

					high = ((char) round(MAX_VAL*cos(2*PI*pos[k]*sin(dir[simulated_direction])/wavelength)));
					low  = ((char) round(MAX_VAL*sin(2*PI*pos[k]*sin(dir[simulated_direction])/wavelength)));

					data[i*B_stride*tot_avging + j*N_ANTENNAS*tot_avging + k*tot_avging + l] = (high << 4) | (0x0F & low);
					// B[i*B_stride + j*N_ANTENNAS + k].y = ;
				}
			}
		}
	}


	/* Allocate and Move Memory to Device */
	cudaMalloc(&d_A, 	A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t));
	cudaMalloc(&d_B, 	B_rows*B_cols*N_FREQUENCIES*sizeof(CxInt8_t));
	cudaMalloc(&d_C, 	C_rows*C_cols*N_FREQUENCIES*sizeof(cuComplex));
	// cudaMalloc(&d_data, BYTES_PER_BLOCK*N_BLOCKS_on_GPU);

	cudaMemcpy(d_A, A, A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, B_rows*B_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
	// cudaMemcpy(&(d_data[BYTES_PER_BLOCK*current_block]), data, BYTES_PER_BLOCK, cudaMemcpyHostToDevice);

	// reduce_input<<<40, 32>>>(d_data, d_B);

	cublasHandle_t handle;
	cublasCreate(&handle);

	// Multiplicative Constants
	cuComplex inv_max_value, zero;
	inv_max_value.x = 1.0/MAX_VAL;
	inv_max_value.y = 0;
	zero.x = 0;
	zero.y = 0;

	cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
								A_rows, B_cols, A_cols,
								&inv_max_value,
								d_A, CUDA_C_8I, A_rows, A_stride,
								d_B, CUDA_C_8I, B_rows, B_stride,
								&zero,
								d_C, CUDA_C_32F, C_rows, C_stride,
								N_FREQUENCIES, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);


	cudaMemcpy(C, d_C, C_rows*C_cols*N_FREQUENCIES*sizeof(cuComplex), cudaMemcpyDeviceToHost);

	float max = 0;
	float rms = 0.0;
	int max_i = 0;
	for (int i = 0; i < 256; i++){
		if (C[i].x>max){
			rms += C[i].x*C[i].x;
			max = C[i].x;
			max_i = i;
		}
		std::cout << "C[" << i <<"] = " << C[i].x << "+" << C[i].y << "j" << std::endl;
	}

	std::cout << "max(C) = " << max_i << ", " << max << std::endl;
	std::cout << "rms(c) = " << sqrt(rms/256.0) << std::endl;
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_data);

	delete[] A;
	delete[] data;
	delete[] C;
	delete[] pos;
	delete[] dir;

	cublasDestroy(handle);
	return 0;
}