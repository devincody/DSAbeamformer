#include <iostream>
#include <cublas_v2.h>
#include <cmath>

// DSA CONSTANTS
#define N_BEAMS 256
#define N_ANTENNAS 64
#define N_FREQUENCIES 256
#define N_AVERAGING 16
#define N_TIMESTEPS_PER_CALL 64
#define N_BLOCKS_on_GPU 4

// Data Indexing, Offsets
#define N_GPUS 8
#define TOT_CHANNELS 2048
#define START_F 1.28
#define END_F 1.53
#define ZERO_PT 0

// Numerical Constants
#define C_SPEED 299792458.0
#define PI 3.14159265358979
#define N_POL 2
#define N_CX 2

// Type Constants
#define N_BITS 8
#define MAX_VAL 127

#define DEG2RAD(x) ((x)*PI/180.0)

typedef char2 CxInt8_t;



int main(){
	std::cout << "hello" << std::endl;

	/* Variables */
	CxInt8_t *d_A; 				// Weight matrix (N_BEAMS X N_ANTENNAS, for N_FREQUENCIES)
	CxInt8_t *d_B; 				// Data Matrix (N_ANTENNAS X N_TIMESTEPS_PER_CALL, for N_FREQUENCIES)
	CxInt8_t *d_data;			// Raw input data (Before data massaging)
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
	cuComplex *C = new cuComplex[C_cols*C_rows*N_FREQUENCIES];

	float* pos = new float[N_ANTENNAS];		// Locations of antennas
	float* dir = new float[N_BEAMS];		// Direction of beams
	int gpu = 0;


	/* Populate Helper Matricies */
	for (int i = 0; i < N_ANTENNAS; i++){
		pos[i] = i*500.0/N_ANTENNAS - 250.0;
	}

	for (int i = 0; i < N_BEAMS; i++){
		dir[i] = i*DEG2RAD(7.0)/N_BEAMS - DEG2RAD(3.5);
	}

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
 	
	int simulated_direction = 100;

	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/N_GPUS + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_TIMESTEPS_PER_CALL; j++){
			for (int k = 0; k < N_ANTENNAS; k++){
				B[i*B_stride + j*N_ANTENNAS + k].x = round(MAX_VAL*cos(2*PI*pos[k]*sin(dir[simulated_direction])/wavelength));
				B[i*B_stride + j*N_ANTENNAS + k].y = round(MAX_VAL*sin(2*PI*pos[k]*sin(dir[simulated_direction])/wavelength));
			}
		}
	}


	/* Allocate and Move Memory to Device */
	cudaMalloc(&d_A, 	A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t));
	cudaMalloc(&d_B, 	B_rows*B_cols*N_FREQUENCIES*sizeof(CxInt8_t));
	cudaMalloc(&d_C, 	C_rows*C_cols*N_FREQUENCIES*sizeof(cuComplex));
	cudaMalloc(&d_data, N_ANTENNAS*N_FREQUENCIES*N_TIMESTEPS_PER_CALL*N_BLOCKS_on_GPU*sizeof(CxInt8_t));

	cudaMemcpy(d_A, A, A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, B_cols*B_rows*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice);



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
	delete[] B;
	delete[] C;
	delete[] pos;
	delete[] dir;

	cublasDestroy(handle);
	return 0;
}