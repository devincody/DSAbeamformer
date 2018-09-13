#include <iostream>
#include <cublas_v2.h>
#include <cmath>

// DSA CONSTANTS
#define N_BEAMS 256
#define N_ANTENNAS 64
#define N_FREQUENCIES 256
#define N_AVERAGING 16
#define N_TIMESTEPS_PER_CALL 64
#define N_POL 2
#define N_CX 2

// Data Indexing, Offsets
#define N_GPUS 8
#define TOT_CHANNELS 2048
#define START_F 1.28
#define END_F 1.53
#define ZERO_PT 0

// Universal Constants
#define C_SPEED 299792458.0
#define PI 3.14159265358979

// Bit Constants
#define N_BITS 8
#define MAX_VAL 127

#define DEG2RAD(x) ((x)*PI/180.0)

typedef char2 CxInt8_t;



int main(){
	int gpu = 0;

	std::cout << "hello" << std::endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	CxInt8_t *A = new CxInt8_t[N_BEAMS*N_ANTENNAS*N_FREQUENCIES];
	float* pos = new float[N_ANTENNAS];
	float* dir = new float[N_BEAMS];

	for (int i = 0; i < N_ANTENNAS; i++){
		pos[i] = i*500.0/N_ANTENNAS - 250.0;
	}

	for (int i = 0; i < N_BEAMS; i++){
		dir[i] = i*DEG2RAD(7.0)/N_BEAMS - DEG2RAD(3.5);
	}

	float bw_per_channel = (END_F - START_F)/TOT_CHANNELS;

	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/N_GPUS + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_ANTENNAS; j++){
			for (int k = 0; k < N_BEAMS; k++){
				A[i*N_ANTENNAS*N_BEAMS + j*N_BEAMS + k].x = round(MAX_VAL*cos(-2*PI*pos[j]*sin(dir[k])/wavelength));
				A[i*N_ANTENNAS*N_BEAMS + j*N_BEAMS + k].y = round(MAX_VAL*sin(-2*PI*pos[j]*sin(dir[k])/wavelength));
				
			}
		}
		std::cout << "A[] = " << (int) A[i*N_ANTENNAS*N_BEAMS].x << "+"<< (int) A[i*N_ANTENNAS*N_BEAMS].y << "j" << std::endl;

	}

	CxInt8_t *d_A;
	cudaMalloc(&d_A, N_BEAMS*N_ANTENNAS*N_FREQUENCIES*sizeof(CxInt8_t));
	cudaMemcpy(d_A, A, N_BEAMS*N_ANTENNAS*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice);


	// CUDA_R_16F te = 12.4;
	// std::cout << "rt " << te << std::endl;

	cudaFree(d_A);
	delete[] A;
	delete[] pos;
	delete[] dir;
	cublasDestroy(handle);
	return 0;
}