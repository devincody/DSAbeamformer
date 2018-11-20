#include <cstdlib>
#include <iostream>
#include <curand.h>
#include <cublas_v2.h>



//https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
//nvcc mm.cu -lcublas -lcurand -o mm1


void GPU_fill_rand(float *A, int nr_rows, int nr_cols){
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	curandGenerateUniform(prng, A, nr_rows*nr_cols);
}


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


int main(int argc, char const *argv[]){

	// std::cout << "matrix mult" << std::endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	float *d_A, *d_B, *d_C;

	long na, nb, nmax;
	if (argc == 2){
		nmax = (long) atoi(argv[1]);
		na = 512;
		nb = 512;
	} else if (argc == 4){
		na = (long) atoi(argv[1]);
		nb = (long) atoi(argv[2]);
		nmax = (long) atoi(argv[3]);
	}
	std::cout << na << "," << nb << "," << nmax << ",";

	cudaMalloc(&d_A, na*nmax * sizeof(float));
	cudaMalloc(&d_B, nmax*nb * sizeof(float));
	cudaMalloc(&d_C, na*nb * sizeof(float));

	GPU_fill_rand(d_A, na, nmax);
	GPU_fill_rand(d_B, nmax, nb);
	GPU_fill_rand(d_C, na, nb);

	float alf = 1.0;
	float bet = 0.0;

	float ops;
	int k_avg = 5;

	for (int n = 1; n < nmax; n++){
		float timer = 0;
		float tot = 0;
		for (int k = 0; k < k_avg; k++){
			
			START_TIMER()
			gpuBLASchk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, na, nb, n, &alf, d_A, na, d_B, n, &bet, d_C, na));
			STOP_RECORD_TIMER(timer);
			tot += timer;
		}

		ops = (float) na*nb;
		ops *= 2*n-1;

		std::cout << ops/tot*1000.0*(k_avg);
		if( n != nmax -1 ){
			std::cout << ",";
		}
		//<< "FLOP = " << n*n*(2*n-1) << ", Time = " << timer << " ms, FLOPS = "
	}

	std::cout << std::endl;

	return 0;

}


