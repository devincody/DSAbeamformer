#include <cublas_v2.h>
#include <iostream>


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



int main(){

	std::cout << "Hello " << std::endl;
	// Set the math mode to allow cuBLAS to use Tensor Cores:
	cublasHandle_t handle;
	cublasCreate(&handle);


	int *d_A, *d_B;
	int *d_C;
	float *dfC;

	int m = 2048;
	int n = 2048;
	int k = 2048;

	cudaMalloc(&d_A, m*k*sizeof(int));
	cudaMalloc(&d_B, n*k*sizeof(int));
	cudaMalloc(&d_C, m*n*sizeof(int));
	cudaMalloc(&dfC, m*n*sizeof(float));


	cudaMemset(d_A, 0x00, m*k*sizeof(int));
	cudaMemset(d_B, 0x00, n*k*sizeof(int));
	cudaMemset(d_C, 0, m*n*sizeof(int));
	cudaMemset(dfC, 0, m*n*sizeof(float));

	// Invoke the GEMM, ensuring k, lda, ldb, and ldc are all multiples of 8, 
	// and m is a multiple of 4:
	int alpha = 1;
	int beta = 0;

	float falpha = 1;
	float fbeta = 0;

	// half halpha = 1;
	// half hbeta = 0;	


	// cublasGemmAlgo_t alg[] = {CUBLAS_GEMM_ALGO0,
	// 						  CUBLAS_GEMM_ALGO1,
	// 						  CUBLAS_GEMM_ALGO2,
	// 						  CUBLAS_GEMM_ALGO3,
	// 						  CUBLAS_GEMM_ALGO4,
	// 						  CUBLAS_GEMM_ALGO5,
	// 						  CUBLAS_GEMM_ALGO6,
	// 						  CUBLAS_GEMM_ALGO7,
	// 						  CUBLAS_GEMM_ALGO8,
	// 						  CUBLAS_GEMM_ALGO9,
	// 						  CUBLAS_GEMM_ALGO10,
	// 						  CUBLAS_GEMM_ALGO11,
	// 						  CUBLAS_GEMM_ALGO12,
	// 						  CUBLAS_GEMM_ALGO13,
	// 						  CUBLAS_GEMM_ALGO14,
	// 						  CUBLAS_GEMM_ALGO15,
	// 						  CUBLAS_GEMM_ALGO16,
	// 						  CUBLAS_GEMM_ALGO17,
	// 						  CUBLAS_GEMM_ALGO18,
	// 						  CUBLAS_GEMM_ALGO19,
	// 						  CUBLAS_GEMM_ALGO20,
	// 						  CUBLAS_GEMM_ALGO21,
	// 						  CUBLAS_GEMM_ALGO22};



	cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
	for (int i = 0; i < 10; i++){	
		gpuBLASchk(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &falpha,
	                          d_A, CUDA_R_8I, m,
	                          d_B, CUDA_R_8I, k,
	                          &fbeta, dfC, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

		gpuBLASchk(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
	                          d_A, CUDA_R_8I, m,
	                          d_B, CUDA_R_8I, k,
	                          &beta, d_C, CUDA_R_32I, m, CUDA_R_32I, CUBLAS_GEMM_DEFAULT));


		gpuBLASchk(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &falpha,
	                          d_A, CUDA_R_8I, m,
	                          d_B, CUDA_R_8I, k,
	                          &fbeta, dfC, CUDA_R_32F, m));

		// gpuBLASchk(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
	 //                          d_A, CUDA_R_8I, m,
	 //                          d_B, CUDA_R_8I, k,
	 //                          &beta, d_C, CUDA_R_32I, m));
	}


	std::cout << "Tensor ops:" << std::endl;



	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	for (int i = 0; i < 10; i++){
		gpuBLASchk(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                          d_A, CUDA_R_8I, m,
                          d_B, CUDA_R_8I, k,
                          &beta, d_C, CUDA_R_32I, m, CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));


		gpuBLASchk(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &falpha,	
                 		  d_A, CUDA_R_8I, m,
                          d_B, CUDA_R_8I, k,
                          &fbeta, dfC, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	}
	std::cout << "Done" << std::endl;
	return 0;

}