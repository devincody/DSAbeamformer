#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <iostream>


typedef char2 CxInt8_t;


// C-style indexing
int ci(int row, int column, int nColumns) {
	return row*nColumns+column;
}

int main(void)
{
	// cudaSetDevice(3);
	int rowD = 4 ; // number of rows of D
	int colD = 4; // number of columns of D
	int rowE = colD; // number of rows of E
	int colE = 4; // number of columns of E
	int rowF = rowD;
	int colF = colE;

	std::cout << "hello " << std::endl;

	CxInt8_t test;

	test.x = -1;
	test.y = 3;

	std::cout << "test: " << (int) test.x << "y: " << (int) test.y << std::endl;


	CxInt8_t *D = new CxInt8_t[rowD * colD];
	CxInt8_t *E = new CxInt8_t[rowE * colE];
	cuComplex *F = new cuComplex[rowF * colF];

	

	for (size_t i = 0; i < rowD; i++){
		for (size_t j = 0; j < colD; j++){
			D[ci(i,j,colD)].x=(i+j) ;
			D[ci(i,j,colD)].y=0; ;
			std::cout << (int) D[ci(i,j,colD)].x << " ";
			}
		std::cout << "\n";
	}

	for (size_t i = 0; i < rowE; i++){
		for (size_t j = 0; j < colE; j++){
			E[ci(i,j,colE)].x=(i+j);
			D[ci(i,j,colD)].y=1;
			//std::cout << (int) E[ci(i,j,colE)].x << " ";
			}
		//std::cout << "\n";
	}

	for (size_t i = 0; i < rowF; i++){
		for (size_t j = 0; j < colF; j++){
			F[ci(i,j,colF)].x=0;	
			F[ci(i,j,colF)].y=0;
		}
	}
		

	cublasHandle_t handle;

	/* Initialize CUBLAS */
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "!!!! CUBLAS initialization error\n";
	}

	CxInt8_t *d_D, *d_E;
	cuComplex *d_F;
	cudaMalloc(&d_D, rowD * colD*sizeof(CxInt8_t));
	cudaMalloc(&d_E, rowE * colE*sizeof(CxInt8_t));
	cudaMalloc(&d_F, rowF * colF*sizeof(cuComplex));

	cudaMemcpy(d_D, D, rowD * colD*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, E, rowE * colE*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F, rowF * colF*sizeof(cuComplex), cudaMemcpyHostToDevice);

	cuComplex alpha;
	cuComplex beta;

	alpha.x = 1;
	alpha.y = 0;
	beta.x = 0;
	beta.y = 0;

	#if 0
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
	colE, rowD, colD,
	&alpha, thrust::raw_pointer_cast(&E[0]), colE,
	thrust::raw_pointer_cast(&D[0]), colD,
	&beta, thrust::raw_pointer_cast(&F[0]), colE);// colE x rowD
	#endif

	status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
	colE, rowD, colD,
	&alpha, d_E, CUDA_C_8I ,colE,
	d_D, CUDA_C_8I ,colD,
	&beta, d_F, CUDA_C_32F ,colE, CUDA_C_32F,CUBLAS_GEMM_DFALT);// colE x rowD

	std::cout << "hello " << std::endl;

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "!!!! 0 kernel execution error.\n" << status << std::endl;
	}

	cudaMemcpy(F, d_F, rowF*colF*sizeof(cuComplex), cudaMemcpyDeviceToHost);

	#if 1
	for (size_t i = 0; i < rowF; i++){
		for (size_t j = 0; j < colF; j++){
			std::cout << (int) F[ci(i,j,colF)].x << "+" << (int) F[ci(i,j,colF)].y << "j ";
		}
		std::cout << std::endl;
	}
	#endif

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "!!!! shutdown error (A)\n";
	}


	return 0;
} 