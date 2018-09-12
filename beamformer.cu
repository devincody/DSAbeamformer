#include <iostream>
#include <cublas_v2.h>


int main(){
	
	std::cout << "hello" << std::endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cuComplex temp;
	temp.x = 0;
	temp.y = 3;
	std::cout << "test:" << temp.x << std::endl;


	CUDA_R_16F te = 12.4;
	std::cout << "rt " << te << std::endl;

	cublasDestroy(handle);
	return 0;
}