#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

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


struct detect_functor{
	__host__ __device__
	cuComplex operator() (cuComplex &b){
		cuComplex a;
		a.y = b.y;
		a.x = b.x*b.x;
		return a;
	}
};

struct B_detect_functor{
	__host__ __device__
	void operator() (cuComplex &b){
		b.x *= b.x;
	}
};


struct C_detect_functor{
	__host__ __device__
	float operator() (cuComplex &b){
		return b.x*b.x;
	}
};


int main(){
	
	std::cout << "hello " << std::endl;

	cublasHandle_t handle;
	gpuBLASchk(cublasCreate(&handle));

	
	int n_timesteps = 16;
	int n_pol = 2;
	int n_outputs_per_input = 2;
	int n_time_pol_out = n_timesteps*n_pol*n_outputs_per_input;
	int n_beams = 256;
	int n_freq = 256;


	std::cout << "n_timesteps = " << n_timesteps << std::endl;
	std::cout << "n_pol = " << n_pol << std::endl;
	std::cout << "n_outputs_per_input = " << n_outputs_per_input << std::endl;
	std::cout << "n_time_pol_out = " << n_time_pol_out << std::endl;
	std::cout << "n_beams = " << n_beams << std::endl;
	std::cout << "n_freq = " << n_freq << std::endl;

	int N = n_time_pol_out*n_beams*n_freq;
	std::cout  << "N = " << N << std::endl;


	cudaStream_t s1, s2;
	gpuErrchk(cudaStreamCreate(&s1));
	gpuErrchk(cudaStreamCreate(&s2));
	// int N = 10000000;

	cuComplex *data = new cuComplex[N];

	for (int i = 0; i < N; i++){
		data[i].y = 0;
		data[i].x = i + .7;
	}

	for (int i = 0; i < 10; i ++){
		std::cout << "x[" << i << "] = " << data[i].x << " + " << data[i].y << "j" << std::endl;
	}
	std::cout << std::endl;

	cuComplex *d_data, *d_data2, *detect;

	gpuErrchk(cudaMalloc(&d_data, N*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_data2, N*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&detect, n_beams*n_freq*n_outputs_per_input*sizeof(cuComplex)));

	gpuErrchk(cudaMemcpy(d_data, data, N*sizeof(cuComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_data2, data, N*sizeof(cuComplex), cudaMemcpyHostToDevice));

	// gpuErrchk(cudaMalloc(&detect, n_beams*n_freq*n_outputs_per_input*sizeof(cuComplex)));


	// thrust::device_ptr<cuComplex> dev_data(d_data);
	// thrust::device_ptr<cuComplex> dev_data2(d_data2);


	// float ms_timer = 0;
	// START_TIMER();
	// thrust::for_each(thrust::cuda::par.on(s1), dev_data, dev_data+N, B_detect_functor());
	// thrust::for_each(thrust::cuda::par.on(s2), dev_data2, dev_data2+N, B_detect_functor());
	// STOP_RECORD_TIMER(ms_timer);

	// gpuErrchk(cudaMemcpy(data, d_data, N*sizeof(cuComplex), cudaMemcpyDeviceToHost));

	// for (int i = 0; i < 10; i ++){
	// 	std::cout << "x[" << i << "] = " << data[i].x << " + " << data[i].y << "j" << std::endl;
	// }

	// std::cout << N <<" data points in " << ms_timer	 << " ms" << std::endl;

	int f = 0;
	int o = 0;

	// cudaDeviceSynchronize();

	gpuBLASchk(cublasCdotc(handle, 10,
							d_data, 1,
							d_data2, 1,
							detect));

	// gpuBLASchk(cublasCdotc(handle, n_timesteps*n_pol,
	// 						&d_data[f*n_beams*n_time_pol_out + o*n_timesteps*n_pol], n_beams,
	// 						&d_data2[f*n_beams*n_time_pol_out + o*n_timesteps*n_pol], n_beams,
	// 						&detect[f*n_outputs_per_input*o]));

	std::cout << " data = " << detect[f*n_outputs_per_input*o].x <<std::endl;

	// gpuBLASchk(cublasDotcEx(handle,
	// 						n_time_pol_out,
	// 						&d_data[f*n_beams*n_time_pol_out + o*n_timesteps*n_pol],
	// 						CUDA_C_32F,
	// 						n_beams,
	// 						&d_data2[f*n_beams*n_time_pol_out + o*n_timesteps*n_pol],
	// 						CUDA_C_32F,
	// 						n_beams,
	// 						&detect[f*n_outputs_per_input+o],
	// 						CUDA_C_32F,
	// 						CUDA_C_32F));

	// for (int f = 0; f < n_freq; f++){
	// 	for (int o = 0; o < n_outputs_per_input; o++){
	// 		std::cout << "f = " << f << " o = " << o << std::endl;

	// 	}
	// }




	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(detect));
	gpuErrchk(cudaFree(d_data2));

	gpuBLASchk(cublasDestroy(handle));

	delete[] data;
	return 0;
}



