#include <iostream>
#include <cublas_v2.h>
#include "nvToolsExt.h"
// #include <cuda_runtime.h>

#define PUSH_NVTX_RANGE(name,cid)  \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \



    
// cudaEvent_t start;
// cudaEvent_t stop;

// #define START_TIMER() {                         \
// 	gpuErrchk(cudaEventCreate(&start));       \
// 	gpuErrchk(cudaEventCreate(&stop));        \
// 	gpuErrchk(cudaEventRecord(start));        \
// }

// #define STOP_RECORD_TIMER(name) {                           \
// 	gpuErrchk(cudaEventRecord(stop));                     \
// 	gpuErrchk(cudaEventSynchronize(stop));                \
// 	gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
// 	gpuErrchk(cudaEventDestroy(start));                   \
// 	gpuErrchk(cudaEventDestroy(stop));                    \
// }

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
	
	std::cout << "hello " << std::endl;

	cublasHandle_t handle;
	gpuBLASchk(cublasCreate(&handle));
	gpuBLASchk(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));


	int N =10;
	std::cout  << "N = " << N << std::endl;


	cuComplex *data = new cuComplex[N];

	for (int i = 0; i < N; i++){
		data[i].y = 0;
		data[i].x = i + .5;
	}

	for (int i = 0; i < 10; i ++){
		std::cout << "x[" << i << "] = " << data[i].x << " + " << data[i].y << "j" << std::endl;
	}
	std::cout << std::endl;

	cuComplex *d_data, *d_data2, *detect;

	gpuErrchk(cudaMalloc(&d_data, N*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_data2, N*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&detect, 1*sizeof(cuComplex)));

	gpuErrchk(cudaMemcpy(d_data, data, N*sizeof(cuComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_data2, data, N*sizeof(cuComplex), cudaMemcpyHostToDevice));


	cuComplex ans;

	gpuBLASchk(cublasCdotc(handle, N,
							d_data, 1,
							d_data2, 1,
							detect));

	std::cout << "done" << std::endl;
	// gpuBLASchk(cublasCdotc(handle, n_timesteps*n_pol,
	// 						&d_data[f*n_beams*n_time_pol_out + o*n_timesteps*n_pol], n_beams,
	// 						&d_data2[f*n_beams*n_time_pol_out + o*n_timesteps*n_pol], n_beams,
	// 						&detect[f*n_outputs_per_input*o]));
	gpuErrchk(cudaMemcpy(&ans, detect, 1*sizeof(cuComplex), cudaMemcpyDeviceToHost));
	std::cout << " data = " << ans.x <<std::endl;




	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(detect));
	gpuErrchk(cudaFree(d_data2));

	gpuBLASchk(cublasDestroy(handle));

	delete[] data;
	return 0;
}






