#include "beamformer.cuh"



int main(){
	std::cout << "hello" << std::endl;

	std::ofstream f;
	f.open("bin/data.py");
	f << "A = [[";

	int N_DIRS = 1024;

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

	/* GPU Variables */
	CxInt8_t *d_A; 				// Weight matrix (N_BEAMS X N_ANTENNAS, for N_FREQUENCIES)
	CxInt8_t *d_B; 				// Data Matrix (N_ANTENNAS X N_TIMESTEPS_PER_CALL, for N_FREQUENCIES)
	char *d_data;			// Raw input data (Before data massaging)
	cuComplex *d_C;				// Beamformed output (N_BEAMS X N_TIMESTEPS_PER_CALL, for N_FREQUENCIES)
	float *d_out;			// Data after being averaged over 16 time samples and 2 polarizations
	float *d_dedispersed;	// Data after being de-dispersed
	float *d_vec_ones;		// Vector of all ones for de-dispersion

	/* HOST Variables */
	CxInt8_t *A = new CxInt8_t[A_cols*A_rows*N_FREQUENCIES];
	CxInt8_t *B = new CxInt8_t[B_cols*B_rows*N_FREQUENCIES];
	char *data = new char[BYTES_PER_GEMM*N_BLOCKS_on_GPU]; //should be the size of one "dada block", data is 4-bit so real/imag is packed into one 8-bit char
	float *out_dedispersed = new float[N_BEAMS];
	float *vec_ones = new float[N_FREQUENCIES];

	// thrust::constant_iterator<float> vec_o(1)

	float* pos = new float[N_ANTENNAS];		// Locations of antennas
	float* dir = new float[N_BEAMS];		// Direction of beams
	int gpu = 0;							// Unique identifier for each GPU

	cuComplex h_inv_max_value, h_zero;//, one;
	cuComplex *d_inv_max_value, *d_zero;
	h_inv_max_value.x = 1.0/MAX_VAL;
	h_inv_max_value.y = 0;
	h_zero.x = 0;
	h_zero.y = 0;
	float h_f_one = 1.0;
	float h_f_zero = 0.0;
	float *d_f_one, *d_f_zero;


	/* Populate location/direction Matricies */
	for (int i = 0; i < N_ANTENNAS; i++){
		pos[i] = i*500.0/(N_ANTENNAS-1) - 250.0;
	}

	/* Directions for Beamforming */
	for (int i = 0; i < N_BEAMS; i++){
		dir[i] = i*DEG2RAD(7.0)/(N_BEAMS-1) - DEG2RAD(3.5);
	}

	/* Create vector of ones for Dedispersion */
	for (int i = 0; i < N_FREQUENCIES; i++){
		vec_ones[i] = 1.0;
	}


	// Fourier Coefficient Matrix
	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*bw_per_channel;
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
	// int test_frequency = 10;


	/* Allocate and Move Memory to Device */
	gpuErrchk(cudaMalloc(&d_A, 	A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t)));
	gpuErrchk(cudaMalloc(&d_B, 	B_rows*B_cols*N_FREQUENCIES*sizeof(CxInt8_t)));
	gpuErrchk(cudaMalloc(&d_C, 	C_rows*C_cols*N_FREQUENCIES*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_data, BYTES_PER_GEMM*N_BLOCKS_on_GPU));
	gpuErrchk(cudaMalloc(&d_out,  N_BEAMS*N_FREQUENCIES * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_dedispersed, N_BEAMS*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_vec_ones, N_BEAMS*sizeof(float)));

	gpuErrchk(cudaMalloc(&d_inv_max_value, sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_zero, sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_f_one, sizeof(float)));
	gpuErrchk(cudaMalloc(&d_f_zero, sizeof(float)));


	gpuErrchk(cudaMemcpy(d_A, A, A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_vec_ones, vec_ones, N_FREQUENCIES*sizeof(float), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_inv_max_value, &h_inv_max_value, sizeof(cuComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_zero, &h_zero, sizeof(cuComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_f_one, &h_f_one, sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_f_zero, &h_f_zero, sizeof(float), cudaMemcpyHostToDevice));


	cudaStream_t stream[N_STREAMS];
	cublasHandle_t handle[N_STREAMS];

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamCreate(&(stream[i])));
		gpuBLASchk(cublasCreate(&handle[i]));
		gpuBLASchk(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_DEVICE));
	}

	int current_stream = 0;



	float test_direction = DEG2RAD(-3.5) + 200*DEG2RAD(7.0)/(N_DIRS-1);
	char high, low;

	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*bw_per_channel;
		// std::cout << "freq: " << freq << std::endl;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_TIMESTEPS_PER_CALL; j++){
			for (int k = 0; k < N_ANTENNAS; k++){

				high = ((char) round(SIG_MAX_VAL*cos(2*PI*pos[k]*sin(test_direction)/wavelength))); //real
				low  = ((char) round(SIG_MAX_VAL*sin(2*PI*pos[k]*sin(test_direction)/wavelength))); //imag

				data[i*B_stride + j*N_ANTENNAS + k] = (high << 4) | (0x0F & low);
			}
		}
	}


	for (int iii = 0; iii < N_DIRS; iii++){
		if (iii %100 == 0){
			std::cout << "Direction: " << iii << std::endl;
		}

		gpuBLASchk(cublasSetStream(handle[current_stream], stream[current_stream]));
		// std::cout << "hello1" << std::endl;



		

		// for (int i = 0; i < N_FREQUENCIES; i++){
		// 	float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*bw_per_channel;
		// 	float wavelength = C_SPEED/(1E9*freq);

		// 	for (int j = 0; j < N_TIMESTEPS_PER_CALL; j++){
		// 		for (int k = 0; k < N_ANTENNAS; k++){
		// 			if (i == test_frequency){
		// 				B[i*N_TIMESTEPS_PER_CALL*N_ANTENNAS + j*N_ANTENNAS + k].x = round(MAX_VAL*cos(2*PI*pos[k]*sin(test_direction)/wavelength));
		// 				B[i*N_TIMESTEPS_PER_CALL*N_ANTENNAS + j*N_ANTENNAS + k].y = round(MAX_VAL*sin(2*PI*pos[k]*sin(test_direction)/wavelength));
		// 			}
		// 		}
		// 	}

		// }
	 	
		// int simulated_direction = 100;
		int current_block = 0;
		// int tot_avging = N_POL*N_AVERAGING;



		// std::cout << "hello2" << std::endl;

		// cudaMemcpy(d_B, B, B_rows*B_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice);
		gpuErrchk(cudaMemcpyAsync(&(d_data[BYTES_PER_GEMM*current_block]), 
									data, BYTES_PER_GEMM, 
									cudaMemcpyHostToDevice,
									stream[current_stream]));
		// std::cout << "hello2a" << std::endl;
		
		expand_input<<<1000, 32, 0, stream[current_stream]>>>(d_data, (char *) d_B, B_stride*N_FREQUENCIES);

		// cuComplex inv_max_value, zero;//, one;
		// inv_max_value.x = 1.0/MAX_VAL;
		// inv_max_value.y = 0;
		// zero.x = 0;
		// zero.y = 0;

		// std::cout << "hello2b" << std::endl;
		gpuBLASchk(cublasGemmStridedBatchedEx(handle[current_stream], CUBLAS_OP_N, CUBLAS_OP_N,
									A_rows, B_cols, A_cols,
									d_inv_max_value,
									d_A, CUDA_C_8I, A_rows, A_stride,
									d_B, CUDA_C_8I, B_rows, B_stride,
									d_zero,
									d_C, CUDA_C_32F, C_rows, C_stride,
									N_FREQUENCIES, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

		// std::cout << "hello3" << std::endl;
		detect_sum<<<N_FREQUENCIES, N_BEAMS, 0, stream[current_stream]>>>(d_C, d_out);

		// std::cout << "hello4" << std::endl;
		gpuBLASchk(cublasSgemv(handle[current_stream], CUBLAS_OP_N,
					N_BEAMS, N_FREQUENCIES,
					d_f_one,
					d_out, N_BEAMS,
					d_vec_ones, 1,
					d_f_zero,
					d_dedispersed, 1));


		//gemv to dedisperse
		//copy to host
		//sfml image

		gpuErrchk(cudaMemcpyAsync(out_dedispersed, 
								  d_dedispersed, N_BEAMS*sizeof(float), 
								  cudaMemcpyDeviceToHost,
								  stream[current_stream]));

		#if DEBUG
			for (int i = 0; i < N_BEAMS; i++){
				f << out_dedispersed[i];
				if (i != N_BEAMS - 1){
					f << ",";
				}
			}

			if (iii != N_DIRS-1){
				f << "],\n[";
			} else {
				f<< "]]";
			}
		#endif


		/* Increment Stream Counter */
		current_stream++;
		if (current_stream >= N_STREAMS){
			current_stream = 0;
		}
		// std::cout << "cs : " << current_stream << std::endl;

	}

	#if 0
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
	#endif

	// cuProfilerStop();

	f.close();

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamDestroy(stream[i]));
		gpuBLASchk(cublasDestroy(handle[i]));
	}

	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_C));
	gpuErrchk(cudaFree(d_B));
	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_dedispersed));
	gpuErrchk(cudaFree(d_vec_ones));

	delete[] vec_ones;
	delete[] A;
	delete[] out_dedispersed;
	delete[] data;
	delete[] B;
	delete[] pos;
	delete[] dir;

	
	return 0;
}