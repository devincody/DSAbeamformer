#include "beamformer.cuh"

// nvcc src/beamformer.cu -o bin/beam -lcublas

//sudo dada_db -k baab -d
// sudo dada_db -k baab -n 8 -b 268435456 -l -p
//bin/beam -k baab -c 0 -g 0


int main(int argc, char *argv[]){
	
	int gpu = 0;
	int observation_complete=0;

	/***************************************************
	DADA VARIABLES
	***************************************************/
	#ifndef DEBUG
		dada_hdu_t* hdu_in = 0;
		multilog_t* log = 0;
		int core = -1;
		key_t in_key = 0x0000dada;
		uint64_t header_size = 0;
	#endif

	/***************************************************
	Parse Command Line Options
	***************************************************/
	#ifndef DEBUG
	int arg = 0;
	while ((arg=getopt(argc,argv,"c:k:g:h")) != -1) {
		switch (arg) {
		// to bind to a cpu core
			case 'c':
				if (optarg){
					core = atoi(optarg);
					break;
				} else {
					printf ("ERROR: -c flag requires argument\n");
					return EXIT_FAILURE;
				}
				// to set the dada key
			case 'k':
				if (sscanf (optarg, "%x", &in_key) != 1) {
				fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
				return EXIT_FAILURE;
				}
				break;
			case 'g':
				if (optarg){
					gpu = atoi(optarg);
					break;
				} else {
					printf ("ERROR: -g flag requires argument\n");
					return EXIT_FAILURE;
				}
			case 'h':
				usage();
				return EXIT_SUCCESS;
		}
	}
	#endif



	/***********************************
	 *			GPU Variables		   *
	 ***********************************/

	std::cout << "Executing beamformer.cu" << std::endl;
	print_all_defines();

	char prefered_dev_name[] = "GeForce GTX 1080";
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
	    cudaDeviceProp deviceProperties;
	    cudaGetDeviceProperties(&deviceProperties, deviceIndex);
	    if (prefered_dev_name[14] == deviceProperties.name[14]){
		    std::cout <<  "Selected: " << deviceProperties.name << std::endl;
		    std::cout << "letter: " << prefered_dev_name[14] << std::endl;
		    gpuErrchk(cudaSetDevice(deviceIndex));
		}
	}


	/* CUBLAS Dimensions */
	int A_rows	 = N_BEAMS;
	int A_cols 	 = N_ANTENNAS;
	int A_stride = A_rows*A_cols;
	int B_cols	 = N_TIMESTEPS_PER_GEMM;
	int B_rows	 = A_cols;
	int B_stride = B_rows*B_cols;
	int C_rows	 = A_rows;
	int C_cols	 = B_cols;
	int C_stride = C_rows*C_cols;
	float bw_per_channel = BW_PER_CHANNEL; 

	/***********************************
	 *			GPU Variables		   *
	 ***********************************/
	CxInt8_t *d_A; 				// Weight matrix (N_BEAMS X N_ANTENNAS, for N_FREQUENCIES)
	CxInt8_t *d_B; 				// Data Matrix (N_ANTENNAS X N_TIMESTEPS_PER_GEMM, for N_FREQUENCIES)
	char *d_data;				// Raw input data (Before data massaging)
	cuComplex *d_C;				// Beamformed output (N_BEAMS X N_TIMESTEPS_PER_GEMM, for N_FREQUENCIES)
	float *d_out;				// Data after being averaged over 16 time samples and 2 polarizations

	/* CUBLAS Constants */
	cuComplex  h_inv_max_value,  h_zero; 		//Host Values
	cuComplex *d_inv_max_value, *d_zero; 	//Device Values

	h_inv_max_value.x = 1.0/MAX_VAL;
	h_inv_max_value.y = 0;
	h_zero.x = 0;
	h_zero.y = 0;

	#if DEBUG
		float  h_f_one,  h_f_zero;				//Host Values
		float *d_f_one, *d_f_zero;				//Device Values	
		h_f_one = 1.0;
		h_f_zero = 0.0;
	#endif

	/***********************************
	 *			HOST Variables		   *
	 ***********************************/
	#if DEBUG
		char *data = new char[N_BYTES_PRE_EXPANSION_PER_GEMM*N_DIRS]();
		gpuErrchk(cudaHostRegister(data, N_BYTES_PRE_EXPANSION_PER_GEMM*N_DIRS*sizeof(char), cudaHostRegisterPortable)); //need pinned memory
	#endif

	CxInt8_t *A = new CxInt8_t[A_cols*A_rows*N_FREQUENCIES];
	float *beam_out = new float[N_F_PER_DETECT*N_STREAMS]();
	gpuErrchk(cudaHostRegister(beam_out, N_FREQUENCIES*N_BEAMS*N_OUTPUTS_PER_GEMM*N_STREAMS*sizeof(float), cudaHostRegisterPortable)); //need pinned memory



	#if DEBUG
		std::cout << "data size: " << N_BYTES_PRE_EXPANSION_PER_GEMM*N_DIRS << std::endl;
		std::ofstream f;
		f.open("bin/data.py");
		f << "A = [[";
		// std::mutex file_mutex;

		float *d_dedispersed;	// Data after being de-dispersed
		
		float *out_dedispersed = new float[N_BEAMS*N_DIRS]();

		gpuErrchk(cudaHostRegister(out_dedispersed, N_BEAMS*N_DIRS*sizeof(float), cudaHostRegisterPortable));
	#endif

	float *d_vec_ones;		// Vector of all ones for de-dispersion
	float *vec_ones = new float[N_FREQUENCIES];


	/***********************************
	 *		Beamforming Variables	   *
	 ***********************************/
	float* pos = new float[N_ANTENNAS];		// Locations of antennas
	float* dir = new float[N_BEAMS];		// Direction of bemformed beams

	/* Populate location/direction Matricies */
	for (int i = 0; i < N_ANTENNAS; i++){
		pos[i] = i*500.0/(N_ANTENNAS-1) - 250.0;
	}

	/* Directions for Beamforming */
	for (int i = 0; i < N_BEAMS; i++){
		dir[i] = i*DEG2RAD(7.0)/(N_BEAMS-1) - DEG2RAD(3.5);
	}

	#if DEBUG
	/* Create vector of ones for Dedispersion */
	for (int i = 0; i < N_FREQUENCIES; i++){
		vec_ones[i] = 1.0;
	}
	#endif


	/* Fourier Coefficient Matrix */
	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_ANTENNAS; j++){
			for (int k = 0; k < N_BEAMS; k++){
				A[i*A_stride + j*N_BEAMS + k].x = round(MAX_VAL*cos(-2*PI*pos[j]*sin(dir[k])/wavelength));
				A[i*A_stride + j*N_BEAMS + k].y = round(MAX_VAL*sin(-2*PI*pos[j]*sin(dir[k])/wavelength));
			}
		}
	}


	/***********************************
	 *			Memory Allocation 	   *
	 ***********************************/
	gpuErrchk(cudaMalloc(&d_A, 	A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t)));
	gpuErrchk(cudaMalloc(&d_B, 	N_CX_IN_PER_GEMM*N_STREAMS*sizeof(CxInt8_t)));
	gpuErrchk(cudaMalloc(&d_C, 	N_CX_OUT_PER_GEMM*N_STREAMS*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_data, N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK*N_BLOCKS_on_GPU)); 							// array for raw data
	gpuErrchk(cudaMalloc(&d_out, N_F_PER_DETECT*N_STREAMS * sizeof(float)));			// array for detected, averaged data

	/* Cublas Constant Memory */
	gpuErrchk(cudaMalloc(&d_inv_max_value, sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_zero, sizeof(cuComplex)));
	


	#if DEBUG
		gpuErrchk(cudaMalloc(&d_vec_ones, N_FREQUENCIES*sizeof(float)));
		gpuErrchk(cudaMalloc(&d_f_one, sizeof(float)));
		gpuErrchk(cudaMalloc(&d_f_zero, sizeof(float)));
		gpuErrchk(cudaMalloc(&d_dedispersed, N_BEAMS*N_STREAMS*sizeof(float)));						// array for frequency averaged data
		
	#endif

	/* Copy constants to memory */
	gpuErrchk(cudaMemcpy(d_A, A, A_rows*A_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_inv_max_value, &h_inv_max_value, sizeof(cuComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_zero, &h_zero, sizeof(cuComplex), cudaMemcpyHostToDevice));
	


	#if DEBUG
		gpuErrchk(cudaMemcpy(d_vec_ones, vec_ones, N_FREQUENCIES*sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_f_one, &h_f_one, sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_f_zero, &h_f_zero, sizeof(float), cudaMemcpyHostToDevice));
		std::cout << "First: " << h_f_zero << " and " << h_f_one << std::endl;
		print_data_scalar<<<1, 1>>>(d_f_one);
		print_data_scalar<<<1, 1>>>(d_f_zero);
	#endif

	// gpuErrchk(cudaMemset(d_dedispersed, 0, N_BEAMS*N_STREAMS*sizeof(float)));

	/***********************************
	 *		Concurrency Handles		   *
	 ***********************************/

	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

	cudaStream_t HtoDstream;
	cudaStream_t stream[N_STREAMS];

	cublasHandle_t handle[N_STREAMS];
	// std::thread thread[N_STREAMS];
	int timeSlice[N_STREAMS];
	cudaEvent_t BlockTransferredSync[5*N_BLOCKS_on_GPU];
	cudaEvent_t BlockAnalyzeddSync[5*N_BLOCKS_on_GPU];

	// gpuErrchk(cudaStreamCreateWithPriority(&HtoDstream, cudaStreamNonBlocking, priority_high));
	gpuErrchk(cudaStreamCreate(&HtoDstream));

	for (int i = 0; i < N_STREAMS; i++){
		// gpuErrchk(cudaStreamCreateWithPriority(&stream[i], cudaStreamNonBlocking, priority_high));
		gpuErrchk(cudaStreamCreate(&stream[i]));
		gpuBLASchk(cublasCreate(&handle[i]));
		gpuBLASchk(cublasSetStream(handle[i], stream[i]));
		gpuBLASchk(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_DEVICE));
		// cublasMath_t mode; 
		// gpuBLASchk(cublasGetMathMode(handle[i], &mode));
		// std::cout << "Mode[" << i << "] == " << mode << std::endl;
		gpuBLASchk(cublasSetMathMode(handle[i], CUBLAS_TENSOR_OP_MATH ));
		// gpuBLASchk(cublasGetMathMode(handle[i], &mode));
		// std::cout << "Mode[" << i << "] == " << mode << std::endl;
		timeSlice[i] = i;
	}
	

	for (int i = 0; i < 5*N_BLOCKS_on_GPU; i++){
		gpuErrchk(cudaEventCreateWithFlags(&BlockTransferredSync[i], cudaEventDisableTiming));
		gpuErrchk(cudaEventCreateWithFlags(&BlockAnalyzeddSync[i], cudaEventDisableTiming));
	}

	/***********************************
	 *			TEST SIGNAL			   *
	 ***********************************/

	#if DEBUG
		#if 1
			float test_direction;
			char high, low;
			for (int iii = 0; iii < N_DIRS; iii++){
				test_direction = DEG2RAD(-3.5) + iii*DEG2RAD(7.0)/(N_DIRS-1);
				for (int i = 0; i < N_FREQUENCIES; i++){
					float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*BW_PER_CHANNEL;
					// std::cout << "freq: " << freq << std::endl;
					float wavelength = C_SPEED/(1E9*freq);
					for (int j = 0; j < N_TIMESTEPS_PER_GEMM; j++){
						for (int k = 0; k < N_ANTENNAS; k++){

							high = ((char) round(SIG_MAX_VAL*cos(2*PI*pos[k]*sin(test_direction)/wavelength))); //real
							low  = ((char) round(SIG_MAX_VAL*sin(2*PI*pos[k]*sin(test_direction)/wavelength))); //imag

							data[iii*N_BYTES_PRE_EXPANSION_PER_GEMM + i*B_stride + j*N_ANTENNAS + k] = (high << 4) | (0x0F & low);
						}
					}
				}
			}
		#else
			memset(data, 0x70, N_BYTES_PRE_EXPANSION_PER_GEMM*N_DIRS*sizeof(char));
			std::cout << "BOGUS DATA " << std::endl;
		#endif
	#endif 

	// std::cout << "done writing data" << std::endl;

	// int observation_complete = 0;
	uint64_t blocks_analyzed = 0;
	uint64_t blocks_transferred = 0;
	uint64_t blocks_analysis_queue = 0;
	uint64_t blocks_transfer_queue = 0;

	#if DEBUG
		int current_gemm = 0;
	#endif



	/***************************************************
	Initialize hdu (FOR DADA)
	***************************************************/

	#ifndef DEBUG
	// DADA stuff
	log = multilog_open ("real", 0);
	multilog_add (log, stderr);
	multilog (log, LOG_INFO, "creating hdu\n");

	// create dada hdu
	hdu_in	= dada_hdu_create (log);
	// set the input hdu key
	dada_hdu_set_key (hdu_in, in_key);

	// connect to dada buffer
	if (dada_hdu_connect (hdu_in) < 0) {
		printf ("could not connect to dada buffer\n");
		return EXIT_FAILURE;
	}

	// lock read on buffer
	if (dada_hdu_lock_read (hdu_in) < 0) {
		printf ("could not lock to dada buffer\n");
		return EXIT_FAILURE;
	}

	// Bind to cpu core
	if (core >= 0)
	{
		printf("binding to core %d\n", core);
		if (dada_bind_thread_to_core(core) < 0)
		printf("failed to bind to core %d\n", core);
	}

	multilog (log, LOG_INFO, "Done setting up buffer\n");
	#endif

	/***************************************************
	Deal with Headers (FOR DADA)
	***************************************************/
	#ifndef DEBUG
	// read the headers from the input HDU and mark as cleared
	// will block until header is present in dada ring buffer
	char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
	if (!header_in)
	{
		multilog(log ,LOG_ERR, "main: could not read next header\n");
		dsaX_dbgpu_cleanup (hdu_in, log);
		return EXIT_FAILURE;
	}

	if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
	{
		multilog (log, LOG_ERR, "could not mark header block cleared\n");
		dsaX_dbgpu_cleanup (hdu_in, log);
		return EXIT_FAILURE;
	}

	// size of block in dada buffer
	uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
	uint64_t bytes_read = 0, block_id;
	char *block;

	multilog (log, LOG_INFO, "Done setting up header \n");
	
	std::cout << "block size is: " << block_size << std::endl;
	#endif



int transfer_separation = 2;
int total_separation = 5;

/*******************************************************************************************************************************
									START OBSERVATION LOOP
*******************************************************************************************************************************/
	while (!observation_complete){
		
		#if VERBOSE
			std::cout << "##########################################" << std::endl;
			std::cout << "A: " << blocks_analyzed <<  ", AQ: " << blocks_analysis_queue << ", T: " << blocks_transferred << ", TQ: " << blocks_transfer_queue << std::endl;
		#endif 



		if ((blocks_transfer_queue - blocks_analyzed < total_separation) && (blocks_transfer_queue - blocks_transferred < transfer_separation)){

			std::cout << "index: " << N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK*(blocks_transfer_queue%N_BLOCKS_on_GPU) << std::endl;
			std::cout << "index: " << N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK*(blocks_transfer_queue) << std::endl;
			std::cout << "index: " << N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK << std::endl;


			#if DEBUG
				if (blocks_transfer_queue < N_DIRS/N_GEMMS_PER_BLOCK){
					#if VERBOSE 
						// multilog(log, LOG_INFO, "A: Open new block for analysis\n");
						std::cout << "DEBUG: Async copy" << std::endl;
					#endif
					gpuErrchk(cudaMemcpyAsync(&d_data[N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK*(blocks_transfer_queue%N_BLOCKS_on_GPU)], 
												&data[N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK*blocks_transfer_queue],
												N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK, 
												cudaMemcpyHostToDevice,
												HtoDstream));
					std::cout << "DEBUG: transfer scheduled" << std::endl;

					gpuErrchk(cudaEventRecord(BlockTransferredSync[blocks_transfer_queue%(5*N_BLOCKS_on_GPU)], HtoDstream));
					blocks_transfer_queue++;
				}
			#else
				block = ipcio_open_block_read(hdu_in->data_block,&bytes_read, &block_id);

				#if VERBOSE
					// multilog(log, LOG_INFO, "A: Open new block for analysis\n");
					std::cout << "Async, Bytes Read: " << bytes_read << ", Should also be 268435456" << std::endl;
				#endif

				if (bytes_read < block_size){
					// multilog(log, LOG_INFO, "Not enough bytes\n");
					observation_complete = 1;
					// cudaProfilerStop();
					break;
				}

				gpuErrchk(cudaMemcpyAsync(&d_data[N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK*(blocks_transfer_queue%N_BLOCKS_on_GPU)], 
											block,
											N_BYTES_PRE_EXPANSION_PER_GEMM*N_GEMMS_PER_BLOCK, 
											cudaMemcpyHostToDevice,
											HtoDstream));

				ipcio_close_block_read (hdu_in->data_block, bytes_read);
				gpuErrchk(cudaEventRecord(BlockTransferredSync[blocks_transfer_queue%(5*N_BLOCKS_on_GPU)], HtoDstream));
				blocks_transfer_queue++;
			#endif


			
			
		}
		std::cout << "hello2a" << std::endl;


		for (uint64_t event = blocks_transferred; event < blocks_transfer_queue; event ++){
			if(cudaEventQuery(BlockTransferredSync[event%(5*N_BLOCKS_on_GPU)]) == cudaSuccess){
				blocks_transferred ++;
				#if VERBOSE
					std::cout << "async, Surplus of blocks, Asynchronous copy\n";
					std::cout << "async, Transferred: " << blocks_transferred << " Analyzed: " <<blocks_analyzed << std::endl;
					std::cout << "async, Transferred Q: " << blocks_transfer_queue << " Analyzed Q : " <<blocks_analysis_queue << std::endl;
				#endif
				gpuErrchk(cudaEventDestroy(BlockTransferredSync[event%(5*N_BLOCKS_on_GPU)]));
				gpuErrchk(cudaEventCreateWithFlags(&BlockTransferredSync[event%(5*N_BLOCKS_on_GPU)], cudaEventDisableTiming));
			} else {
				break;
			}
		}

		
		if (blocks_analysis_queue < blocks_transferred){
			for (int part = 0; part < N_GEMMS_PER_BLOCK/N_STREAMS; part++){

				#if VERBOSE
					std::cout << "Queueing slavo. Analyzed = " << blocks_analyzed << " Transferred = " << blocks_transferred << " Start Dir = " << blocks_analysis_queue*N_GEMMS_PER_BLOCK + timeSlice[0] << std::endl; 
				#endif

				for (int st = 0; st < N_STREAMS; st++){
					// gpuErrchk(cudaStreamWaitEvent(stream[st], BlockTransferredSync[(blocks_analyzed+1)%N_BLOCKS_on_GPU], 0));
					// gpuErrchk(cudaStreamSynchronize(stream[st]));
					//cudaStreamSynchronize(stream[st]);

					expand_input<<<10000, 32, 0, stream[st]>>>(&d_data[N_BYTES_PRE_EXPANSION_PER_GEMM*(N_GEMMS_PER_BLOCK*(blocks_analysis_queue%N_BLOCKS_on_GPU) + timeSlice[st])],
														      (char *) &d_B[N_CX_IN_PER_GEMM*st], 
														      B_stride*N_FREQUENCIES);

					// std::cout << "hello2b" << std::endl;
					gpuBLASchk(cublasGemmStridedBatchedEx(handle[st], CUBLAS_OP_N, CUBLAS_OP_N,
												A_rows, B_cols, A_cols,
												d_inv_max_value,
												d_A, CUDA_C_8I, A_rows, A_stride,
												&d_B[N_CX_IN_PER_GEMM*st], CUDA_C_8I, B_rows, B_stride,
												d_zero,
												&d_C[N_CX_OUT_PER_GEMM*st], CUDA_C_32F, C_rows, C_stride,
												N_FREQUENCIES, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

					// std::cout << "hello3" << std::endl;
					detect_sum<<<detect_dimGrid, detect_dimBlock, 0, stream[st]>>>(&d_C[N_CX_OUT_PER_GEMM*st], N_INPUTS_PER_OUTPUT, &d_out[st*N_F_PER_DETECT]);

					// print_data_scalar<<<1,1,0,stream[st]>>>(&d_out[st*N_F_PER_DETECT]);

					gpuErrchk(cudaMemcpyAsync(&beam_out[st*N_F_PER_DETECT], 
											  &d_out[st*N_F_PER_DETECT], N_OUTPUTS_PER_GEMM*N_FREQUENCIES*N_BEAMS*sizeof(float), 
											  cudaMemcpyDeviceToHost,
											  stream[st]));

					#if DEBUG
						current_gemm = blocks_analysis_queue*N_GEMMS_PER_BLOCK + timeSlice[st];

						// print_data<<<1, 10,0, stream[st]>>>(&d_out[st*N_F_PER_DETECT]);
						gpuBLASchk(cublasSgemv(handle[st], CUBLAS_OP_N,
									N_BEAMS, N_FREQUENCIES,
									d_f_one,
									&d_out[st*N_F_PER_DETECT], N_BEAMS,
									d_vec_ones, 1,
									d_f_zero,
									&d_dedispersed[st*N_BEAMS], 1));

						// gpuErrchk(cudaStreamSynchronize(stream[st]));
						// std::cout << "Second:" << std::endl;
						// print_data<<<1, 10>>>(&d_dedispersed[st*N_BEAMS]);
						gpuErrchk(cudaMemcpyAsync(&out_dedispersed[current_gemm*N_BEAMS], 
												  &d_dedispersed[st*N_BEAMS], N_BEAMS*sizeof(float), 
												  cudaMemcpyDeviceToHost,
												  stream[st]));

					#endif

					// std::cout << "timeSlice["<< st << "] = " << timeSlice[st] << " and " << current_gemm << std::endl;
					
					timeSlice[st] += N_STREAMS; // increment so the next time slice is processed next

					if (timeSlice[st] >= N_GEMMS_PER_BLOCK){
						timeSlice[st] -= N_GEMMS_PER_BLOCK; //wrap back to the beginning once each gemm in a block has been processed
					}

				}
			}
			gpuErrchk(cudaEventRecord(BlockAnalyzeddSync[blocks_analysis_queue%(5*N_BLOCKS_on_GPU)], stream[N_STREAMS-1]));
			blocks_analysis_queue ++;
		}



		// if (timeSlice[st] == N_GEMMS_PER_BLOCK-1){
		/* Check to see which blocks have been successfully analyzed */
		for (uint64_t event = blocks_analyzed; event < blocks_analysis_queue; event ++){
			if(cudaEventQuery(BlockAnalyzeddSync[blocks_analyzed%(5*N_BLOCKS_on_GPU)]) == cudaSuccess){
				//This is incremented once each time slice in each block is analyzed (or more accurately, scheduled)
				blocks_analyzed++;
				#if VERBOSE
					std::cout<< "Done analyzing block. Analyzed = " << blocks_analyzed << " Transferred = " << blocks_transferred <<std::endl;
					std::cout << "async, Transferred Q: " << blocks_transfer_queue << " Analyzed Q : " << blocks_analysis_queue << std::endl;
				#endif
				gpuErrchk(cudaEventDestroy(BlockAnalyzeddSync[blocks_analyzed%(5*N_BLOCKS_on_GPU)]));
				gpuErrchk(cudaEventCreateWithFlags(&BlockAnalyzeddSync[blocks_analyzed%(5*N_BLOCKS_on_GPU)], cudaEventDisableTiming));
				#if DEBUG
					if ((current_gemm == N_DIRS-1) && (blocks_analyzed == blocks_transfer_queue)){
						observation_complete = 1;
						std::cout << "obs Complete" << std::endl;
						break;
					}
				#endif
				
			} else {
				break;
			}
		}

		// std::cout << "cs : " << current_stream << std::endl;
	} // end while (!observation_complete)


	// cuProfilerStop();

	for (int st = 0; st < N_STREAMS; st++){
		gpuErrchk(cudaStreamSynchronize(stream[st]));
	}
	std::cout << "Synchronized" << std::endl;

	for (int event = 0; event < 5*N_BLOCKS_on_GPU; event++){
		gpuErrchk(cudaEventDestroy(BlockAnalyzeddSync[event]));
		gpuErrchk(cudaEventDestroy(BlockTransferredSync[event]));
	}


	#if DEBUG
		for (int jj = 0; jj < N_DIRS; jj++){
			for (int ii = 0; ii < N_BEAMS; ii++){
				f << out_dedispersed[jj*N_BEAMS + ii];
				// std::cout << out_dedispersed[jj*N_BEAMS + ii] << ", ";
				if (ii != N_BEAMS - 1){
					f << ",";
				}
			}

			if (jj != N_DIRS-1){
				f << "],\n[";
			} else {
				f<< "]]"<<std::endl;
			}
		}

		f.close();
	#endif

	std::cout << "freeing" << std::endl;

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamDestroy(stream[i]));
		gpuBLASchk(cublasDestroy(handle[i]));
	}

	std::cout << "freeing cuda" << std::endl;

	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_C));
	gpuErrchk(cudaFree(d_B));
	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_inv_max_value));
	gpuErrchk(cudaFree(d_zero));

	#if DEBUG
		gpuErrchk(cudaFree(d_dedispersed));
		gpuErrchk(cudaFree(d_vec_ones));
		gpuErrchk(cudaFree(d_f_one));
		gpuErrchk(cudaFree(d_f_zero));		
	#endif


	std::cout << "freeing host" << std::endl;

	// gpuErrchk(cudaHostUnregister(data));
	gpuErrchk(cudaHostUnregister(beam_out));



	delete[] A;
	delete[] pos;
	delete[] dir;
	delete[] beam_out;
	delete[] vec_ones;
	std::cout << "freed all1" << std::endl;


	#if DEBUG
		delete[] data;
		gpuErrchk(cudaHostUnregister(out_dedispersed));
		std::cout << "freed all2" << std::endl;
		delete[] out_dedispersed;
	#endif

	std::cout << "freed all4" << std::endl;

	return 0;
}







