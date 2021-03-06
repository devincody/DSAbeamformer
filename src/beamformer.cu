#include "beamformer.hh" 	// non-cuda header file
#include "beamformer.cuh"	// cuda header file

#if DEBUG
#include "test_data_generator.hh"
#else
#include "dada_handler.hh"
#endif

#include "observation_loop.hh"

int main(int argc, char *argv[]){

	/***************************************************
	Antenna Location & Beam Direction Variables
	Will be set with a command line option (filename),
	or with test positions.
	***************************************************/

	antenna* pos = new antenna[N_ANTENNAS]();				// Locations of antennas
	beam_direction* dir = new beam_direction[N_BEAMS]();	// Direction of bemformed beams

	bool pos_set = false;
	bool dir_set = false;

	#if DEBUG
		test_data_generator input_data_generator;

		// beam_direction* sources; //= new beam_direction[n_pt_sources]();  // Array to hold direction of the test sources
		// bool use_source_catalog = false;
		// int n_pt_sources = 1024;
		// int n_source_batches = 1;
	#endif


	/***************************************************
	Parse Command Line Options
	***************************************************/
	
	#if DEBUG	
		char legal_commandline_options[] = "s:g:p:d:h";//{'g','f',':','d',':','h'};
	#else
		char legal_commandline_options[] = "c:k:g:p:d:h";//{'c',':','k',':','g',':','f',':','d',':','h'}; //
	#endif

	int arg = 0;
	int gpu = 0;

	#ifndef DEBUG
		/* DADA VARIABLES */
		int core = 0;
		key_t in_key = 0x0000dada;
	#endif

	char* file_name = (char *) calloc(256, sizeof(char));
	while ((arg=getopt(argc, argv, legal_commandline_options)) != -1) {
		switch (arg) {
			#ifndef DEBUG
				case 'c':
					/* to bind to a cpu core */
					if (optarg){
						core = atoi(optarg);
						break;
					} else {
						printf ("ERROR: -c flag requires argument\n");
						return EXIT_FAILURE;
					}
					
				case 'k':
					/* to set the dada key */
					if (sscanf (optarg, "%x", &in_key) != 1) {
						fprintf (stderr, "beam: could not parse key from %s\n", optarg);
						return EXIT_FAILURE;
					}
					break;
			#else
				case 's':
					/* To setup source directions */
					if (sscanf (optarg, "%s", file_name) != 1) {
						fprintf (stderr, "beam: could not parse source direction file from %s\n", optarg);
						return EXIT_FAILURE;
					}
					input_data_generator.read_in_source_directions(file_name);


					// sources = read_in_source_directions(file_name, &n_pt_sources);
					// n_source_batches = CEILING(n_pt_sources, N_SOURCES_PER_BATCH);
					// use_source_catalog = true;
					break;
			#endif

			case 'g':
				/* to select a predefined frequency range */
				if (optarg){
					gpu = atoi(optarg);
					break;
				} else {
					printf ("ERROR: -g flag requires argument\n");
					return EXIT_FAILURE;
				}
			
			case 'p':
				/* To setup antenna position locations */
				
				if (sscanf (optarg, "%s", file_name) != 1) {
					fprintf (stderr, "beam: could not parse antenna location file from %s\n", optarg);
					return EXIT_FAILURE;
				}
				read_in_position_locations(file_name, pos);
				pos_set = true;
				break;

			case 'd':
				/* To setup beamforming directions */
				if (sscanf (optarg, "%s", file_name) != 1) {
					fprintf (stderr, "beam: could not parse direction file from %s\n", optarg);
					return EXIT_FAILURE;
				}
				read_in_beam_directions(file_name, N_BEAMS, dir);
				dir_set = true;
				break;

			case 'h':
				usage();
				return EXIT_SUCCESS;
		}
	}
	free(file_name);

	#ifndef DEBUG
		char name[] = "beam";
		dada_handler dada_handle(name, core, in_key);
	#endif

	if (!pos_set){
		/* Populate location/direction Matricies if they were not set by command-line arguments */
		for (int i = 0; i < N_ANTENNAS; i++){
			pos[i].x = i*500.0/(N_ANTENNAS-1) - 250.0;
		}
	}

	if (!dir_set){
		/* Directions for Beamforming if they were not set by command-line arguments */
		for (int i = 0; i < N_BEAMS; i++){
			dir[i].theta = i*DEG2RAD(2*HALF_FOV)/(N_BEAMS-1) - DEG2RAD(HALF_FOV);
		}
	}


	#if VERBOSE
		/* Print Information about all defined variables */
		print_all_defines();
	#endif

	/*Look for and select desired GPU type (if it exists) */
	char prefered_device_name[] = "GeForce GTX 1080";
	CUDA_select_GPU(prefered_device_name);

	/***********************************
	 *			GPU Variables		   *
	 ***********************************/

	/* CUBLAS matrix dimensions */
	int fourier_coefficients_rows	 = N_BEAMS;
	int fourier_coefficients_cols 	 = N_ANTENNAS;
	int fourier_coefficients_stride  = fourier_coefficients_rows*fourier_coefficients_cols;
	int B_cols	 = N_TIMESTEPS_PER_GEMM;
	int B_rows	 = fourier_coefficients_cols;
	int B_stride = B_rows*B_cols;
	int C_rows	 = fourier_coefficients_rows;
	int C_cols	 = B_cols;
	int C_stride = C_rows*C_cols;
	float bw_per_channel = BW_PER_CHANNEL; 

	CxInt8_t *d_fourier_coefficients; 				// Weight matrix (N_BEAMS X N_ANTENNAS, for N_FREQUENCIES)
	CxInt8_t *d_B; 				// Data Matrix (N_ANTENNAS X N_TIMESTEPS_PER_GEMM, for N_FREQUENCIES)
	char *d_data;				// Raw input data (Before data massaging)
	cuComplex *d_C;				// Beamformed output (N_BEAMS X N_TIMESTEPS_PER_GEMM, for N_FREQUENCIES)
	float *d_out;				// Data after being averaged over 16 time samples and 2 polarizations
	float *beam_out; 			// Final Data product location
	
	#if DEBUG
		float *d_vec_ones;		// Vectors of all ones for de-dispersion
		float *d_dedispersed;	// Data after being de-dispersed
	#endif

	/* CUBLAS Constants (and host variables for transfering to GPU) */
	cuComplex  h_inv_max_value,  h_zero; 		// Host Values
	cuComplex *d_inv_max_value, *d_zero; 		// Device Values

	h_inv_max_value.x = 1.0/MAX_VAL;
	h_inv_max_value.y = 0;
	h_zero.x = 0;
	h_zero.y = 0;

	#if DEBUG
		float  h_f_one,  h_f_zero;				// Host Values
		float *d_f_one, *d_f_zero;				// Device Values	
		h_f_one = 1.0;
		h_f_zero = 0.0;
	#endif

	/***********************************
	 *			HOST Variables		   *
	 ***********************************/
	#if DEBUG
		// char *data; 									// Input data for test cases need pinned memory
		float *dedispersed_out;							// Ouput data for dedispersed data (DM = 0)
		float *vec_ones = new float[N_FREQUENCIES]; 	// A vector of all ones for dedispersion (frequency averaging)

		// gpuErrchk(cudaHostAlloc( (void**) &data, INPUT_DATA_SIZE*sizeof(char), 0));
		gpuErrchk(cudaHostAlloc( (void**) &dedispersed_out, N_BEAMS*input_data_generator.get_n_pt_sources()*sizeof(float), 0));

		// if (!use_source_catalog){
		// 	/* set memory to zero if a source catalog wasn't provided */
		// 	memset(data, BOGUS_DATA, INPUT_DATA_SIZE*sizeof(char)); // Generates Bogus data, typically 0x70
		// 	std::cout << "Using BOGUS DATA " << std::endl;
		// }

		for (int i = 0; i < N_FREQUENCIES; i++){
			/* Set data for a vector of all ones */
			vec_ones[i] = 1.0;
		}
	#endif


	/***********************************
	 *		Fourier Coefficients 	   *
	 ***********************************/
	CxInt8_t *fourier_coefficients = new CxInt8_t[fourier_coefficients_cols*fourier_coefficients_rows*N_FREQUENCIES];

	for (int i = 0; i < N_FREQUENCIES; i++){
		float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*bw_per_channel;
		float wavelength = C_SPEED/(1E9*freq);
		for (int j = 0; j < N_ANTENNAS; j++){
			for (int k = 0; k < N_BEAMS; k++){
				fourier_coefficients[i*fourier_coefficients_stride + j*N_BEAMS + k].x = round(MAX_VAL*cos(-2*PI*(pos[j].x*sin(dir[k].theta) + pos[j].y*sin(dir[k].phi))/wavelength));
				fourier_coefficients[i*fourier_coefficients_stride + j*N_BEAMS + k].y = round(MAX_VAL*sin(-2*PI*(pos[j].x*sin(dir[k].theta) + pos[j].y*sin(dir[k].phi))/wavelength));
			}
		}
	}


	/***********************************
	 *			Memory Allocation 	   *
	 ***********************************/
	

	gpuErrchk(cudaHostAlloc(&beam_out, N_F_PER_DETECT*N_STREAMS*sizeof(float), 0));

	gpuErrchk(cudaMalloc(&d_fourier_coefficients, 		fourier_coefficients_rows*fourier_coefficients_cols*N_FREQUENCIES*sizeof(CxInt8_t)));
	gpuErrchk(cudaMalloc(&d_B, 		N_CX_IN_PER_GEMM*N_STREAMS*sizeof(CxInt8_t)));
	gpuErrchk(cudaMalloc(&d_C, 		N_CX_OUT_PER_GEMM*N_STREAMS*sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_data, 	N_BYTES_PRE_EXPANSION_PER_BLOCK*N_BLOCKS_ON_GPU)); 							// array for raw data
	gpuErrchk(cudaMalloc(&d_out, 	N_F_PER_DETECT*N_STREAMS * sizeof(float)));					// array for detected, averaged data

	/* Cublas Constants */
	gpuErrchk(cudaMalloc(&d_inv_max_value, sizeof(cuComplex)));
	gpuErrchk(cudaMalloc(&d_zero, sizeof(cuComplex)));
	
	#if DEBUG
		gpuErrchk(cudaMalloc(&d_vec_ones, 	 N_FREQUENCIES*sizeof(float)));
		gpuErrchk(cudaMalloc(&d_f_one, 		 sizeof(float)));
		gpuErrchk(cudaMalloc(&d_f_zero, 	 sizeof(float)));
		gpuErrchk(cudaMalloc(&d_dedispersed, N_BEAMS*N_STREAMS*sizeof(float)));						// array for frequency averaged data
	#endif

	/***********************************
	 *			Memory Copies	 	   *
	 ***********************************/

	gpuErrchk(cudaMemcpy(d_fourier_coefficients, fourier_coefficients, fourier_coefficients_rows*fourier_coefficients_cols*N_FREQUENCIES*sizeof(CxInt8_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_inv_max_value, &h_inv_max_value, sizeof(cuComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_zero, &h_zero, sizeof(cuComplex), cudaMemcpyHostToDevice));

	#if DEBUG
		gpuErrchk(cudaMemcpy(d_vec_ones, vec_ones, N_FREQUENCIES*sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_f_one, &h_f_one, sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_f_zero, &h_f_zero, sizeof(float), cudaMemcpyHostToDevice));
		std::cout << "Host-side zero: " << h_f_zero << " and host-side one: " << h_f_one << std::endl;
		std::cout << "GPU-side one & zero:" << std::endl;
		print_data_scalar<<<1, 1>>>(d_f_one);
		print_data_scalar<<<1, 1>>>(d_f_zero);
	#endif

	/***********************************
	 *  Memory Initialization (memset) *
	 ***********************************/

	/* Zero out GPU memory with cudaMemset (this is helpful when the number of antennas != N_ANTENNAS) */
	gpuErrchk(cudaMemset(d_B, 0,  	N_CX_IN_PER_GEMM*N_STREAMS*sizeof(CxInt8_t)));
	gpuErrchk(cudaMemset(d_C, 0,  	N_CX_OUT_PER_GEMM*N_STREAMS*sizeof(cuComplex)));
	gpuErrchk(cudaMemset(d_data, 0, N_BYTES_PRE_EXPANSION_PER_BLOCK*N_BLOCKS_ON_GPU)); 			// array for raw data
	gpuErrchk(cudaMemset(d_out, 0,  N_F_PER_DETECT*N_STREAMS * sizeof(float)));					// array for detected, averaged data

	#if DEBUG
		gpuErrchk(cudaMemset(d_dedispersed, 0, N_BEAMS*N_STREAMS*sizeof(float)));
	#endif

	/***********************************
	 *		Concurrency Handles		   *
	 ***********************************/


	cudaStream_t HtoDstream;
	cudaStream_t stream[N_STREAMS];
	cublasHandle_t handle[N_STREAMS];
	int timeSlice[N_STREAMS];

	gpuErrchk(cudaStreamCreate(&HtoDstream));

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamCreate(&stream[i]));
		gpuBLASchk(cublasCreate(&handle[i]));
		gpuBLASchk(cublasSetStream(handle[i], stream[i]));
		gpuBLASchk(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_DEVICE)); //All constants reside on GPU
		gpuBLASchk(cublasSetMathMode(handle[i], CUBLAS_TENSOR_OP_MATH )); //Enable use of tensor cores

		timeSlice[i] = i;
	}

	observation_loop_state obs_state(MAX_TRANSFER_SEP, MAX_TOTAL_SEP);

	#if DEBUG
		obs_state.set_n_pt_sources(input_data_generator.get_n_pt_sources());
	#endif

	/***************************************************
	Initialize hdu (FOR DADA)
	***************************************************/

	#ifndef DEBUG
		char *block;
		dada_handle.read_headers();
	#endif


	/*********************************************************************************
	START OBSERVATION LOOP
	*********************************************************************************/

	#if VERBOSE
		std::cout << "Executing beamformer.cu" << "\n";
		std::cout << "MAX_TOTAL_SEP: "<< MAX_TOTAL_SEP << "\n";
		std::cout << "MAX_TRANSFER_SEP: "<< MAX_TRANSFER_SEP << std::endl;
	#endif

	#if BURNIN && !DEBUG
		std::cout << "Burning IN" << std::endl;
		for (int i = 0; i < BURNIN; i++){
			dada_handle.read();
			dada_handle.close();
		}
		std::cout << "Done Burn in" << std::endl;
	#endif

	#if DEBUG || VERBOSE
		START_TIMER();
		float time_accumulator_ms = 0;
		float observation_time_ms = 0;
	#endif


	while (!obs_state.check_observations_complete()){
		
		#if VERBOSE
			/* Header to be printed during every loop */
			std::cout << "##########################################" << std::endl;
			std::cout << obs_state << std::endl;
		#endif 


		/**************************************************
						Copy Data to GPU
		**************************************************/

		/* Data is copied iff the analysis steps and transfer rates are keeping up and there is still data */
		if (obs_state.check_ready_for_transfer()){
			#ifndef DEBUG
				#if VERBOSE
					std::cout << "READING FROM PSRDADA" << std::endl;
				#endif
				
				block = dada_handle.read();

				if (!dada_handle.check_transfers_complete()){
					
					/* Copy Block */
					gpuErrchk(cudaMemcpyAsync(&d_data[N_BYTES_PRE_EXPANSION_PER_BLOCK * obs_state.get_next_gpu_transfer_block()],//(blocks_transfer_queue % N_BLOCKS_ON_GPU)], 
												block,
												N_BYTES_PRE_EXPANSION_PER_BLOCK, 
												cudaMemcpyHostToDevice,
												HtoDstream));

					/* Generate Cuda event which will indicate when the block has been transfered*/
					obs_state.generate_transfer_event(HtoDstream);
				} else {
					obs_state.set_transfers_complete(true);
				}

				dada_handle.close();

			#else
				// if (use_source_catalog && (obs_state.get_blocks_transferred() == (source_batch_counter * N_SOURCES_PER_BATCH) / N_GEMMS_PER_BLOCK) ){
				if(input_data_generator.check_need_to_generate_more_input_data(obs_state.get_blocks_transferred())){
					//Generates the dummy data given a set of directions.
					std::cout << "Generating new source data" << std::endl;
					STOP_RECORD_TIMER(time_accumulator_ms);
					observation_time_ms += time_accumulator_ms;
					input_data_generator.generate_test_data(pos, gpu);
					// generate_test_data(data, sources, n_pt_sources, pos, gpu, B_stride, source_batch_counter);
					START_TIMER();
					// source_batch_counter ++;
					// if (source_batch_counter == n_source_batches){
					// 	std::cout << "Program should be over soon" << std::endl;
					// }
					std::cout << "done generating test data" << std::endl;
				}

				// if (obs_state.get_blocks_transfer_queue() < (source_batch_counter * N_SOURCES_PER_BATCH) / N_GEMMS_PER_BLOCK) {
				if(input_data_generator.check_data_ready_for_transfer(obs_state.get_blocks_transfer_queue())){
					/* Only initiate transfers if there is valid data in data[] */

					char* input_data = input_data_generator.get_data();
					gpuErrchk(cudaMemcpyAsync(&d_data[N_BYTES_PRE_EXPANSION_PER_BLOCK * obs_state.get_next_gpu_transfer_block()], //(blocks_transfer_queue % N_BLOCKS_ON_GPU)], 
												&input_data[(N_BYTES_PRE_EXPANSION_PER_BLOCK * obs_state.get_blocks_transfer_queue()) % INPUT_DATA_SIZE],
												N_BYTES_PRE_EXPANSION_PER_BLOCK, 
												cudaMemcpyHostToDevice,
												HtoDstream));

					obs_state.generate_transfer_event(HtoDstream);
				}


				/***********************************
				 Check if transfers are done			   
				 ***********************************/
				obs_state.check_transfers_complete();

			#endif
		}

		/**************************************************
				Check if data has been transfered
		**************************************************/
		obs_state.check_transfer_events();

		/**************************************************
					Initiate Beamforming
		**************************************************/
		// if (blocks_analysis_queue < blocks_transferred){
		if(obs_state.check_ready_for_analysis()){

			for (int part = 0; part < N_GEMMS_PER_BLOCK/N_STREAMS; part++){
				/* Call routine once per each GEMM in a BLOCK */

				#if VERBOSE
					std::cout << "Queueing Beamforming. Start Dir = " << obs_state.get_current_analysis_gemm(timeSlice[0]) << std::endl; 
				#endif

				for (int st = 0; st < N_STREAMS; st++){

					/* Expand input from 4-bit integers to 8-bit integers */
					expand_input<<<10000, 32, 0, stream[st]>>>(&d_data[N_BYTES_PRE_EXPANSION_PER_GEMM*(N_GEMMS_PER_BLOCK*obs_state.get_next_gpu_analysis_block() + timeSlice[st])],
														      (char *) &d_B[N_CX_IN_PER_GEMM*st], 
														      B_stride*N_FREQUENCIES);


					/* Execute Beamforming Matrix Multiplication */
					gpuBLASchk(cublasGemmStridedBatchedEx(handle[st], CUBLAS_OP_N, CUBLAS_OP_N,
												fourier_coefficients_rows, B_cols, fourier_coefficients_cols,
												d_inv_max_value,
												d_fourier_coefficients, CUDA_C_8I, fourier_coefficients_rows, fourier_coefficients_stride,
												&d_B[N_CX_IN_PER_GEMM*st], CUDA_C_8I, B_rows, B_stride,
												d_zero,
												&d_C[N_CX_OUT_PER_GEMM*st], CUDA_C_32F, C_rows, C_stride,
												N_FREQUENCIES, CUDA_C_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));


					/* Square results and average */
					detect_sum<<<detect_dimGrid, detect_dimBlock, 0, stream[st]>>>(&d_C[N_CX_OUT_PER_GEMM*st], N_INPUTS_PER_OUTPUT, &d_out[st*N_F_PER_DETECT]);


					/* Copy Data back to CPU RAM */
					gpuErrchk(cudaMemcpyAsync(&beam_out[st * N_F_PER_DETECT], 
											  &d_out[st * N_F_PER_DETECT], N_OUTPUTS_PER_GEMM * N_FREQUENCIES * N_BEAMS * sizeof(float), 
											  cudaMemcpyDeviceToHost,
											  stream[st]));

					#if DEBUG
						
						if (obs_state.check_ready_for_dh2_transfer(timeSlice[st])){ // no need to copy more than the number of sources.
							int current_gemm = obs_state.get_current_analysis_gemm(timeSlice[st]);

							std::cout << "Current GEMM: " << current_gemm << std::endl;

							/* Sum over all 256 frequencies with a matrix-vector multiplication. */
							gpuBLASchk(cublasSgemv(handle[st], CUBLAS_OP_N,
										N_BEAMS, N_FREQUENCIES,
										d_f_one,
										&d_out[st*N_F_PER_DETECT], N_BEAMS,
										d_vec_ones, 1,
										d_f_zero,
										&d_dedispersed[st*N_BEAMS], 1));

							/* Copy Frequency-averaged data back to CPU */
							gpuErrchk(cudaMemcpyAsync(&dedispersed_out[current_gemm * N_BEAMS], 
													  &d_dedispersed[st * N_BEAMS], N_BEAMS * sizeof(float), 
													  cudaMemcpyDeviceToHost,
													  stream[st]));
						}
					#endif

					
					timeSlice[st] += N_STREAMS; // increment so the next time slice is processed next

					if (timeSlice[st] >= N_GEMMS_PER_BLOCK){
						timeSlice[st] -= N_GEMMS_PER_BLOCK; //wrap back to the beginning once each gemm in a block has been processed
					}

				}
			}
			// gpuErrchk(cudaEventRecord(BlockAnalyzedSync[blocks_analysis_queue % (N_EVENTS_ON_GPU)], stream[N_STREAMS-1]));
			// blocks_analysis_queue ++;
			obs_state.generate_analysis_event(stream[N_STREAMS-1]);
		}


		/**************************************************
			Check if beamforming analysis has completed
		**************************************************/
		obs_state.check_analysis_events();

	} // end while (!observation_complete)




	#if DEBUG
		STOP_RECORD_TIMER(time_accumulator_ms);
		observation_time_ms += time_accumulator_ms;
		std::cout << "Observation ran in " << observation_time_ms << "milliseconds.\n";

		std::cout << "Code produced outputs for " << input_data_generator.get_n_pt_sources()*N_OUTPUTS_PER_GEMM << " data chunks.\n";
		std::cout << "Time per data chunk: " << observation_time_ms/(input_data_generator.get_n_pt_sources()*N_OUTPUTS_PER_GEMM) << " milliseconds.\n";
		std::cout << "Approximate datarate: " << N_BYTES_PRE_EXPANSION_PER_GEMM*input_data_generator.get_n_pt_sources()/observation_time_ms/1e6 << "GB/s" << std::endl;
	#else
		#if VERBOSE
			STOP_RECORD_TIMER(time_accumulator_ms);
			observation_time_ms += time_accumulator_ms;
			std::cout << "Observation ran in " << observation_time_ms << "milliseconds.\n";
			std::cout << "Code produced outputs for " << obs_state.get_current_transfer_gemm()*N_OUTPUTS_PER_GEMM << " data chunks.\n";
			std::cout << "Time per data chunk: " << observation_time_ms/(obs_state.get_current_transfer_gemm()*N_OUTPUTS_PER_GEMM) << " milliseconds.\n";
			std::cout << "Approximate datarate: " << dada_handle.get_block_size()*obs_state.get_blocks_transfer_queue()/observation_time_ms/1e6 << "GB/s" << std::endl;

		#endif
	#endif


	for (int st = 0; st < N_STREAMS; st++){
		gpuErrchk(cudaStreamSynchronize(stream[st]));
	}
	std::cout << "Synchronized" << std::endl;




	#if DEBUG
		char filename[] = "bin/data.py";
		write_array_to_disk_as_python_file(dedispersed_out, input_data_generator.get_n_pt_sources(), N_BEAMS, filename);
	#endif

	std::cout << "Freeing CUDA Structures" << std::endl;

	// for (int event = 0; event < N_EVENTS_ON_GPU; event++){
	// 	gpuErrchk(cudaEventDestroy(BlockAnalyzedSync[event]));
	// 	gpuErrchk(cudaEventDestroy(BlockTransferredSync[event]));
	// }

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamDestroy(stream[i]));
		gpuBLASchk(cublasDestroy(handle[i]));
	}

	std::cout << "Freed cuda streams and handles" << std::endl;

	gpuErrchk(cudaFree(d_fourier_coefficients));
	gpuErrchk(cudaFree(d_B));
	gpuErrchk(cudaFree(d_C));
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


	std::cout << "Freed GPU memory" << std::endl;

	gpuErrchk(cudaFreeHost(beam_out));

	delete[] fourier_coefficients;
	delete[] pos;
	delete[] dir;

	#if DEBUG
		// gpuErrchk(cudaFreeHost(data));
		gpuErrchk(cudaFreeHost(dedispersed_out));

		delete[] vec_ones;
	#endif

	std::cout << "Freed CPU memory" << std::endl;

	return 0;
}







