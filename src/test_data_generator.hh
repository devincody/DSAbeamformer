#define BOGUS_DATA 0x70


class test_data_generator{
	private:
		int n_pt_sources = 1024;
		int n_source_batches = 1;
		int source_batch_counter = 0;
		beam_direction* sources;
		bool use_source_catalog = false;
		char *data;

	public:
		
		test_data_generator();
		~test_data_generator();

		int get_n_pt_sources() const {return n_pt_sources;}
		char * get_data() const {return data;}

		void generate_test_data(antenna pos[], int gpu);
		void read_in_source_directions(char * file_name);
		bool check_need_to_generate_more_input_data(int blocks_transfered);
		bool check_data_ready_for_transfer(int blocks_transfer_queue);
};

test_data_generator::test_data_generator(){
	gpuErrchk(cudaHostAlloc( (void**) &data, INPUT_DATA_SIZE*sizeof(char), 0));
	memset(data, BOGUS_DATA, INPUT_DATA_SIZE*sizeof(char)); // Generates Bogus data, typically 0x70
}

test_data_generator::~test_data_generator(){
	gpuErrchk(cudaFreeHost(data));
}

void test_data_generator::read_in_source_directions(char * file_name){
	if (!use_source_catalog){
		std::ifstream input_file;

		input_file.open(file_name);

		input_file >> n_pt_sources;

		sources = new beam_direction[n_pt_sources]();  // Array to hold direction of the test sources

		for (int beam_idx = 0; beam_idx < n_pt_sources; beam_idx++){
			input_file >> sources[beam_idx];
		}
		use_source_catalog = true;
		n_source_batches = CEILING(n_pt_sources, N_SOURCES_PER_BATCH);

		#if VERBOSE
		std::cout << "Read in " << n_pt_sources << " source directions" << std::endl;
		#endif
	}
}


void test_data_generator::generate_test_data(antenna pos[], int gpu){
	// float test_direction;
	char high, low;

	for (long direction = 0; direction < N_SOURCES_PER_BATCH; direction++){
		//test_direction = DEG2RAD(-HALF_FOV) + ((float) direction)*DEG2RAD(2*HALF_FOV)/(N_PT_SOURCES-1);

		for (int i = 0; i < N_FREQUENCIES; i++){
			float freq = END_F - (ZERO_PT + gpu*TOT_CHANNELS/(N_GPUS-1) + i)*BW_PER_CHANNEL;
			// std::cout << "freq: " << freq << std::endl;
			float wavelength = C_SPEED/(1E9*freq);
			for (int j = 0; j < N_TIMESTEPS_PER_GEMM; j++){
				for (int k = 0; k < N_ANTENNAS; k++){
					int source_look_up = direction + source_batch_counter*N_SOURCES_PER_BATCH;

					if (source_look_up < n_pt_sources){
						high = ((char) round(SIG_MAX_VAL*cos(2 * PI * (pos[k].x * sin(sources[source_look_up].theta) + pos[k].y * sin(sources[source_look_up].phi)) / wavelength))); //real
						low  = ((char) round(SIG_MAX_VAL*sin(2 * PI * (pos[k].x * sin(sources[source_look_up].theta) + pos[k].y * sin(sources[source_look_up].phi)) / wavelength))); //imag

						data[direction * N_BYTES_PRE_EXPANSION_PER_GEMM + i * N_TIMESTEPS_PER_GEMM * N_ANTENNAS + j * N_ANTENNAS + k] = (high << 4) | (0x0F & low);
					} else {
						data[direction * N_BYTES_PRE_EXPANSION_PER_GEMM + i * N_TIMESTEPS_PER_GEMM * N_ANTENNAS + j * N_ANTENNAS + k] = 0;
					}
				}
			}
		}
	}



	source_batch_counter ++;
}


bool test_data_generator::check_need_to_generate_more_input_data(int blocks_transfered){
	return (use_source_catalog && (blocks_transfered == (source_batch_counter * N_SOURCES_PER_BATCH) / N_GEMMS_PER_BLOCK) );
}

bool test_data_generator::check_data_ready_for_transfer(int blocks_transfer_queue){
	return (blocks_transfer_queue < (source_batch_counter * N_SOURCES_PER_BATCH) / N_GEMMS_PER_BLOCK);
}
