class observation_loop_state{
private:
	uint64_t blocks_analyzed = 0;
	uint64_t blocks_transferred = 0;
	uint64_t blocks_analysis_queue = 0;
	uint64_t blocks_transfer_queue = 0;

	uint64_t maximum_transfer_seperation;
	uint64_t maximum_total_seperation;

	bool transfers_complete = false;

	cudaEvent_t BlockTransferredSync[N_EVENTS_ON_GPU];
	cudaEvent_t BlockAnalyzedSync[N_EVENTS_ON_GPU];

	int most_recent_gemm = 0;
	int n_pt_sources = 0;

public:
	observation_loop_state(uint64_t maximum_transfer_seperation, uint64_t maximum_total_seperation);
	~observation_loop_state();
	
	void generate_transfer_event(cudaStream_t HtoDstream);
	void generate_analysis_event(cudaStream_t stream);
	void check_transfer_events();
	void check_analysis_events();

	uint64_t get_blocks_analyzed() const {return blocks_analyzed;}
	uint64_t get_blocks_transferred() const {return blocks_transferred;}
	uint64_t get_blocks_analysis_queue() const {return blocks_analysis_queue;}
	uint64_t get_blocks_transfer_queue() const {return blocks_transfer_queue;}

	uint64_t get_current_analysis_gemm(int time_slice);
	uint64_t get_current_transfer_gemm() const;
	uint64_t get_next_gpu_analysis_block() const {return blocks_analysis_queue%N_BLOCKS_ON_GPU;}
	uint64_t get_next_gpu_transfer_block() const {return blocks_transfer_queue%N_BLOCKS_ON_GPU;}

	void set_transfers_complete(bool value) {transfers_complete = value;}
	bool get_transfers_complete() const {return transfers_complete;}

	bool check_ready_for_transfer() const;
	bool check_ready_for_analysis() const;
	bool check_ready_for_dh2_transfer(int time_slice);

	bool check_observations_complete();
	#if DEBUG
		void set_n_pt_sources(int val){n_pt_sources = val;}
		bool check_transfers_complete();
	#endif

	friend std::ostream & operator << (std::ostream &out, const observation_loop_state &a);
};

observation_loop_state::observation_loop_state(uint64_t maximum_transfer_seperation, uint64_t maximum_total_seperation) {
	this->maximum_transfer_seperation = maximum_transfer_seperation;
	this->maximum_total_seperation = maximum_total_seperation;

	for (int i = 0; i < N_EVENTS_ON_GPU; i++) {
		gpuErrchk(cudaEventCreateWithFlags(&BlockTransferredSync[i], cudaEventDisableTiming));
		gpuErrchk(cudaEventCreateWithFlags(&BlockAnalyzedSync[i],    cudaEventDisableTiming));
	}
}

observation_loop_state::~observation_loop_state(){
	for (int event = 0; event < N_EVENTS_ON_GPU; event++){
		gpuErrchk(cudaEventDestroy(BlockAnalyzedSync[event]));
		gpuErrchk(cudaEventDestroy(BlockTransferredSync[event]));
	}
}

void observation_loop_state::generate_transfer_event(cudaStream_t HtoDstream) {
	/* Generate Cuda event which will indicate when the block has been transfered*/
	gpuErrchk(cudaEventRecord(BlockTransferredSync[blocks_transfer_queue % (N_EVENTS_ON_GPU)], HtoDstream));
	blocks_transfer_queue++;
}

void observation_loop_state::generate_analysis_event(cudaStream_t stream) {
	/* Generate Cuda event which will indicate when the block has been analyzed*/
	gpuErrchk(cudaEventRecord(BlockAnalyzedSync[blocks_analysis_queue % (N_EVENTS_ON_GPU)], stream));
	blocks_analysis_queue ++;
}

void observation_loop_state::check_transfer_events() {
	/* Iterate through all left-over blocks and see if they've been finished */
	for (uint64_t event = blocks_transferred; event < blocks_transfer_queue; event ++) {
		if(cudaEventQuery(BlockTransferredSync[event % (N_EVENTS_ON_GPU)]) == cudaSuccess) {
		
			#if VERBOSE
				std::cout << "Block " << event << " transfered to GPU" << std::endl;
			#endif

			blocks_transferred ++;

			/* Destroy and Recreate Flags */
			gpuErrchk(cudaEventDestroy(BlockTransferredSync[event % (N_EVENTS_ON_GPU)]));
			gpuErrchk(cudaEventCreateWithFlags(&BlockTransferredSync[event % (N_EVENTS_ON_GPU)], cudaEventDisableTiming));
		} else {
			break; // dont need to check later blocks if current block has not finished
		}
	}
}

void observation_loop_state::check_analysis_events() {
	for (uint64_t event = blocks_analyzed; event < blocks_analysis_queue; event ++) {
		if(cudaEventQuery(BlockAnalyzedSync[event % (N_EVENTS_ON_GPU)]) == cudaSuccess) {

			//This is incremented once each time slice in each block is analyzed (or more accurately, scheduled)
			blocks_analyzed++;
			#if VERBOSE
				std::cout << "Block " << event << " Analyzed" << std::endl;
			#endif
			gpuErrchk(cudaEventDestroy(BlockAnalyzedSync[event % (N_EVENTS_ON_GPU)]));
			gpuErrchk(cudaEventCreateWithFlags(&BlockAnalyzedSync[event % (N_EVENTS_ON_GPU)], cudaEventDisableTiming));
			
		} else {
			break; // If previous analyzed blocks have not been finished, there's no reason to check the next blocks
		}
	}
}

uint64_t observation_loop_state::get_current_analysis_gemm(int time_slice) {
	most_recent_gemm = blocks_analysis_queue * N_GEMMS_PER_BLOCK + time_slice;
	return most_recent_gemm;
}

uint64_t observation_loop_state::get_current_transfer_gemm() const{
	return blocks_transfer_queue * N_GEMMS_PER_BLOCK;
}

bool observation_loop_state::check_ready_for_transfer() const {
	return ( (blocks_transfer_queue - blocks_analyzed < maximum_total_seperation)
			 && (blocks_transfer_queue - blocks_transferred < maximum_transfer_seperation)
			 && !transfers_complete );
}

bool observation_loop_state::check_ready_for_dh2_transfer(int time_slice) {
	int current_gemm = get_current_analysis_gemm(time_slice);
	return (current_gemm < n_pt_sources);
}

bool observation_loop_state::check_ready_for_analysis() const {
	return (blocks_analysis_queue < blocks_transferred);
}

bool observation_loop_state::check_observations_complete() {
#if DEBUG
	if ((most_recent_gemm >= n_pt_sources-1) && (blocks_analyzed == blocks_transfer_queue) && transfers_complete) {
		std::cout << "obs Complete" << std::endl;
		return true;
	}
	return false;
#else
	if ((blocks_analyzed == blocks_transfer_queue) && transfers_complete) {
		std::cout << "obs Complete" << std::endl;
		return true;
	}
	return false;
#endif
}

#if DEBUG
bool observation_loop_state::check_transfers_complete() {
	if (blocks_transfer_queue * N_GEMMS_PER_BLOCK >= n_pt_sources) {
		 /* If the amount of data queued for transfer is greater than the amount needed for analyzing N_PT_SOURCES, stop */
		transfers_complete = 1;
		return true;
	}
	return false;
}
#endif

std::ostream & operator << (std::ostream &out, const observation_loop_state &a) {
	return out << "A: " << a.blocks_analyzed <<  ", AQ: " << a.blocks_analysis_queue << ", T: " << a.blocks_transferred 
			<< ", TQ: " << a.blocks_transfer_queue << "\n" << "current_gemm: " << a.most_recent_gemm
			<< ", transfers_complete: " << a.transfers_complete;
}

