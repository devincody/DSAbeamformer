/* beamformer.cuh

	Contains all functions and defines related to CUDA

*/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h> // needed for printf() which can be called from GPU




/***********************************
 *			Error Checking		   *
 ***********************************/

/* Helper Macro for gpu error checking */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* Function for gpu error checking */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

/* Error checking for cuBLAS */
void gpuBLASchk(int errval) {
	if (errval != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Failed BLAS call, error code " << errval << std::endl;
	}
}

/***********************************
 *			Timing Functions	   *
 ***********************************/

/* Variables and Functions for timing analysis */
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








class observation_loop_state{
private:
	uint64_t blocks_analyzed = 0;
	uint64_t blocks_transferred = 0;
	uint64_t blocks_analysis_queue = 0;
	uint64_t blocks_transfer_queue = 0;
	// #if DEBUG
	// 	int source_batch_counter = 0;
	// #endif

	uint64_t maximum_transfer_seperation;
	uint64_t maximum_total_seperation;

	bool observation_complete = false;
	bool transfers_complete = false;

	cudaEvent_t BlockTransferredSync[N_EVENTS_ON_GPU];
	cudaEvent_t BlockAnalyzedSync[N_EVENTS_ON_GPU];

	int most_recent_gemm = 0;

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
	bool get_observation_complete() const {return observation_complete;}
	bool get_transfers_complete() const {return transfers_complete;}
	

	bool check_ready_for_transfer() const;
	bool check_ready_for_analysis() const;

	bool check_observations_complete();
	#if DEBUG
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

bool observation_loop_state::check_ready_for_analysis() const {
	return (blocks_analysis_queue < blocks_transferred);
}

bool observation_loop_state::check_observations_complete() {
#if DEBUG
	if ((most_recent_gemm >= N_PT_SOURCES-1) && (blocks_analyzed == blocks_transfer_queue) && transfers_complete) {
		observation_complete = true;
		std::cout << "obs Complete" << std::endl;
		return true;
	}
	return false;
#else
	if ((blocks_analyzed == blocks_transfer_queue) && transfers_complete) {
		observation_complete = true;
		std::cout << "obs Complete" << std::endl;
		return true;
	}
	return false;
#endif
}

#if DEBUG
bool observation_loop_state::check_transfers_complete() {
	if (blocks_transfer_queue * N_GEMMS_PER_BLOCK >= N_PT_SOURCES) {
		 /* If the amount of data queued for transfer is greater than the amount needed for analyzing N_PT_SOURCES, stop */
		transfers_complete = 1;
		return true;
	}
	return false;
}
#endif

std::ostream & operator << (std::ostream &out, const observation_loop_state &a) {
	return out << "A: " << a.blocks_analyzed <<  ", AQ: " << a.blocks_analysis_queue << ", T: " << a.blocks_transferred << ", TQ: " << a.blocks_transfer_queue;
}










/***********************************
 *			Kernels				   *
 ***********************************/


__global__
void expand_input(char const * __restrict__ input, char *output, int input_size) {
	/*
	This code takes in an array of 4-bit integers and returns an array of 8-bit integers.
	To maximize global memory bandwidth and symplicity, two special char types are 
	defined: char4_t and char8_t. The size of these types are 32-bit and 64-bit respectively. These
	enable coalesced memory accesses, but then require the gpu to handle the 4-bit to 8-bit
	conversion simultaneously for 8 numbers (4 real/imaginary pairs). 
	*/

	__shared__ float shmem_in[32];
	__shared__ double shmem_out[32];

	char4_t *char_shmem_in;
	cuChar4_t *char_shmem_out;

	char_shmem_in = reinterpret_cast<char4_t *>(shmem_in);
	char_shmem_out = reinterpret_cast<cuChar4_t *>(shmem_out);

	int local_idx = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (global_idx < input_size/sizeof(float)) {
		shmem_in[local_idx] = ((float *) input)[global_idx]; // read eight pieces of 4-bit memory into shared memory

		//#pragma unroll
		for (int i = 0; i < 4; i++) {
			char temp = char_shmem_in[local_idx][i];

			// break the char into two 4-bit chunks, then convert to 8-bit
			char high = (temp >> 4); // roll the 		most  significant 4 bits over the least significant 4 bits
			char low = (temp << 4);  // roll the 		least significant 4 bits over the most  significant 4 bits
			low = (low >> 4); 	 	 // roll the *new* 	most  significant 4 bits over the least significant 4 bits

			// store the two 8-bit numbers to the output shared memory array
			char_shmem_out[local_idx][i].x = high; 
			char_shmem_out[local_idx][i].y = low;
		}

		((double *) output)[global_idx] = shmem_out[local_idx];	// write eight pieces of 8-bit memory out to global memory

		global_idx += gridDim.x * blockDim.x;
	}
}


dim3 detect_dimGrid(N_OUTPUTS_PER_GEMM, N_FREQUENCIES, 1);
dim3 detect_dimBlock(N_BEAMS, 1,1);

__global__
void detect_sum(cuComplex const * __restrict__ input, int n_avg,  float * __restrict__ output) {
	/*
	Sum over the power in each N_INPUTS_PER_OUTPUT (32) time samples for each beam.
	number of threads = N_BEAMS (blockDim.x)
	number of blocks = N_OUTPUTS_PER_GEMM (blockIdx.x), N_FREQUENCIES (blockIdx.y)

	input indicies (slow to fast): freq, time batching, time, pol, beam, r/i
									256,            ~2,   16,   2,  256,   2

	out indicies (slow to fast):  time batching, freq, beam, r/i
								             ~2,  256,  256,   1

	*/

	const int n_beams = blockDim.x;	 	// should be = N_BEAMS
	const int n_freq = gridDim.y;   		// should be = N_FREQUENCIES
	const int n_batch = gridDim.x;		// should be = N_OUTPUTS_PER_GEMM = N_AVERAGING * N_POL

	const int batching_idx = blockIdx.x;	// index along batching dimension
	const int freq_idx = blockIdx.y;		// index along frequency dimension
	const int beam_idx = threadIdx.x;		// index along beam dimension

	__shared__ float shmem[N_BEAMS];// cannot use n_beams here since shmem needs a constant value
	shmem[beam_idx] = 0;			// initialize shared memory to zero

	const int input_idx  	 = freq_idx * n_batch * n_avg * n_beams 	// deal with frequency component first
								+ batching_idx * n_avg * n_beams  		// deal with the indexing for each output next
																		// Each thread starts on the first time-pol index 
																		// (b/c we iterate over all time-pol indicies in for loop below)
								+ beam_idx;							   	// start each thread on the first element of each beam

	const int output_idx = batching_idx * n_freq * n_beams + freq_idx * n_beams + beam_idx;

	#pragma unroll
	for (int i = input_idx; i < input_idx + n_avg*n_beams; i += n_beams) {
		shmem[beam_idx] += input[i].x*input[i].x + input[i].y*input[i].y;
	}

	output[output_idx] = shmem[beam_idx]; // slowest to fastest indicies: freq, beam
}

__global__
void print_data(float* data) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("data[%d] = %f\n", idx, data[idx]);
}

__global__
void print_data_scalar(float* data) {
	// int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("*data = %f\n", *data);
}



void CUDA_select_GPU(char * prefered_device_name) {
	/*

	 Code iterates through a list of available gpus provided by the system,
	 compares them to the given "prefered device name" variable and then 
	 selects the first one that matches.  
	
	*/
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
	    cudaDeviceProp deviceProperties;
	    cudaGetDeviceProperties(&deviceProperties, deviceIndex);

	    if (!strcmp(prefered_device_name, deviceProperties.name)) {
		    std::cout <<  "Selected: " << deviceProperties.name << std::endl;
		    gpuErrchk(cudaSetDevice(deviceIndex));
		    break;
		}
	}
}













