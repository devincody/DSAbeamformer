#include <iostream>

#include <cublas_v2.h>
#include <cuComplex.h>
// #include <>

#define N_BEAMS 256
#define N_AVERAGING 16
#define N_TIMESTEPS_PER_CALL 64
#define N_POL 2
#define N 1000

typedef char2 CxInt8_t;
typedef char c4[4]; //32-bit so global memory bandwidth usage is optimal
typedef char c8[8]; //64-bit so global memory bandwidth usage is optimal
typedef CxInt8_t cu4[4];


union Sc4{
	c4 c;
	float s;
};

union Dc8{
	c8 c;
	CxInt8_t cu[4];
	double d;
};


__global__
void reduce_input(Sc4 *input, Dc8 *output, int input_size){
	/*
	This code takes in an array of 4-bit integers and returns an array of 8-bit integers.
	To maximize global memory bandwidth and symplicity, two special char types are 
	defined: c4 and c8. The size of these types are 32-bit and 64-bit respectively. These
	enable coalesced memory accesses, but then require the gpu to handle the 4 to 8-bit
	conversion simultaneously for 8 numbers (4 real/imaginary pairs). 
	*/

	__shared__ Sc4 shmem_in[32];
	__shared__ Dc8 shmem_out[32];

	int local_idx = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (global_idx < input_size){
		shmem_in[local_idx].s = input[global_idx].s; // read eight pieces of 4-bit memory into shared memory

		//#pragma unroll
		for (int i = 0; i < 4; i++){
			char temp = shmem_in[local_idx].c[i];

			// break the char into two 4-bit chunks, then convert to 8-bit
			char high = (temp >> 4); // roll the 		most  significant 4 bits over the least significant 4 bits
			char low = (temp << 4);  // roll the 		least significant 4 bits over the most  significant 4 bits
			low = (low >> 4); 	 	 // roll the *new* 	most  significant 4 bits over the least significant 4 bits

			// store the two 8-bit numbers to the output shared memory array
			shmem_out[local_idx].cu[i].x = high; 
			shmem_out[local_idx].cu[i].y = low;
		}

		output[global_idx].d = shmem_out[local_idx].d;	// write eight pieces of 8-bit memory out to global memory

		global_idx += gridDim.x *blockDim.x;
	}
}


__global__
void detect_sum(cuComplex *input, cuComplex *output){
	__shared__ float shmem[N_BEAMS];

	int tid = blockIdx.x * blockDim.x*N_POL*N_AVERAGING + threadIdx.x;

	cuComplex in;

	#pragma unroll
	for (int i = 0; i < N_POL*N_AVERAGING; i++){
		in = input[tid];
		shmem[tid] += in.x*in.x + in.y*in.y;
		tid += blockDim.x;
	}

}

__global__
void expand_input(float *input, double *output, int input_size){
	/*
	This code takes in an array of 4-bit integers and returns an array of 8-bit integers.
	To maximize global memory bandwidth and symplicity, two special char types are 
	defined: c4 and c8. The size of these types are 32-bit and 64-bit respectively. These
	enable coalesced memory accesses, but then require the gpu to handle the 4 to 8-bit
	conversion simultaneously for 8 numbers (4 real/imaginary pairs). 
	*/

	__shared__ float shmem_in[32];
	__shared__ double shmem_out[32];

	c4 *char_shmem_in;
	cu4 *char_shmem_out;

	char_shmem_in = reinterpret_cast<c4*>(shmem_in);
	char_shmem_out = reinterpret_cast<cu4 *>(shmem_out);

	int local_idx = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (global_idx < input_size){
		shmem_in[local_idx] = input[global_idx]; // read eight pieces of 4-bit memory into shared memory

		//#pragma unroll
		for (int i = 0; i < 4; i++){
			char temp = char_shmem_in[local_idx][i];

			// break the char into two 4-bit chunks, then convert to 8-bit
			char high = (temp >> 4); // roll the 		most  significant 4 bits over the least significant 4 bits
			char low = (temp << 4);  // roll the 		least significant 4 bits over the most  significant 4 bits
			low = (low >> 4); 	 	 // roll the *new* 	most  significant 4 bits over the least significant 4 bits

			// store the two 8-bit numbers to the output shared memory array
			char_shmem_out[local_idx][i].x = high; 
			char_shmem_out[local_idx][i].y = low;
		}

		output[global_idx] = shmem_out[local_idx];	// write eight pieces of 8-bit memory out to global memory

		global_idx += gridDim.x *blockDim.x;
	}
}








int main (){
	std::cout << "hello " << std::endl;



	// char *in = new char[N];
	// // doubleType *out = new doubleType[N];

	// for (int i = 0; i < N; i ++){
	// 	in[i] = 0xE7;
	// }

	// char *d_in;
	// doubleType *d_out;

	// cudaMalloc(&d_in, N*sizeof(char));
	// cudaMalloc(&d_out, N*sizeof(doubleType));

	// cudaMemcpy(d_in, in, N*sizeof(char), cudaMemcpyHostToDevice);

	// reduce_input<<<20,32>>>(d_in, d_out);

	// cudaMemcpy(d_out, out, N*sizeof(doubleType), cudaMemcpyDeviceToHost);

	// for (int i = 0;)





	return 0;
}



