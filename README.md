# DSA Real Time Beamformer

![Crab Pulsar](https://github.com/devincody/DSAimager/blob/master/Images/pulse.gif)

The Crab pulsar as imaged by DSA10.

## What is the Deep Synoptic Array (DSA100)?
DSA is a 100-element radio interferometer located at the Owens Valley Radio Observatory (OVRO) in California. The purpose of this array is to detect and localize enigmatic pulses of radio energy known as fast radio bursts (FRBs).

## What does this code do?
This is a collection of gpu-accelerated code which searches for FRBs in real time using beamforming. When run on RTX 2080 Ti devices, the code will produce 1 set of (256) beams every ~0.13 ms.

## Prerequisites
You should have the following code packages installed before using DART:

1. [Cuda 10.0](https://developer.nvidia.com/cuda-downloads)
2. [PSRDADA](http://psrdada.sourceforge.net/download.shtml)
	1. csh
	2. m4
	3. libtools
	4. autoconf
	5. automake


## How does it work?

### Overview
The code is split into 4 primary kernels:
1. Data reordering
2. 4-bit to 8-bit data expansion
3. Beamforming
4. Detection

### Data reordering (Not currently implemented)
Data (int4 + int4) arrives from FPGA snap boards with the following data indexing:

~~Time   X  Frequency X Time_Batch X Time X Polarization X Antenna X Real/Imag~~

~~(cont.)      (256)	      (~3)      (16) 	     (2)	       (64) 	    (2)~~

Once on the GPU, the code is rearranged to the following order:

Time   X  Frequency X Time_Batch X Time X Polarization X Antenna X Real/Imag

(cont.)      (256)	      (~3)      (16) 	     (2)	       (64) 	    (2)

#### Data hierarchy
The hierarchy of data in the gpu can be confusing, so a explanation is warranted. The basic unit of data that arrives from an FPGA is the dada block. Within the dada block, there is (depending on the configuration) enough data for multiple cuBlas GEMMs. Within a single cuBlas GEMM call, there's (typically) enough data for multiple time *outputs*. However, because we average multiple time *inputs* for each *output*, we also have a number of *inputs* for each output. Finally, within one *input* time step, we typically have complex data from 64 antennas, 2 polarizations, 256 frequencies. 

### 4-bit to 8-bit data conversion
The GPU receives 4-bit data from the signal capture FPGAs. In order for the data to work with the gpu tensor cores, we need to convert this 4-bit data into 8-bit data. The following code accomplishes this:

```c++
char high = (temp >> 4); // roll the       most  significant 4 bits over the least significant 4 bits
char low = (temp << 4);  // roll the       least significant 4 bits over the most  significant 4 bits
low = (low >> 4);        // roll the *new* most  significant 4 bits over the least significant 4 bits
 ```
 
Note that the last two steps will get "optimized" away by some compilers if combined into one line. Therefore it's important to keep them separate.

#### A note about global memory access bandwidth
To maximize the throughput of global memory accesses, it's best to use coalesced data accesses with 32-bit or larger data types. We can therefore maximize throughput by defining the following data structures:

```c++
typedef char2 CxInt8_t;        // Define our complex type
typedef char char4_t[4];       // 32-bit char array so global memory bandwidth usage is optimal
typedef char char8_t[8];       // 64-bit char array so global memory bandwidth usage is optimal
typedef CxInt8_t cuChar4_t[4]; // 64-bit array consisting of 4 complex numbers
```

By using reinterpret_cast() and the above data structures, we can convince cuda to read/write multiple 4-bit/8-bit numbers in a coalesced manner. 

### Beamforming
Beamforming is accomplished with a single `cublasGemmStridedBatchedEx()` call. To understand the indexing and striding of this function, we need to take a look at how the beamforming step is constructed. Consider first the monochromatic beamformer (top). Here, the beam forming step is a simple matrix vector multiplication where the vector is data from each of the (64) antennas and the matrix is a Fourier coefficient matrix whose weights are determined by the position of the antennas and direction of the beam steering. 

![beamforming steps](https://github.com/devincody/DSAbeamformer/blob/docs/images/Beamforming%20steps.png "Beamforming Steps")

We can next expand our data vector with multiple time steps (middle). While not physically motivated, this will help improve the throughput of our GPU system (since we would otherwise have to do a matrix-vector multiplication for each time step). Lastly, we can tell CUBLAS to do multiple matrix-matrix multiplications at once to again increase throughput. This can be exploited to simultaneously do beamforming for all frequencies.

`cublasGemmStridedBatchedEx()` takes 19 parameters which are defined below:

| Parameter   | Value                           | Notes                                                 |
|-------------|---------------------------------|-------------------------------------------------------|
| Transa      | CUDA_OP_N                       | Matrix A (Fourier coefficient matrix) is not transposed     |
| Transb      | CUDA_OP_N                       | Matrix B (input data) is not transposed               |
| M           | N_BEAMS                         | Number of rows of A/C                                 |
| N           | N_TIMESTEPS_PER_CALL            | Number of columns of B/C                              |
| K           | N_ANTENNAS                      | Number of columns of A, number of rows in B           |
| Alpha       | 1.0/127                         | Fourier coefficients are 8-bit ints so must be normalized to 1 |
| Atype       | CUDA_C_8I                       | Data type of Fourier coefficient matrix (i.e. 8-bit + 8-bit integers)                |
| Lda         | N_BEAMS                         | Leading dimension of Fourier coefficient matrix                   |
| strideA     | N_BEAMS*N_ANTENNAS              | Stride between different frequencies in A             |
| Btype       | CUDA_C_8I                       | Data type of input data matrix                        |
| Ldb         | N_ANTENNAS                      | Leading dimension of input matrix                     |
| StrideB     | N_ANTENNAS*N_TIMESTEPS_PER_CALL | Stride between different frequencies in input matrix  |
| Beta        | 0                               | Zero out the output data tensor                       |
| Ctype       | CUDA_C_32F                      | Data type of output matrix                            |
| Ldc         | N_BEAMS                         | Leading Dimension of output matrix                    |
| strideC     | N_BEAMS*N_TIMESTEPS_PER_CALL    | Stride between different frequencies in output matrix |
| batchCount  | N_FREQUENCIES                   | How many frequencies                                  |
| computeType | CUDA_C_32F                      | Internal datatype                                     |
| Algo        | CUBLAS_GEMM_DEFAULT_TENSOR_OP   | Use tensor operations                                 |


### Detection and Averaging
The data coming out of the beamforming step is a complex number corresponding to the voltage of every beam. To make a meaning full detection, we need to take the power of each beam. The detection step, executed by the `detect_sum()` cuda kernel, squares and sums the real and imaginary parts of each beam. It furthermore averages over 16 time samples to reduce the data rate.


## Real Time Theory of Operation

Real-time operation is achieved by juggling the two most important GPU operations: transfering data onto the GPU and processing the data. This is done continuously and indefinitely with a `while(data_valid)` loop. At all times, the program maintains four numbers which describe the state of the beamformer. These numbers track the movement of blocks as they progress through the GPU. `blocks_transfer_queue` (TQ) keeps track of the total number of block transfer requests that have been queued for transfer onto the GPU, and `blocks_analysis_queue` (AQ) keeps track of the number of blocks that have been queued for analysis (transfer off the GPU is implicitly part of "analysis"). `blocks_transferred` (T) and `blocks_analyzed` (A) keep track of the total number of blocks that have been transferred to the GPU and analyzed respectively. 

![RealtimeQueues](https://github.com/devincody/DSAbeamformer/blob/devincody-doc2/images/RealtimeQueues.PNG "Realtime principle of operation")

It's perhaps easiest to visualize the relationship between these four numbers as pointers on the number line. In this representation, the numbers between A and AQ and the numbers between T and TQ form two queues. A and T, are the fronts of the queues and AQ and TQ are the ends of the queues. Every time an `asyncCudaMemcpy()` (i.e. a transfer request) is issued (requested), TQ is incremented by one. Every time a transfer is completed, T is incremented. Similarly, when all the kernels for a block have been issued, AQ is incremented; when all the kernels for a block have completed, A is incremented. Because the transfers and kernel calls are issued asynchronously, we use cudaEvents to keep track of when they are completed.

With this model in mind, we can start developing rules to determine what actions are taken based on the state of these four numbers. We can think of this as somewhat akin to a mealy finite state machine. Ultimately, there are four basic update rules for each of the numbers:

1. Update TQ: Blocks should not be added to the transfer queue faster than blocks are analyzed or faster than blocks are being transferred. To implement this, we define two separation metrics `total_separation` and `transfer_separation` which control how far apart A and TQ and similarly T and TQ can be apart respectively.

2. Update AQ: Blocks should not be added to the analysis queue unless they've been transferred to the GPU. That is to say: if and only if AQ < T, add blocks to the queue and increment AQ.

3. Update T: For every transfer request there is a corresponding cudaEvent. During our loop, we check all of the cudaEvents between T and TQ for completion and if we get a cudaSuccess, then we increment T.

4. Update A: For every analysis request there is a corresponding cudaEvent. During our loop, we check all of the cudaEvents between A and AQ for completion and if we get a cudaSuccess, then we increment A.

Lastly, we also define a condition for ending the observation -- namely if there's no more valid data. This causes the program to leave the `while()` loop, free all the memory, and shut down.

## Running the code

There are several modes in which the beamformer can be run. These modes can be turned on or off through both compile-time options and through command-line options.

### Observation mode
This is the "normal" observation mode of operation for operating the beamformer. Generally, the template for preparing the system for execution is

#### make verbose
The `make verbose` mode enables many print statements which communicates state information about the program (see Real Time Theory of Operation above). 

#### make debug
The `make debug` mode allows the user 

``` bash
make #compile the code
dada_db -k <buffer name> -d # delete previous dada buffer
dada_db -k <buffer name> -n <number of blocks> -b <block size (bytes)> -l -p # create new dada buffer
```
As a concrete example:
``` bash
make #compile the code
dada_db -k baab -d # delete previous dada buffer
dada_db -k baab -n 8 -b 268435456 -l -p # create new dada buffer
```


### Starting the system with psrdada (not debug mode)

The following template starts the system:
``` bash
bin/beam -k <buffer name> -c <cpu number> -g <gpu number> -p <position_filename> -d <direction_filename>
dada_junkdb -c <cpu number> -z -k <buffer name> -r <buffer fill rate (MB/s)> -t <fill time (s)> <header>
```

For example:
``` bash
bin/beam -k baab -c 0 -g 0
dada_junkdb -c 0 -z -k baab -r 4000 -t 10 lib/correlator_header_dsaX.txt
```

Here, the first line starts the script and the second starts filling the dada buffer. Note that we have included a correlation header in the `lib` folder and the make file will execute the above junkdb command as per the example by calling the pithy command:

```bash
make junk
```

### Starting the system in debug mode

The following template starts the system:
``` bash
bin/beam -g <gpu number> -p <position_filename> -d <direction_filename> -s <source_filename>
```

## Demonstration of Correctness
This system was prototyped in python (see for example `Beamformer Theory.ipynb`). Program correctness is determined exclusively in relation to the python implementation. 

![Percent Difference](https://github.com/devincody/DSAbeamformer/blob/streams/images/BeamformerValidation.png "GPU Correctness Validation")

The left two graphs show beam power as a function of source direction (1024) and beam number (256) for the GPU implementation and python implementations respectively. The graph on the right shows percent difference between the two implementations on a scale from 0 to 1 percent.

![Implementation Histograms](https://github.com/devincody/DSAbeamformer/blob/streams/images/BeamformerValidationHistograms.png "GPU Correctness Histograms")

The above figure shows a histogram of beam powers for the two images in the previous plot. Note the log-log axes. The graph on the right shows a histogram of the percent errors between the two implementations. As shown, the error has an average value of 0.03% and is bounded by 0.8% across all pixels. In general, this is acceptable given the 4-bit accuracy (~6%) of the input data.

## Future Work
Right now, the beamforming is done using a single cuBlas call, however, this may not be the most efficient way of doing things. Here are some alternate approaches and thoughts on their potential success. 

1. Use [Beanfarmer](https://github.com/ewanbarr/beanfarmer). Beanfarmer fuses the beamforming step with the detection step, effectively eliminating a costly trip to global memory. This, however, comes at the cost of increased complexity, reduced maintainability, and fewer opportunities for "free" upgrades with cuda library improvements (i.e. 4-bit arithmetic). 
2. Use [Cutlass](https://github.com/NVIDIA/cutlass) to enable 4-bit GEMM. Can also fuse detection step to the GEMM with the Cutlass `epilogue` functionality, although this doesn't eliminate the global memory trip since it doesn't appear possible to do averaging with `epilogue`.
3. Use Facebook's [Tensor Comprehension Library](https://github.com/facebookresearch/TensorComprehensions) to implement a fused beamforming, detection, and averaging kernel which can be automatically tuned for maximum speed. It's unclear though, if the library can operate on the tensorcores.
4. Use [openCL](https://www.khronos.org/opencl/). Enables access to low(er) cost GPUs via AMD, but again the tensorcores may not be available and requires significant rewrite cost.

## Similar Projects
https://github.com/ewanbarr/beanfarmer

https://arxiv.org/abs/1412.4907

http://journals.pan.pl/Content/87923/PDF/47.pdf

https://www.researchgate.net/publication/220734131_Digital_beamforming_using_a_GPU

https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=7622&context=etd
