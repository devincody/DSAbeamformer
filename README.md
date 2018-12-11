# DSA beamformer

![Crab Pulsar](https://github.com/devincody/DSAimager/blob/master/Images/pulse.gif)

The Crab pulsar as imaged by DSA.

## What is the Deep Synoptic Array (DSA)?
DSA is a 10-element radio interferometer located at the Owens Valley Radio Observatory (OVRO) in California. The purpose of this array is to detect and localize enignmatic pulses of radio energy known as fast radio bursts (FRBs). If you're interested in learning more about radio interferometers, check out my blog post about how they work [here](https://devincody.github.io/Blog/2018/02/27/An-Introduction-to-Radio-Interferometry-for-Engineers.html). 

## What does this code do?
This is a collection of gpu-accelerated code which searches for FRBs in realtime using beamforming. When run on GTX 1080 Ti devices, the code will produce 1 set of (256) beams every 0.131 ms.

## How does it work?

### Overview
The code is split into 4 primary kernels:
1. Data reordering
2. 4-bit to 8-bit data expansion
3. Beamforming
4. Detection

### Data reordering
Data (int4 + int4) arrives from FPGA snap boards with the following data indexing:

~~Time   X  Frequency X Time_Batch X Time X Polarization X Antenna X Real/Imag~~

~~(cont.)      (256)	      (~3)      (16) 	     (2)	       (64) 	    (2)~~

Once on the GPU, the code is rearranged to the following order:

Time   X  Frequency X Time_Batch X Time X Polarization X Antenna X Real/Imag

(cont.)      (256)	      (~3)      (16) 	     (2)	       (64) 	    (2)

### 4-bit to 8-bit data conversion
The GPU recieves 4-bit data from the signal capture FPGAs. In order for the data to work with the CUBLAS tensor core, we need to convert this 4-bit data into 8-bit data. The following code accomplishes this:

```c++
char high = (temp >> 4); // roll the       most  significant 4 bits over the least significant 4 bits
char low = (temp << 4);  // roll the       least significant 4 bits over the most  significant 4 bits
low = (low >> 4);        // roll the *new* most  significant 4 bits over the least significant 4 bits
 ```

#### A note about global memory access bandwidth
To maximize the throughput of global memory accesses, it's best to use coalesced data accesses with 32-bit or larger data types. We can therefore maximize throughput by defining the following data structures:

```c++
typedef char2 CxInt8_t;
typedef char char4_t[4]; //32-bit so global memory bandwidth usage is optimal
typedef char char8_t[8]; //64-bit so global memory bandwidth usage is optimal
typedef CxInt8_t cuChar4_t[4];
```

By using reinterpret_cast() and the above data structures, we can convince cuda to read/write multiple 4-bit/8-bit numbers.

### Beamforming
Beamforming is accomplished with a single `cublasGemmStridedBatchedEx()` call. To understand the indexing and striding of this function, we need to take a look at how the beamforming step is constructed. Consider first the monochromatic beamformer (top). Here, the beam forming step is a simple matrix vector multiplication where the vector is data from each of the (64) antennas and the matrix is a fourier coefficient matrix whose weights are determined by the position of the antennas and direction of the beam steering. 

![beamforming steps](https://github.com/devincody/DSAbeamformer/blob/docs/images/Beamforming%20steps.png "Beamforming Steps")

We can next expand our data vector with multiple timesteps (middle). While not physically motivated, this will help improve the throughput of our GPU system (since we would otherwise have to do a matrix-vector multiplication for each time step). Lastly, we can tell CUBLAS to do multiple matrix-matrix multiplications at once to again increase throughput. This can be exploited to simultaneously do beamforming for all frequencies.

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


### Detection
The data coming out of the beamforming step is a complex number corresponding to the voltage of every beam. To make a meaning full detection, we need to take the power of each beam. The detection step, executed by the `detect_sum()` cuda kernel, squares and sums the real and imaginary parts of each beam. It furthermore averages over 16 time samples to reduce the data rate.


### Real-time operation

At all times, the program maintains four numbers which describe the state of the beamformer. These numbers track the movement of blocks as they progress through the GPU. `blocks_transfer_queue` (TQ) keeps track of the total number of block transfer requests that have been issued, and `blocks_analysis_queue` (AQ) keeps track of the number of blocks that have been queued for analysis with the cudaRuntime. `blocks_transfered` (T) and `blocks_analyzed` (A) keep track of the total number of blocks that have been transfered to the GPU and analyzed respectively. 

![RealtimeQueues](https://github.com/devincody/DSAbeamformer/blob/devincody-doc2/images/RealtimeQueues.PNG "Realtime principle of operation")

It's perhaps easiest to visualize the relationship between these four numbers as pointers on the number line. In this representation, the numbers between A and AQ and the numbers between T and TQ form two queues, where A and T, are the fronts of the queues and AQ and TQ are the ends of the queues. Every time an `asyncCudaMemcpy()` is issued, TQ is moved down the line and every time a transfer is completed, T is shifted. Similarly, when the kernels for a block have been issued, AQ is moved down the line and when all the kernels for a block have completed, A is incremented. Because the transfers and kernel calls are issued asynchronously, we use cudaEvents to keep track of when they are completed.

With this model in mind, we can start developing rules to determine what actions are taken based on the state of these four numbers. We can think of this as somewhat akin to a mealy finite state machine. Ultimately, there are four basic update rules for each of the numbers:

1. Update TQ: Blocks should not be added to the transfer queue faster than blocks are analyzed or faster than blocks are being transfered. To implement this, we define two seperation metrics `total_separation` and `transfer_separation` which control how far apart A and TQ and similarly T and TQ can be apart respectively.

2. Update AQ: Blocks should not be added to the analysis queue unless they've been transfered to the GPU that is to say: if and only if AQ < T, add blocks to the queue and increment AQ.

3. Update T: For every transfer request there is a corresponding cudaEvent. During our loop, we check all of the cudaEvents between T and TQ for completion and if we get a cudaSuccess, then we increment T.

4. Update A: For every analysis request there is a corresponding cudaEvent. During our loop, we check all of the cudaEvents between A and AQ for completion and if we get a cudaSuccess, then we increment A.

## Running the code

A makefile is provided to facilitate compilation. There are two options which can be used: `make verbose` and `make debug` which optionally toggle print statements and debugging utilities respectively.

Generally, the template for preparing the system for execution is:

``` bash
make #compile the code
sudo dada_db -k <buffer name> -d # delete previous dada buffer
sudo dada_db -k <buffer name> -n <number of blocks> -b <block size (bytes)> -l -p # create new dada buffer
```
As a concrete example:
``` bash
make #compile the code
sudo dada_db -k baab -d # delete previous dada buffer
sudo dada_db -k baab -n 8 -b 268435456 -l -p # create new dada buffer
```
The above commands are included in a bash script in `util/exe.sh`.

The following template starts the system:
``` bash
bin/beam -k <buffer name> -c <cpu number> -g <gpu number>
dada_junkdb -c <cpu number> -z -k <buffer name> -r <buffer fill rate (MB/s)> -t <fill time (s)> <header>
```

For example:
``` bash
bin/beam -k baab -c 0 -g 0
dada_junkdb -c 0 -z -k baab -r 4000 -t 10 lib/correlator_header_dsaX.txt
```
Here, the first line starts the script and the second starts filling the dada buffer. Note that we include a correlation header in the `lib` folder.

## Demonstration of Correctness
This system was prototyped in python (see for example `Beamformer Theory.ipynb`). Program correctness is determined exclusively in relation to the python implementation. 

![Percent Difference](https://github.com/devincody/DSAbeamformer/blob/streams/images/BeamformerValidation.png "GPU Correctness Validation")

The left two graphs show beam power as a function of source direction (1024) and beam number (256) for the GPU implementation and python implementations respectively. The graph on the right shows percent difference between the two implementations on a scale from 0 to 1 percent.

![Implementation Histograms](https://github.com/devincody/DSAbeamformer/blob/streams/images/BeamformerValidationHistograms.png "GPU Correctness Histograms")

The above figure shows a histogram of beam powers for the two images in the previous plot. Note the log-log axes. The graph on the right shows a histogram of the percent errors between the two implementations. As shown, the error has an average value of 0.03% and is bounded by 0.8% across all pixels.

## Similar Projects
https://arxiv.org/abs/1412.4907
http://journals.pan.pl/Content/87923/PDF/47.pdf
https://www.researchgate.net/publication/220734131_Digital_beamforming_using_a_GPU
https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=7622&context=etd
