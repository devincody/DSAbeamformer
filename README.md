# DSA beamformer

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
Beamforming is accomplished with a single `cublasGemmStridedBatchedEx()` call. To understand the indexing and striding of this function, we need to take a look at how the beamforming step is constructed. Consider first the monochromatic beamformer (top image). Here, the beam forming step is a simple matrix vector multiplication where the vector is data from each of the (64) antennas and the matrix is a fourier coefficient matrix whose weights are determined by the position of the antennas and direction of the beam steering. 

![beamforming steps](https://github.com/devincody/DSAbeamformer/blob/docs/images/Beamforming%20steps.png "Beamforming Steps")

We can next expand our data vector with multiple timesteps (middle image). While not physically motivated, this will help improve the throughput of our GPU system (since we would otherwise have to do a matrix-vector multiplication for each time step). Lastly, we can tell CUBLAS to do multiple matrix-matrix multiplications at once to again increase throughput. This can be exploited to simultaneously do beamforming for all frequencies.

`cublasGemmStridedBatchedEx()` takes 19 parameters which are defined below:

| Parameter   | Value                           | Notes                                                 |
|-------------|---------------------------------|-------------------------------------------------------|
| Transa      | CUDA_OP_N                       | Matrix A is not transposed                            |
| Transb      | CUDA_OP_N                       | Matrix B is not transposed                            |
| M           | N_BEAMS                         | Number of Rows of A/C                                 |
| N           | N_TIMESTEPS_PER_CALL            | Number of Columns of B/C                              |
| K           | N_ANTENNAS                      | Number of Columns of A, Number of Rows in B           |
| Alpha       | 1.0/127                         | Prefactor Scaling                                     |
| Atype       | CUDA_C_8I                       | Data type of Fourier Matrix                           |
| Lda         | N_BEAMS                         | Leading dimension of Fourier Matrix                   |
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
The data coming out of the beamforming step is a complex number corresponding to the voltage of every beam. To make a meaning full detection, we need to take the power of each beam. The detection step, executed by the `detect_sum()` cuda kernel squares the real part of each beam and furthermore averages over 16 time samples to reduce the data rate.


### Real-time operation
To increase throughput of data, several streams are used to overlap memory transfers and computations. Furthermore, the code keeps track of the number of blocks transfered to the GPU and the number of block analyzed at every moment and issues a transfer command any time the number of blocks transfered is within two of the number of blocks analyzed. When the number of blocks transfered is equal to the number of block analyzed, the code will use a synchronous copy, however when there are more transfered blocks than analyzed blocks, the code will use an asynchronous copy on a non-computational stream.

## Similar Projects
https://arxiv.org/abs/1412.4907
http://journals.pan.pl/Content/87923/PDF/47.pdf
https://www.researchgate.net/publication/220734131_Digital_beamforming_using_a_GPU
https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=7622&context=etd
