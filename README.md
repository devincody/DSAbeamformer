# DSAbeamformer

GPU implementation of a beamformer for the DSA project

## How does the code work?

### Overview
The code is split into 4 primary kernels:
1. Data reordering
2. 4-bit to 8-bit data expansion
3. Beamforming
4. Data Detection

### Data reordering
Data (int4 + int4) arrives from FPGA snap boards with the following data indexing:

~~Time   X  Frequency X Time_Batch X Time X Polarization X Antenna X Real/Imag
(cont.)      (256)	      (~3)      (16) 	     (2)	       (64) 	    (2)~~

Once on the GPU, the code is rearranged to the following order:

Time   X  Frequency X Time_Batch X Time X Polarization X Antenna X Real/Imag
(cont.)      (256)	      (~3)      (16) 	     (2)	       (64) 	    (2)

### 4-bit to 8-bit data conversion
The GPU recieves 4-bit data from the signal capture FPGAs. In order for the data to work with the CUBLAS tensor core, we need to convert this 4-bit data into 8-bit data. The following code accomplishes this:

```c++
char high = (temp >> 4); // roll the 		    most  significant 4 bits over the least significant 4 bits
char low = (temp << 4);  // roll the 		    least significant 4 bits over the most  significant 4 bits
low = (low >> 4);        // roll the *new* 	most  significant 4 bits over the least significant 4 bits
 ```

#### A note about global memory access bandwidth
To maximize the throughput of global memory accesses, it's best to use coalesced data accesses with 32-bit or larger data types. To optimize the memory access 


### Memory Management
Code keeps track of the number of blocks transfered to the GPU and the number of block analyzed at every moment and issues a transfer command any time the number of blocks analyzed is within 2 of the number of blocks transfered.



## Similar Projects
http://journals.pan.pl/Content/87923/PDF/47.pdf
https://www.researchgate.net/publication/220734131_Digital_beamforming_using_a_GPU
https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=7622&context=etd
