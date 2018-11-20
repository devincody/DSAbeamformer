#include <iostream>

int main(){

	std::cout << "hi" << std::endl;

	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
	    cudaDeviceProp deviceProperties;
	    cudaGetDeviceProperties(&deviceProperties, deviceIndex);
	//     printf("Device name: %s", deviceProperties.name);
	    std::cout << deviceProperties.name << std::endl;
	}

	return 0;

}