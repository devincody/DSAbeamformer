//datastructureTest.cpp
#include <iostream>


union doubleType{
	char a[2];
	int16_t b;
};


int main(){
	
	std::cout << "hello! " << std::endl;


	doubleType *test = new doubleType[100];

	for (int i = 0; i < 100; i++){
		test[i].a[1] = 0x70;
		test[i].a[0] = i;
	}

	for (int i = 0; i < 100; i ++){
		std::cout << "test = " << test[i].b << std::endl;
	}


	return 0;
}


