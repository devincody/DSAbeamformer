//datastructureTest.cpp
#include <iostream>


union doubleType{
	char a[2];
	int16_t b;
};

struct char2{
	char x;
	char y;
};

typedef struct char2 char2;

typedef char2 CxInt8_t;


union quadType{
	char a[4];
	int16_t b[2];
	CxInt8_t c[2];
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



	quadType qtype[100];
	std::cout << "sizeof(quadType) = " << sizeof(quadType) << std::endl;
	qtype[3].a[0] = 0x70;
	qtype[3].a[1] = 0x71;
	qtype[3].a[2] = 0x72;
	qtype[3].a[3] = 0x73;

	std::cout << "q[0]: " << (int) qtype[3].c[0].x << std::endl;
	std::cout << "q[1]: " <<  (int) qtype[3].c[0].y << std::endl;
	std::cout << "q[3]: " << (int) qtype[3].c[1].x << std::endl;
	std::cout << "q[4]: " <<  (int) qtype[3].c[1].y << std::endl;

	CxInt8_t * cnew = (CxInt8_t *) qtype;

	for (int i = 0; i < 7; i++){
		std::cout << "c[0]: " << (int) cnew[i].x << std::endl;
		std::cout << "c[1]: " << (int) cnew[i].y << std::endl;
	}


	return 0;
}


