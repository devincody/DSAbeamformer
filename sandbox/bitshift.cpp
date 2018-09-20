#include <iostream>
#include <bitset>

int main(){
	char a = 4;
	//a = (a << 4) >> 3;
	//a = (a >> 4);
	std::bitset<16>x( (int16_t) a);
	std::cout << "a = " << x << std::endl;

	return 0;
}
