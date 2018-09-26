#include <iostream>
#include <bitset>

int main(){
	char positive_number = 0x25;
	char negative_number = 0xA8;


	std::cout << "\nfirst SLOT: " << std::endl;
	char positive_number_first_slot = (positive_number >> 4);
	std::bitset<8>w( positive_number_first_slot);
	std::cout << "positive_number_first_slot = " << w << std::endl;
	
	char negative_number_first_slot = (negative_number >> 4);
	std::bitset<8>z(negative_number_first_slot);
	std::cout << "negative_number_first_slot = " << z << std::endl;

	std::cout << "\nsecond SLOT: " << std::endl;
	char positive_number_second_slot = (positive_number << 4);
	positive_number_second_slot = (positive_number_second_slot >> 4);
	std::bitset<8>y( positive_number_second_slot);
	std::cout << "positive_number_second_slot = " << y << std::endl;


	char negative_number_second_slot = (negative_number << 4);
	negative_number_second_slot = (negative_number_second_slot >> 4);
	std::bitset<8>x(negative_number_second_slot);
	std::cout << "negative_number_second_slot = " << x << std::endl;


	return 0;
}
