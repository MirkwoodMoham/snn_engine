#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

typedef unsigned int uint;

template <typename T>
void highlighted_print(
	T o, 
	const std::string& prefix = "",
    const std::string& line0 = "\n----------------\n", 
	const std::string& line1 = "\n----------------\n")
{
	std::cout << line0 << prefix << ": " << o << line1;
}