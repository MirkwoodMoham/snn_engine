#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

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