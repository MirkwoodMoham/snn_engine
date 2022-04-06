#pragma once

#include <utils/cpp_header.h>
#include <utils/cuda_header.h>


struct CuRandStates
{
    int n_states;
    curandState* states;
    
    CuRandStates(    
        const int n_curand_states,
        bool verbose = true
    );
};

void print_random_numbers(curandState* states, const int n_states);
void print_random_numbers2(std::shared_ptr<CuRandStates> p);

struct CuRandStatesPointer
{
    std::shared_ptr<CuRandStates> ptr_;
    
    CuRandStatesPointer(    
        const int n_curand_states,
        bool verbose = true
    ){
        ptr_ = std::make_shared<CuRandStates>(n_curand_states, verbose);
        //print_random_numbers2(ptr_);
    }

    int n_states(){
        return ptr_->n_states;
    }

    std::shared_ptr<CuRandStates> ptr(){
        return ptr_;
    }
};

