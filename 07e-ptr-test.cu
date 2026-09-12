#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

__device__ int my_g_var = 0; // CUDA device global variable
__constant__ int my_c_var = 1; // CUDA constant variable

// CUDA kernel function
__global__ void kernelTest(void) {
	__shared__ int my_s_var; // CUDA shared variable
	// initialize
	if (threadIdx.x == 0) {
		my_s_var = 2;
	}
	__syncthreads();
	// pointers
	int* ptr_to_global = &my_g_var;
	const int* ptr_to_constant = &my_c_var;
	int* ptr_to_shared = &my_s_var;
	// test
	*ptr_to_shared = *ptr_to_global;
	// test2
	int* ptr = nullptr;
	if (threadIdx.x % 2) {
		ptr = &my_g_var;
	} else {
		ptr = &my_s_var;
	}
	*ptr = *ptr_to_constant;
	__syncthreads();
	// print
	if (threadIdx.x == 0) {
		printf("global var = %d\n", my_g_var);
		printf("constant var = %d\n", my_c_var);
		printf("shared var = %d\n", my_s_var);
	}
}

// CUDA kernel function
__global__ void kernelTest2(void) {
	__shared__ int my_s_var; // CUDA shared variable
	// initialize
	if (threadIdx.x == 0) {
		my_s_var = 2;
	}
	__syncthreads();
	// pointers
	int* ptr_to_global = &my_g_var;
	const int* ptr_to_constant = &my_c_var;
	int* ptr_to_shared = &my_s_var;
	// test
	*ptr_to_shared = *ptr_to_global;
	// test2
	int* ptr = nullptr;
	ptr = &my_g_var;
	*ptr = *ptr_to_constant;
	ptr = &my_s_var;
	*ptr = *ptr_to_constant;
	__syncthreads();
	// print
	if (threadIdx.x == 0) {
		printf("global var = %d\n", my_g_var);
		printf("constant var = %d\n", my_c_var);
		printf("shared var = %d\n", my_s_var);
	}
}

int main(const int argc, const char* argv[]) {
	// CUDA kernel launch
	steady_clock::time_point start = steady_clock::now();
	kernelTest <<< 1, 32>>>();
	cudaDeviceSynchronize();
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	start = steady_clock::now();
	kernelTest2 <<< 1, 32>>>();
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: ptr-test.cu
static const char rcsid[] __attribute__((used)) = "$Id: 07e-ptr-test.cu,v 1.2 2026/09/12 23:10:35 wayfarecru Exp $";

