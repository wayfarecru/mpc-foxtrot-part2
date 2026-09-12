#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"


__device__ int g_dim[5] = { 0, 1, 2, 3, 4 };
int mem_dim[5] = { 0, 1, 2, 3, 4 };

// CUDA kernel function
__global__ void kernelTest(int dim[]) {
	// print
	if (threadIdx.x == 0) {
		printf("dim = { %d, %d, %d, ... };\n", dim[0], dim[1], dim[2]);
	}
}

int main(const int argc, const char* argv[]) {
	// CUDA kernel launch
	steady_clock::time_point start = steady_clock::now();
	kernelTest <<< 1, 32>>>( g_dim );
	cudaDeviceSynchronize();
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	start = steady_clock::now();
	kernelTest <<< 1, 32>>>( mem_dim );
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: param-test.cu
static const char rcsid[] __attribute__((used)) = "$Id: 07f-param-test.cu,v 1.2 2026/09/12 23:11:20 wayfarecru Exp $";

