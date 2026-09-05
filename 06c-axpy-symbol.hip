#include "hip/hip_runtime.h"
#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

const int N = 64 * 1024 * 1024; // big-size elements
const float hostA = 0.789F;
__constant__ float devA = 0.0F;

float X[N];
float Y[N];
float Z[N];

__device__ float devX[N];
__device__ float devY[N];
__device__ float devZ[N];


// CPU kernel function
__host__ void kernel_cpu_axpy(int i) {
	Z[i] = hostA * X[i] + Y[i];
}

// CUDA kernel function
__global__ void kernel_axpy(int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; // global index
	if (i < N) {
		devZ[i] = devA * devX[i] + devY[i];
	}
}

int main( const int argc, const char* argv[] ) {
	srand(time(NULL));
#if 0
	N = parseNum(argc, argv, N); // we cannot change N in this example !
#endif
	printf("N = %d\n", N);
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	setRand( N, X );
	setRand( N, Y );
	printf("done\n");
	printf("A = %f\n", hostA);
	printf("X[%d] = ", N);
	printArray(N, X);
	printf("Y[%d] = ", N);
	printArray(N, Y);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_axpy(i);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's Z[%d] = ", N);
	printArray(N, Z);
	memset(Z, 0, N * sizeof(float));
	fflush(stdout);
	// GPU part : warm up
	CUDA_WARM_UP();
	// host to device copy
	hipMemcpyToSymbol(HIP_SYMBOL(devA), &hostA, sizeof(float));
	hipMemcpyToSymbol(HIP_SYMBOL(devX), X, N * sizeof(float));
	hipMemcpyToSymbol(HIP_SYMBOL(devY), Y, N * sizeof(float));
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// kernel action
	start = steady_clock::now();
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernel_axpy <<< dimGrid, dimBlock>>>(N);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// device to host copy
	hipMemcpyFromSymbol(Z,HIP_SYMBOL(devZ), N * sizeof(float));
	printf("GPU's Z[%d] = ", N);
	printArray(N, Z);
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// done
	return 0;
}

// HISTORY: axpy-symbol.cu <-- axpy.cu
static const char rcsid[] __attribute__((used)) = "$Id: 06c-axpy-symbol.cu,v 1.2 2026/09/05 09:40:16 wayfarecru Exp $";

