#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

int N = 1024 * 1024 * 1024; // 1G elements

float* ptrA; // host memory (main memory)
float* ptrB;

float* devPtrA; // device memory (graphics memory)
float* devPtrB;

// CPU kernel function
__host__ void kernel_cpu_adj_diff(int i, float* B, const float* A) {
	if (i == 0) {
		B[i] = A[i] - 0.0F;
	} else {
		B[i] = A[i] - A[i - 1];
	}
}

// CUDA kernel function
__global__ void kernel_adj_diff(float* B, const float* A, int N) {
	extern __shared__ float s_Buf[]; // size not fixed at compile time
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	if (i < N) {
		s_Buf[tx] = A[i];
		__syncthreads();
		if (tx > 0) {
			B[i] = s_Buf[tx] - s_Buf[tx - 1];
		} else if (i > 0) {
			B[i] = s_Buf[tx] - A[i - 1];
		} else { // i == 0
			B[i] = s_Buf[i] - 0.0F;
		}
	}
}


int main(int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	ptrA = (float*)malloc(N * sizeof(float));
	ptrB = (float*)malloc(N * sizeof(float));
	assert(ptrA != nullptr);
	assert(ptrB != nullptr);
	setRand(N, ptrA);
	printf("done\n");
	printf("A[%d] = ", N);
	printArray(N, ptrA);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_adj_diff(i, ptrB, ptrA);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's B[%d] = ", N);
	printArray(N, ptrB);
	memset(ptrB, 0, N * sizeof(float));
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrA, N * sizeof(float));
	cudaMalloc((void**)&devPtrB, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// host to device copy
	cudaMemcpy(devPtrA, ptrA, N * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// kernel action
	dim3 dimBlock(512, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	int bufSizeInByte = dimBlock.x * sizeof(float);
	CUDA_PRINT_CONFIG(N);
	printf("buf size in byte = %d\n", bufSizeInByte);
	start = steady_clock::now();
	kernel_adj_diff <<< dimGrid, dimBlock, bufSizeInByte>>>(devPtrB, devPtrA, N);
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// device to host copy
	cudaMemcpy(ptrB, devPtrB, N * sizeof(float), cudaMemcpyDeviceToHost); // ptrC <- devPtrC
	printf("GPU's B[%d] = ", N);
	printArray(N, ptrB);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: adj-diff-flexible.cu <-- adj-diff-shared.cu <-- adj-diff.cu <-- axpy.cu
static const char rcsid[] __attribute__((used)) = "$Id: 07b-adj-diff-flexible.cu,v 1.2 2026/09/12 23:09:22 wayfarecru Exp $";

