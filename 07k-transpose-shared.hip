#include "hip/hip_runtime.h"
#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

// matrix size
int N = 10000;

float* matA = nullptr;
float* matC = nullptr;
size_t hostPitchA = 0;
size_t hostPitchC = 0;

float* devMatA = nullptr;
float* devMatC = nullptr;
size_t devPitchA = 0;
size_t devPitchC = 0;

// CPU kernel function
__host__ void kernel_cpu_mat_transpose(int y, int x, float* C, const float* A) {
	int indA = y * N + x;
	int indC = x * N + y;
	C[indC] = A[indA];
}

// GPU kernel function
__global__ void kernel_mat_transpose(float* C, const float* A, int N, size_t stride) {
	__shared__ float s_mat[32][32];
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gy < N && gx < N) {
		int ty = threadIdx.y;
		int tx = threadIdx.x;
		int indA = gy * stride + gx;
		int indC = gx * stride + gy;
		s_mat[ty][tx] = A[indA];
		__syncthreads();
		C[indC] = s_mat[ty][tx];
	}
}

// GPU kernel function for oversized matrices
__global__ void kernel_mat_transpose_oversize(float* C, const float* A, size_t stride) {
	__shared__ float s_mat[32][32];
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int indA = gy * stride + gx;
	int indC = gx * stride + gy;
	s_mat[ty][tx] = A[indA];
	__syncthreads();
	C[indC] = s_mat[ty][tx];
}


int main(int argc, const char* argv[]) {
	srand(time(NULL));
	// argv processing
	N = parseNum(argc, argv, N);
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	try {
		matA = new float[N * N];
		matC = new float[N * N];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	setRand(N * N, matA);
	printf("done\n");
	printf("A[%d][%d] =\n", N, N);
	printMat(N, N, matA);
	// set host pitch
	hostPitchA = N * sizeof(float); // host side: packed compactly
	hostPitchC = N * sizeof(float); // host side: packed compactly
	printf("host pitch = %zd (byte), host stride = %u (element)\n", hostPitchA, N);
	// CPU kernel execution
	steady_clock::time_point start = steady_clock::now();
	for (int r = 0; r < N; ++r) {
		for (int c = 0; c < N; ++c) {
			kernel_cpu_mat_transpose(r, c, matC, matA);
		}
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's C[%d][%d] =\n", N, N);
	printMat(N, N, matC);
	memset(matC, 0, N * N * sizeof(float));
	// GPU part: warmup
	CUDA_WARM_UP();
	// GPU part: pre-calculate the kernel dimensions
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y, 1);
	CUDA_PRINT_CONFIG_2D(N, N); // defined in "common.hpp"
	dim3 numThread(dimBlock.x * dimGrid.x, dimBlock.y * dimGrid.y, dimBlock.z * dimGrid.z);
	// GPU part: memory allocation
	hipMallocPitch( (void**)&devMatA, &devPitchA, N * sizeof(float), numThread.y );
	hipMallocPitch( (void**)&devMatC, &devPitchC, N * sizeof(float), numThread.y );
	CUDA_CHECK_ERROR();
	assert(devPitchA == devPitchC);
	// copy to device from host
	hipMemcpy2D( devMatA, devPitchA, matA, hostPitchA, N * sizeof(float), N, hipMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	int stride = devPitchA / sizeof(float);
	printf("device pitch = %zd (byte), device stride = %u (element)\n", devPitchA, stride);
	printf("oversized case = %s\n", (numThread.x <= devPitchA) ? "true" : "false");
	if (numThread.x <= devPitchA) { // oversized matrix case
		start = steady_clock::now();
		kernel_mat_transpose_oversize <<< dimGrid, dimBlock>>>( devMatC, devMatA, stride );
		hipDeviceSynchronize();
		end = steady_clock::now();
	} else {
		start = steady_clock::now();
		kernel_mat_transpose <<< dimGrid, dimBlock>>>( devMatC, devMatA, N, stride );
		hipDeviceSynchronize();
		end = steady_clock::now();
	}
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// copy to host from device
	hipMemcpy2D( matC, hostPitchC, devMatC, devPitchC, N * sizeof(float), N, hipMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	printf("GPU's C[%d][%d] =\n", N, N);
	printMat(N, N, matC);
	// free device memory
	hipFree( devMatA );
	hipFree( devMatC );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] matA;
	delete[] matC;
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: transpose.cu <-- mat-copy.cu <-- mat-add-pitched <-- mat-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 07k-transpose-shared.cu,v 1.2 2026/09/12 23:17:41 wayfarecru Exp $";

