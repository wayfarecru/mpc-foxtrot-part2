#include "hip/hip_runtime.h"
#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

// matrix size
int NROW = 10000;
int NCOL = 10000;

float* matA = nullptr;
float* matB = nullptr;
float* matC = nullptr;
size_t hostPitchA = 0;
size_t hostPitchB = 0;
size_t hostPitchC = 0;

float* devMatA = nullptr;
float* devMatB = nullptr;
float* devMatC = nullptr;
size_t devPitchA = 0;
size_t devPitchB = 0;
size_t devPitchC = 0;

// CPU kernel function
__host__ void kernel_cpu_mat_add(int row, int col, float* C, const float* A, const float* B) {
	int i = row * NCOL + col;
	C[i] = A[i] + B[i];
}

// GPU kernel function
__global__ void kernel_mat_add(float* C, const float* A, const float* B, int NROW, int NCOL, size_t stride) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;  // gx
	int row = blockIdx.y * blockDim.y + threadIdx.y;  // gy
	if (row < NROW && col < NCOL) {
		int i = row * stride + col;
		C[i] = A[i] + B[i];
	}
}

// GPU kernel function for oversized matrices
__global__ void kernel_mat_add_oversize(float* C, const float* A, const float* B, size_t stride) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;  // gx
	int row = blockIdx.y * blockDim.y + threadIdx.y;  // gy
	int i = row * stride + col;
	C[i] = A[i] + B[i];
}


int main(const int argc, const char* argv[]) {
	srand(time(NULL));
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		NROW = NCOL = procArg( argv[0], argv[1], 32, 256 * 1024 );
		break;
	case 3:
		NROW = procArg( argv[0], argv[1], 32, 256 * 1024 );
		NCOL = procArg( argv[0], argv[2], 32, 256 * 1024 );
		break;
	default:
		printf("usage: %s [nrow] [ncol]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// CPU part: set values
	printf("generating %d x %d random numbers: ", NROW, NCOL);
	fflush(stdout);
	try {
		matA = new float[NROW * NCOL];
		matB = new float[NROW * NCOL];
		matC = new float[NROW * NCOL];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	setRand(NROW * NCOL, matA);
	setRand(NROW * NCOL, matB);
	printf("done\n");
	printf("A[%d][%d] =\n", NROW, NCOL);
	printMat(NROW, NCOL, matA);
	printf("B[%d][%d] =\n", NROW, NCOL);
	printMat(NROW, NCOL, matB);
	fflush(stdout);
	// set host pitch
	hostPitchA = NCOL * sizeof(float); // host side: packed compactly
	hostPitchB = NCOL * sizeof(float); // host side: packed compactly
	hostPitchC = NCOL * sizeof(float); // host side: packed compactly
	printf("host pitch = %zd (byte), host stride = %u (element)\n", hostPitchA, NCOL);
	// CPU kernel execution
	steady_clock::time_point start = steady_clock::now();
	for (int r = 0; r < NROW; ++r) {
		for (int c = 0; c < NCOL; ++c) {
			kernel_cpu_mat_add(r, c, matC, matA, matB);
		}
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's C[%d][%d] =\n", NROW, NCOL);
	printMat(NROW, NCOL, matC);
	memset(matC, 0, NROW * NCOL * sizeof(float));
	// GPU part: warmup
	CUDA_WARM_UP();
	// GPU part: pre-calculate the kernel dimensions
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((NCOL + dimBlock.x - 1) / dimBlock.x, (NROW + dimBlock.y - 1) / dimBlock.y, 1);
	CUDA_PRINT_CONFIG_2D(NROW, NCOL); // defined in "common.hpp"
	dim3 numThread(dimBlock.x * dimGrid.x, dimBlock.y * dimGrid.y, dimBlock.z * dimGrid.z);
	// GPU part: memory allocation
	hipMallocPitch( (void**)&devMatA, &devPitchA, NCOL * sizeof(float), numThread.y );
	hipMallocPitch( (void**)&devMatB, &devPitchB, NCOL * sizeof(float), numThread.y );
	hipMallocPitch( (void**)&devMatC, &devPitchC, NCOL * sizeof(float), numThread.y );
	CUDA_CHECK_ERROR();
	assert(devPitchA == devPitchB);
	assert(devPitchA == devPitchC);
	// copy to device from host
	hipMemcpy2D( devMatA, devPitchA, matA, hostPitchA, NCOL * sizeof(float), NROW, hipMemcpyHostToDevice );
	hipMemcpy2D( devMatB, devPitchB, matB, hostPitchB, NCOL * sizeof(float), NROW, hipMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	int stride = devPitchA / sizeof(float);
	printf("device pitch = %zd (byte), device stride = %u (element)\n", devPitchA, stride);
	printf("oversized case = %s\n", (numThread.x <= devPitchA) ? "true" : "false");
	if (numThread.x <= devPitchA) { // oversized matrix case
		start = steady_clock::now();
		kernel_mat_add_oversize <<< dimGrid, dimBlock>>>( devMatC, devMatA, devMatB, stride );
		hipDeviceSynchronize();
		end = steady_clock::now();
	} else {
		start = steady_clock::now();
		kernel_mat_add <<< dimGrid, dimBlock>>>( devMatC, devMatA, devMatB, NROW, NCOL, stride );
		hipDeviceSynchronize();
		end = steady_clock::now();
	}
	CUDA_CHECK_ERROR();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	// copy to host from device
	hipMemcpy2D( matC, hostPitchC, devMatC, devPitchC, NCOL * sizeof(float), NROW, hipMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	printf("GPU's C[%d][%d] =\n", NROW, NCOL);
	printMat(NROW, NCOL, matC);
	// free device memory
	hipFree( devMatA );
	hipFree( devMatB );
	hipFree( devMatC );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: mat-add-pitched <-- mat-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 05d-mat-add-pitched.cu,v 1.2 2026/09/05 08:53:57 wayfarecru Exp $";
