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

float* devMatA = nullptr;
float* devMatB = nullptr;
float* devMatC = nullptr;

// CPU kernel function
__host__ void kernel_cpu_mat_add(int row, int col, float* C, const float* A, const float* B) {
	int i = row * NCOL + col;
	C[i] = A[i] + B[i];
}

// GPU kernel function
__global__ void kernel_mat_add(float* C, const float* A, const float* B, const int NROW, const int NCOL) {
	int row = blockIdx.x * blockDim.x + threadIdx.x; // gx
	int col = blockIdx.y * blockDim.y + threadIdx.y; // gy
	if (row < NROW && col < NCOL) {
		int i = row * NCOL + col;
		C[i] = A[i] + B[i];
	}
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
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc( (void**)&devMatA, NROW * NCOL * sizeof(float) );
	hipMalloc( (void**)&devMatB, NROW * NCOL * sizeof(float) );
	hipMalloc( (void**)&devMatC, NROW * NCOL * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	hipMemcpy( devMatA, matA, NROW * NCOL * sizeof(float), hipMemcpyHostToDevice );
	hipMemcpy( devMatB, matB, NROW * NCOL * sizeof(float), hipMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((NCOL + dimBlock.x - 1) / dimBlock.x, (NROW + dimBlock.y - 1) / dimBlock.y, 1);
	CUDA_PRINT_CONFIG_2D(NROW, NCOL); // defined in "common.hpp"
	start = steady_clock::now();
	kernel_mat_add <<< dimGrid, dimBlock>>>( devMatC, devMatA, devMatB, NROW, NCOL );
	hipDeviceSynchronize();
	end = steady_clock::now();
	CUDA_CHECK_ERROR();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	// copy to host from device
	hipMemcpy( matC, devMatC, NROW * NCOL * sizeof(float), hipMemcpyDeviceToHost );
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

// HISTORY: mat-add-col-major.cu <-- mat-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 05c-mat-add-col-major.cu,v 1.2 2026/09/05 08:53:57 wayfarecru Exp $";
