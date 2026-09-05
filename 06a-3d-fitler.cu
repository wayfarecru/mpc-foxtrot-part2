#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

// image size
dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth

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
__host__ void kernel_cpu_3d_filter(int z, int y, int x, float* C, const float* A, const float* B) {
	int i = (z * dimImage.y + y) * dimImage.x + x; // convert to 1D index
	C[i] = A[i] * B[i];
}

// CUDA kernel function
__global__ void kernel_3d_filter(float* C, const float* A, const float* B, const dim3 dimImage) {
	int gz = blockIdx.z * blockDim.z + threadIdx.z;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gx < dimImage.x && gy < dimImage.y && gz < dimImage.z) {
		int i = (gz * dimImage.y + gy) * dimImage.x + gx; // converted to 1D index
		C[i] = A[i] * B[i];
	}
}


int main(const int argc, const char* argv[]) {
	srand(time(NULL));
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		dimImage.x = dimImage.y = dimImage.z = procArg( argv[0], argv[1], 32, 1024 );
		break;
	case 4:
		dimImage.z = procArg( argv[0], argv[1], 32, 1024 );
		dimImage.y = procArg( argv[0], argv[2], 32, 1024 );
		dimImage.x = procArg( argv[0], argv[3], 32, 1024 );
		break;
	default:
		printf("usage: %s [dim.z [dim.y dim.x]]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// CPU part: set values
	printf("generating %d x %d x %d random numbers: ", dimImage.z, dimImage.y, dimImage.x);
	fflush(stdout);
	try {
		matA = new float[dimImage.z * dimImage.y * dimImage.x];
		matB = new float[dimImage.z * dimImage.y * dimImage.x];
		matC = new float[dimImage.z * dimImage.y * dimImage.x];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	setRand( dimImage.z * dimImage.y * dimImage.x, matA );
	setRand( dimImage.z * dimImage.y * dimImage.x, matB );
	printf("done\n");
#if 0
	printf("A[0][%d][%d] =\n", dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, matA);
	printf("A[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &matA[(dimImage.z - 1) * dimImage.y * dimImage.x]);
	printf("B[0][%d][%d] =\n", dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, matB);
	printf("B[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &matB[(dimImage.z - 1) * dimImage.y * dimImage.x]);
#endif
	fflush(stdout);
	// set host pitch
	hostPitchA = dimImage.x * sizeof(float); // host side: packed compactly
	hostPitchB = dimImage.x * sizeof(float); // host side: packed compactly
	hostPitchC = dimImage.x * sizeof(float); // host side: packed compactly
	printf("host pitch = %zd (byte), host stride = %u (element)\n", hostPitchA, dimImage.x );
	fflush(stdout);
	// CPU kernel execution
	steady_clock::time_point start = steady_clock::now();
	for (int z = 0; z < dimImage.z; ++z) {
		for (int y = 0; y < dimImage.y; ++y) {
			for (int x = 0; x < dimImage.x; ++x) {
				kernel_cpu_3d_filter(z, y, x, matC, matA, matB);
			}
		}
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
#if 0
	printf("CPU's C[0][%d][%d] =\n", dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, matC);
#endif
	printf("CPU's C[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &matC[(dimImage.z - 1) * dimImage.y * dimImage.x]);
	memset(matC, 0, dimImage.z * dimImage.y * dimImage.x * sizeof(float));
	fflush(stdout);
	// GPU part: warmup
	CUDA_WARM_UP();
	// GPU part: memory allocation
	cudaMalloc( (void**)&devMatA, dimImage.z * dimImage.y * dimImage.x * sizeof(float) );
	cudaMalloc( (void**)&devMatB, dimImage.z * dimImage.y * dimImage.x * sizeof(float) );
	cudaMalloc( (void**)&devMatC, dimImage.z * dimImage.y * dimImage.x * sizeof(float) );
	CUDA_CHECK_ERROR();
	printf("device pitch = %zd (byte), device stride = %u (element)\n", dimImage.x * sizeof(float), dimImage.x );
	fflush(stdout);
	// copy to device from host
	cudaMemcpy( devMatA, matA, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( devMatB, matB, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(8, 8, 8);
	dim3 dimGrid((dimImage.x + dimBlock.x - 1) / dimBlock.x,
				(dimImage.y + dimBlock.y - 1) / dimBlock.y,
				(dimImage.z + dimBlock.z - 1) / dimBlock.z);
	CUDA_PRINT_CONFIG_3D(dimImage.x, dimImage.y, dimImage.z);
	start = steady_clock::now();
	kernel_3d_filter <<< dimGrid, dimBlock>>>( devMatC, devMatA, devMatB, dimImage );
	cudaDeviceSynchronize();
	end = steady_clock::now();
	CUDA_CHECK_ERROR();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	fflush(stdout);
	// copy to host from device
	cudaMemcpy( matC, devMatC, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
#if 0
	printf("GPU's C[0][%d][%d] =\n", dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, matC);
#endif
	printf("GPU's C[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &matC[(dimImage.z - 1) * dimImage.y * dimImage.x]);
	fflush(stdout);
	// free device memory
	cudaFree( devMatA );
	cudaFree( devMatB );
	cudaFree( devMatC );
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	// done
	return 0;
}

// HISTORY: 3d-filter.cu
static const char rcsid[] __attribute__((used)) = "$Id: 06a-3d-fitler.cu,v 1.2 2026/09/05 09:40:16 wayfarecru Exp $";
