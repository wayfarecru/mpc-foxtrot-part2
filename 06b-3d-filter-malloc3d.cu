#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

// image size
dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth

float* rawPtrA = nullptr;
float* rawPtrB = nullptr;
float* rawPtrC = nullptr;
struct cudaPitchedPtr ptrA = { 0 }; // pitch, ptr, xsize, ysize
struct cudaPitchedPtr ptrB = { 0 };
struct cudaPitchedPtr ptrC = { 0 };

struct cudaPitchedPtr ptrDevA = { 0 }; // pitch, ptr, xsize, ysize
struct cudaPitchedPtr ptrDevB = { 0 };
struct cudaPitchedPtr ptrDevC = { 0 };


// CPU kernel function
__host__ void kernel_cpu_3d_filter(int z, int y, int x, float* C, const float* A, const float* B) {
	int i = (z * dimImage.y + y) * dimImage.x + x; // convert to 1D index
	C[i] = A[i] * B[i];
}

// CUDA kernel function
__global__ void kernel_3d_filter(struct cudaPitchedPtr ptrDevC,
                                 struct cudaPitchedPtr ptrDevA, struct cudaPitchedPtr ptrDevB,
                                 const dim3 dimImage ) {
	int gz = blockIdx.z * blockDim.z + threadIdx.z;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gy < dimImage.y && gz < dimImage.z) {
		float* rawPtrC = (float*)((char*)ptrDevC.ptr + (gz * ptrDevC.ysize + gy) * ptrDevC.pitch + gx * sizeof(float));
		float* rawPtrA = (float*)((char*)ptrDevA.ptr + (gz * ptrDevA.ysize + gy) * ptrDevA.pitch + gx * sizeof(float));
		float* rawPtrB = (float*)((char*)ptrDevB.ptr + (gz * ptrDevB.ysize + gy) * ptrDevB.pitch + gx * sizeof(float));
		*rawPtrC = *rawPtrA * *rawPtrB;
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
		rawPtrA = new float[dimImage.z * dimImage.y * dimImage.x];
		rawPtrB = new float[dimImage.z * dimImage.y * dimImage.x];
		rawPtrC = new float[dimImage.z * dimImage.y * dimImage.x];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	setRand( dimImage.z * dimImage.y * dimImage.x, rawPtrA );
	setRand( dimImage.z * dimImage.y * dimImage.x, rawPtrB );
	printf("done\n");
#if 0
	printf("A[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &rawPtrA[(dimImage.z - 1) * dimImage.y * dimImage.x]);
	printf("B[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &rawPtrB[(dimImage.z - 1) * dimImage.y * dimImage.x]);
#endif
	fflush(stdout);
	// set host pitch
	int hostPitch = dimImage.x * sizeof(float);
	ptrA = make_cudaPitchedPtr(rawPtrA, hostPitch, dimImage.x, dimImage.y);
	ptrB = make_cudaPitchedPtr(rawPtrB, hostPitch, dimImage.x, dimImage.y);
	ptrC = make_cudaPitchedPtr(rawPtrC, hostPitch, dimImage.x, dimImage.y);
	printf("host pitch = %d (byte), host stride = %u (element)\n", hostPitch, dimImage.x );
	fflush(stdout);
	// CPU kernel execution
	steady_clock::time_point start = steady_clock::now();
	for (int z = 0; z < dimImage.z; ++z) {
		for (int y = 0; y < dimImage.y; ++y) {
			for (int x = 0; x < dimImage.x; ++x) {
				kernel_cpu_3d_filter(z, y, x, rawPtrC, rawPtrA, rawPtrB);
			}
		}
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's C[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &rawPtrC[(dimImage.z - 1) * dimImage.y * dimImage.x]);
	memset(rawPtrC, 0, dimImage.z * dimImage.y * dimImage.x * sizeof(float));
	fflush(stdout);
	// GPU part: warmup
	CUDA_WARM_UP();
	// GPU part: pre-calculate the kernel dimensions
	dim3 dimBlock(8, 8, 8);
	dim3 dimGrid((dimImage.x + dimBlock.x - 1) / dimBlock.x,
				(dimImage.y + dimBlock.y - 1) / dimBlock.y,
				(dimImage.z + dimBlock.z - 1) / dimBlock.z);
	CUDA_PRINT_CONFIG_3D(dimImage.x, dimImage.y, dimImage.z);
	dim3 numThread(dimBlock.x * dimGrid.x, dimBlock.y * dimGrid.y, dimBlock.z * dimGrid.z);
	// GPU part: memory allocation
	struct cudaExtent extentInByte = make_cudaExtent( dimImage.x * sizeof(float), dimImage.y, dimImage.z );
	cudaMalloc3D( &ptrDevA, extentInByte );
	cudaMalloc3D( &ptrDevB, extentInByte );
	cudaMalloc3D( &ptrDevC, extentInByte );
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// copy to device from host
	struct cudaPos posOrg = make_cudaPos( 0, 0, 0 );
	struct cudaMemcpy3DParms paramA = { 0 };
	struct cudaMemcpy3DParms paramB = { 0 };
	paramA.srcPos = posOrg;
	paramA.srcPtr = ptrA;
	paramA.dstPos = posOrg;
	paramA.dstPtr = ptrDevA;
	paramA.extent = extentInByte;
	paramA.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D( &paramA );
	paramB.srcPos = posOrg;
	paramB.srcPtr = ptrB;
	paramB.dstPos = posOrg;
	paramB.dstPtr = ptrDevB;
	paramB.extent = extentInByte;
	paramB.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D( &paramB );
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// CUDA kernel launch
	int stride = ptrDevA.pitch / sizeof(float);
	printf("device pitch = %zd (byte), device stride = %u (element)\n", ptrDevA.pitch, stride);
	start = steady_clock::now();
	kernel_3d_filter <<< dimGrid, dimBlock>>>( ptrDevC, ptrDevA, ptrDevB, dimImage );
	cudaDeviceSynchronize();
	end = steady_clock::now();
	CUDA_CHECK_ERROR();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	fflush(stdout);
	// copy to host from device
	struct cudaMemcpy3DParms paramC = { 0 };
	paramC.srcPos = posOrg;
	paramC.srcPtr = ptrDevC;
	paramC.dstPos = posOrg;
	paramC.dstPtr = ptrC;
	paramC.extent = extentInByte;
	paramC.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D( &paramC );
	CUDA_CHECK_ERROR();
#if 0
	printf("GPU's C[0][%d][%d] =\n", dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, rawPtrC);
#endif
	printf("GPU's C[%d][%d][%d] =\n", dimImage.z - 1, dimImage.y, dimImage.x);
	printMat(dimImage.y, dimImage.x, &rawPtrC[(dimImage.z - 1) * dimImage.y * dimImage.x]);
	fflush(stdout);
	// free device memory
	cudaFree( ptrDevA.ptr );
	cudaFree( ptrDevB.ptr );
	cudaFree( ptrDevC.ptr );
	CUDA_CHECK_ERROR();
	fflush(stdout);
	// cleaning
	delete[] rawPtrA;
	delete[] rawPtrB;
	delete[] rawPtrC;
	// done
	return 0;
}

// HISTORY: 3d-filter-malloc3d <-- 3d-filter.cu
static const char rcsid[] __attribute__((used)) = "$Id: 06b-3d-filter-malloc3d.cu,v 1.2 2026/09/05 09:40:16 wayfarecru Exp $";
