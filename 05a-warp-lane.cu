#include <cstdio>
#include "./common.hpp"

__device__ unsigned lane_id(void) {
#if defined(__NVCC__)
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
#endif
#if defined(__HIP__)
	unsigned ret = threadIdx.x % warpSize;
	return ret;
#endif
}

__device__ unsigned warp_id(void) {
#if defined(__NVCC__)
	// this may not be equal to threadIdx.x / 32
	unsigned ret;
	asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
	return ret;
#endif
#if defined(__HIP__)
	unsigned ret = threadIdx.x / warpSize;
	return ret;
#endif
}

__global__ void kernel_warp_lane( void ) {
	unsigned warpid = warp_id();
	unsigned laneid = lane_id();
	if (warpid == 0) {
		printf("lane=%2u threadIdx.x=%2d threadIdx.y=%2d blockIdx.x=%2d blockIdx.y=%2d\n",
		       laneid, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
	}
}

int main( void ) {
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(2, 2, 1);
	kernel_warp_lane <<< dimGrid, dimBlock>>>();
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: warp-lane.cu
static const char rcsid[] __attribute__((used)) = "$Id: 05a-warp-lane.cu,v 1.2 2026/09/05 08:53:57 wayfarecru Exp $";
