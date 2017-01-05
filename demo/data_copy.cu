/*
 * This program uses the device CURAND API to calculate what 
 * proportion of pseudo-random ints have low bit set.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "gettime.h"

#define N 2047
#define M 60
#define W 400

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state,
								unsigned int *result)
{
	int id = threadIdx.x + blockIdx.x;
	unsigned int x;
	curandState localState = state[id];
	x = curand(&localState);
	result[id] = x;
}

int main(int argc, char *argv[])
{
	my_time t;
    double *devResults, *hostResults;
	hostResults = (double *)calloc(49128000, sizeof(double));
	CUDA_CALL(cudaMalloc((void **)&devResults, 49128000 * sizeof(double)));
	t.start();
	CUDA_CALL(cudaMemcpy(devResults, hostResults, 49128000 * sizeof(double), cudaMemcpyHostToDevice));
	t.end();
	printf("copy 3D data to device cost time: %lldms\n", t.used_time());

	double ***host3DData = (double ***)calloc(N * M * W, sizeof(double));
	cudaExtent extent = make_cudaExtent(M * sizeof(double), N, W);
	cudaPitchedPtr devPitchedPtr;
	CUDA_CALL(cudaMalloc3D(&devPitchedPtr, extent));
	cudaMemcpy3DParms p = {0};
	p.srcPtr.ptr = host3DData;
	p.srcPtr.pitch = M * sizeof(double);
	p.srcPtr.xsize = M;
	p.srcPtr.ysize = N;
	p.dstPtr.ptr = devPitchedPtr.ptr;
	p.dstPtr.pitch = devPitchedPtr.pitch;
	p.dstPtr.xsize = M;
	p.dstPtr.ysize = N;
	p.extent.width = M * sizeof(double);
	p.extent.height = N;
	p.extent.depth = W;
	p.kind = cudaMemcpyHostToDevice;
	t.start();
	CUDA_CALL(cudaMemcpy3D(&p));
	t.end();
	printf("copy 3D data to device by Memcpy3D cost time: %lldms\n", t.used_time());
    /* Cleanup */
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("\n^^^^ kernel_mtgp_example PASSED\n");
    return EXIT_SUCCESS;
}


