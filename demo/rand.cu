/*
 * This program uses the device CURAND API to calculate what 
 * proportion of pseudo-random ints have low bit set.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>


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
    unsigned int *devResults, *hostResults;
	curandState *devStates;
	hostResults = (unsigned int *)calloc(1, sizeof(unsigned int));
	CUDA_CALL(cudaMalloc((void **)&devResults, sizeof(unsigned int)));
	CUDA_CALL(cudaMemset(devResults, 0, sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&devStates, sizeof(curandState)));
	setup_kernel<<<1, 1>>>(devStates);
	generate_kernel<<<1, 1>>>(devStates, devResults);
	CUDA_CALL(cudaMemcpy(hostResults, devResults, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\nrand number is %d\n", hostResults);

    /* Cleanup */
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_mtgp_example PASSED\n");
    return EXIT_SUCCESS;
}


