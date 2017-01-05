#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "gettime.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__); \
     exit(-1);}cudaThreadSynchronize();} while(0)

int inline n_blocks(int dim, int block_size)
{
	return dim / block_size + ((dim % block_size == 0)? 0 : 1);
}


template<typename Real>
__device__
static void VecVec(Real *x, Real *y, int dim, Real *res)
{
	Real result = 0;
	for (int i = 0; i < dim; i++)
		result += x[i] * y[i];
	*res = result;
}

template<typename Real>
__global__
static void _MatVecVec(Real *x, Real *y, int rows, int cols, Real *out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < rows)
	{
		VecVec(x + i * cols, y + i * cols, cols, out + i);
	}
}

template <typename Real>
__host__
static void MatVecVec(Real *x, Real *y, int rows, int cols, Real *out)
{
	Real *devx;
	Real *devy;
	Real *devo;
	int threadsPerBlock = 256;
	CUDA_CALL(cudaMalloc((void **)&devx, rows * cols * sizeof(Real)));
	CUDA_CALL(cudaMalloc((void **)&devy, rows * cols * sizeof(Real)));
	CUDA_CALL(cudaMalloc((void **)&devo, rows * sizeof(Real)));
	CUDA_CALL(cudaMemcpy(devx, x, rows * cols * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devy, y, rows * cols * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devo, out, rows * sizeof(Real), cudaMemcpyHostToDevice));
	my_time t;
	t.start();
	_MatVecVec<<<n_blocks(rows, threadsPerBlock), threadsPerBlock>>>(devx, devy, rows, cols, devo);
	t.end();
	std::cout << "\ncompute MatVecVec time: " << t.used_time() << "ms" << std::endl;
	CUDA_CALL(cudaMemcpy(out, devo, rows * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(devx));
	CUDA_CALL(cudaFree(devy));
	CUDA_CALL(cudaFree(devo));
}

int main()
{
	float *data;
	data = (float *)malloc(5 * 10 * sizeof(float));
	float *out;
	out = (float *)malloc(5 * sizeof(float));
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 10; j++)
			data[i * 10 + j] = static_cast<float>(i + j);
	for (int i = 0; i < 5; i++)
	{
		std::cout << "\n[ ";
		for (int j = 0; j < 10; j++)
			std::cout << " " << data[i * 10 + j];
		std::cout << " ]" << std::endl;
	}
	MatVecVec(data, data, 5, 10, out);
	std::cout << "\n[ ";
	for (int i = 0; i < 5; i++)
		std::cout << " " << out[i];
	std::cout << " ]" << std::endl;
	return 0;
}


