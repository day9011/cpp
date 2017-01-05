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

#define PI 3.1415926535897932384626433832795
my_time t;

int inline n_blocks(int dim, int block_size)
{
	return dim / block_size + ((dim % block_size == 0)? 0 : 1);
}

template<typename Real>
__global__
static void _my_addvat2(Real *AP, Real *x, int dim, Real alpha)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i <= j && j < dim)
	{
		 AP[j * (j + 1) / 2 + i] = x[j] * x[i] * alpha;
		 __syncthreads();
	}
}

template<typename Real>
__global__
static void _scale(Real *A, int dim, Real alpha)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dim)
	{
		A[i] = A[i] * alpha;
	}
}

template<typename Real>
__global__
static void _my_addvat3(Real *AP, int numA, Real *x, int cols, Real alpha)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int stride = cols * (cols + 1) / 2;
	if (i < cols && j < cols && k < numA)
	{
		if(i <= j)
		{
			AP[k * stride + j * (j + 1) / 2 + i] = x[k * cols + j] * x[k * cols + i] * alpha;
		}
		Real scal = 0.1;
		_scale<<<stride / 256 + 1, 256>>>(AP, stride, scal);
		__syncthreads();
	 }
}

template<typename Real>
__host__ void scale(Real *A, int dim, Real alpha)
{
	Real *devA;
	CUDA_CALL(cudaMalloc((void **)&devA, dim * sizeof(Real)));
	CUDA_CALL(cudaMemcpy(devA, A, dim * sizeof(Real), cudaMemcpyHostToDevice));
	_scale<<<n_blocks(dim, 256), 256>>>(devA, dim, float(0.5));
	CUDA_CALL(cudaMemcpy(A, devA, dim * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(devA));
}


template<typename Real>
__host__
static void addvat2(Real *A, int dimA, Real *x, int dimx, Real alpha)
{
	Real *devA;
	Real *devx;
	CUDA_CALL(cudaMalloc((void **)&devA, dimA * sizeof(Real)));
	CUDA_CALL(cudaMalloc((void **)&devx, dimx * sizeof(Real)));
	CUDA_CALL(cudaMemcpy(devA, A, dimA * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devx, x, dimx * sizeof(Real), cudaMemcpyHostToDevice));
	t.start();
	_my_addvat2<<<dim3(n_blocks(dimx * dimx, 16)), dim3(16, 16)>>>(devA, devx, dimx, alpha);
	t.end();
	printf("Compute addvat2 time: %lldms\n", t.used_time());
	CUDA_CALL(cudaMemcpy(A, devA, dimA * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(devA));
	CUDA_CALL(cudaFree(devx));
}

template<typename Real>
__host__
static void addvat3(Real *A,int numA, int dimA, Real *x, int rowsx, int colsx, Real alpha)
{
	Real *devA;
	Real *devx;
	CUDA_CALL(cudaMalloc((void **)&devA, numA * dimA * sizeof(Real)));
	CUDA_CALL(cudaMalloc((void **)&devx, rowsx * colsx * sizeof(Real)));
	CUDA_CALL(cudaMemcpy(devA, A, numA * dimA * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devx, x, rowsx * colsx * sizeof(Real), cudaMemcpyHostToDevice));
	t.start();
	_my_addvat3<<<dim3(n_blocks(numA * dimA, 512)), dim3(8, 8, 8)>>>(devA, numA, devx, colsx, alpha);
	t.end();
	printf("Compute addvat3 time: %lldms\n", t.used_time());
	CUDA_CALL(cudaMemcpy(A, devA, numA * dimA * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(devA));
	CUDA_CALL(cudaFree(devx));
}

int main()
{
	t.start();
	CUDA_CALL(cudaSetDevice(4));
	t.end();
	printf("Init Gpu time: %lldms\n", t.used_time());
//	compute addvat3
	int NA = 5;
	int NUMA = 5;
	float *A;
	int lensq = NA * (NA + 1) / 2;
	A = (float *)malloc(NUMA * lensq * sizeof(float));
	for (int i = 0; i < NUMA * lensq;i++)
		A[i] = 0;
	float *x;
	x = (float *)malloc(NA * NUMA * sizeof(float));
	for (int i = 0; i < NUMA; i++)
		for (int j = 0; j < NA; j++)
			x[i * NA + j] = float(j + 1);
	addvat3(A, NUMA, lensq, x, NUMA, NA, float(1.0));
	scale(A, NUMA * lensq, float(0.5));
	for (int i = 0; i < NUMA; i++)
	{
		std::cout << i << " line:";
		for (int j = 0; j < lensq; j++)
			std::cout << " " << A[i * lensq + j];
		std::cout << std::endl;
	}
	//compute addvat2
	float *B;
	int N = 2600;
	int lensqB = N * (N + 1) / 2;
	B = (float *)malloc(lensqB * sizeof(float));
	for (int i = 0; i < lensqB;i++)
		B[i] = 0;
	float *y;
	y = (float *)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++)
		y[i] = float(i + 1);
	addvat2(B, lensqB, y, N, float(1.0));
	/* std::cout << "data:"; */
	/* for (int j = 0; j < lensq; j++) */
	/*     std::cout << " " << B[j]; */
	/* std::cout << std::endl; */
	return 0;
}


