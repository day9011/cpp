#include <cuda_runtime.h>
#include <cfloat>
#include <cufft.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "my-cuda-function-kernel-ansi.h"

#define CUDA_CALL(ret) \
{\
  if((ret) != cudaSuccess) { \
  printf("Error at %s:%d\n", __FILE__, __LINE__); \
  printf("Error code %s", cudaGetErrorString(ret)); \
  exit(-1); \
  } \
  cudaThreadSynchronize(); \
}

template<typename Real>
__device__
static void __insert_sort(Real *__first, Real *__last)
{
	if (__first == __last)
		return;
	Real *p;
	for (Real *iter = __first + 1; iter != __last; ++iter)
	{
		Real tmp = *iter;
		for (p = iter; p != __first && tmp < *(p - 1); --p)
			*p = *(p - 1);
		*p = tmp;
	}
}

//binary sort, get top N numbers
template<typename Real>
__device__
static Real* __partition(Real *__first, Real *__last, Real __pivot)
{
	while(true)
	{
		while (*__first < __pivot)
			++__first;
		--__last;
		while (__pivot < *__last)
			--__last;
		if(!(__first < __last))
			return __first;
		//swap two number
		{
			*__first += *__last;
			*__last = *__first - *__last;
			*__first -= *__last;
		}
	}
}

template<typename Real>
__device__
static void _partition(Real *__first, Real *__nth, Real *__last)
{
	while (__last - __first > 3)
	{
		Real *__cut = __partition(__first, __last, *(__first + (__last - __first) / 2));
		if (__cut <= __nth)
			__first = __cut;
		else
			__last = __cut;
	}
	__insert_sort(__first, __last);
}

template<typename Real>
__global__
static void _gmm_select(Real *data, int32_cuda rows, int32_cuda cols, int32_cuda num_gselect, int32_cuda *gmm_selected)
{
	int32_cuda row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < rows)
	{
		Real *dataCopy = (Real *)malloc(cols * sizeof(Real));
		for (int32_cuda n = 0; n < cols; n++)
			dataCopy[n] = data[row * cols + n];
		_partition(dataCopy, dataCopy + cols - num_gselect, dataCopy + cols);
		printf("\nI can sort data\n");
		Real thresh = dataCopy[cols - num_gselect];
		for (int32_cuda j = 0; j < cols; j++)
			if (*(data + row * cols + j) >= thresh)
			{
				*(gmm_selected + row * num_gselect) = j;
				gmm_selected++;
			}
		__syncthreads();
		free(dataCopy);
	}
}

template<typename Real>
__host__
static void _my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, Real *data, int32_cuda rows, int32_cuda cols, int32_cuda num_gselect, int32_cuda *gmm_out)
{
	int32_cuda *selected_gauss;
	CUDA_CALL(cudaMalloc((void **)&selected_gauss, rows * num_gselect * sizeof(int32_cuda)));
	_gmm_select<<<Gr, Bl>>>(data, rows, cols, num_gselect, selected_gauss);
	CUDA_CALL(cudaMemcpy(gmm_out, selected_gauss, rows * num_gselect * sizeof(int32_cuda), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(selected_gauss));
}


template<typename Real>
__host__
static void _my_cuda_compute_fft(Real *data, int32_cuda dim)
{
	cufftComplex *CompData = (cufftComplex *)malloc(dim * sizeof(cufftComplex));
	for (int32_cuda i = 0; i < dim; i++)
	{
		CompData[i].x = data[i];
		CompData[i].y = 0;
	}
	cufftComplex *devData;
	CUDA_CALL(cudaMalloc((void **)&devData, dim * sizeof(cufftComplex)));
	CUDA_CALL(cudaMemcpy(devData, CompData, dim * sizeof(CompData), cudaMemcpyHostToDevice));

	cufftHandle plan;
	cufftPlan1d(&plan, dim, CUFFT_C2C, 1);
	cufftExecC2C(plan, devData, devData, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	CUDA_CALL(cudaMemcpy(CompData, devData, dim * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	for (int32_cuda i = 0; i < dim / 2; i++)
	{
		data[2 * i] = CompData[i].x;
		data[2 * i + 1] = CompData[i].y;
	}
	data[1] = CompData[dim / 2].x;
	CUDA_CALL(cudaFree(devData));
	free(CompData);
}


//define in my-cuda-function-kernel-ansi.h
void _F_my_cuda_compute_fft(float *data, int32_cuda dim)
{
	_my_cuda_compute_fft(data, dim);
}

void _D_my_cuda_compute_fft(double *data, int32_cuda dim)
{
	_my_cuda_compute_fft(data, dim);
}

void _F_my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, float *data, int32_cuda rows, int32_cuda cols, int32_cuda num_gselect, int32_cuda *gmm_out)
{
	_my_cuda_gmm_select(Gr, Bl, data, rows, cols, num_gselect, gmm_out);
}

void _D_my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, double *data, int32_cuda rows, int32_cuda cols, int32_cuda num_gselect, int32_cuda *gmm_out)
{
	_my_cuda_gmm_select(Gr, Bl, data, rows, cols, num_gselect, gmm_out);
}

