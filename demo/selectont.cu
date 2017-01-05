#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "gettime.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
     printf("Error at %s:%d\n",__FILE__,__LINE__); \
     exit(-1);}cudaThreadSynchronize();} while(0)

void test() {
	printf("\nfor a test");
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
		++__first;
	}
}

template<typename Real>
__device__
static void _partition(Real *__first, Real *__nth, Real *__last)
{
	while(__last - __first > 3)
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
static void _gmm_select(Real *data, Real *dataCopy, int rows, int cols, int num_ceps, int *gmm_selected)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < rows)
	{
		//copy data
		//sort copy data to get greater 20 numbers;
		_partition(dataCopy + row * cols, dataCopy + row * cols + cols - num_ceps, dataCopy + row * cols + cols);
		Real thresh = dataCopy[row * cols + cols - num_ceps];
		printf("thread %d thresh is %f\toffset is %d\n", row, thresh, row * cols + cols - num_ceps);
		for (int j = 0; j < cols; j++)
			if(*(data + row * cols + j) >= thresh)
			{
				*(gmm_selected + row * num_ceps) = j;
				gmm_selected++;
			}
		__syncthreads();
		free(dataCopy);
	}
}


template<typename Real>
__host__
static int *_my_cuda_gmm_select(Real *data, int rows, int cols, int num_ceps)
{
	int threadsPerBlock = 256;
	int blockPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
	int *selected_gauss;
	int *host_selected_gauss = (int *)malloc(rows * num_ceps * sizeof(int));
	Real *devdata;
	Real *copydata;
	CUDA_CALL(cudaMalloc((void **)&selected_gauss, rows * num_ceps * sizeof(int)));
	CUDA_CALL(cudaMalloc((void **)&devdata, rows * cols * sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&copydata, rows * cols * sizeof(float)));
	my_time t;
	t.start();
	CUDA_CALL(cudaMemcpy(devdata, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(copydata, devdata, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
	_gmm_select<<<blockPerGrid, threadsPerBlock>>>(devdata, copydata, rows, cols, num_ceps, selected_gauss);
//	_gmm_select<<<1, 1>>>(devdata, dim, num_ceps, selected_gauss);
	CUDA_CALL(cudaMemcpy(host_selected_gauss, selected_gauss, rows * num_ceps * sizeof(int), cudaMemcpyDeviceToHost));
	t.end();
	printf("gpu gmm select used time is:%lld\n", t.used_time());
	CUDA_CALL(cudaFree(selected_gauss));
	CUDA_CALL(cudaFree(devdata));
	return host_selected_gauss;
}

int main()
{
	float *data;
	data = (float *)malloc(220 * 204 * sizeof(float));
	for (int i = 0; i < 220 * 204; i++)
		*(data + i) = static_cast<float>(i);
	int num_gselect = 20;
	int *p = _my_cuda_gmm_select(data, 220, 204, num_gselect);
	return 0;
}


