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

template<typename Real>
__host__
static void _my_cuda_compute_fft(Real *data, int nx, int ny)
{
	my_time t;
	t.start();
	cufftComplex *CompData = (cufftComplex *)malloc(nx * ny * sizeof(cufftComplex));
	for (int i = 0; i < nx; i++)
	{
		CompData[i].x = data[i];
		CompData[i].y = 0;
	}
	t.end();
	std::cout << "malloc CompData time: " << t.used_time() << "ms" << std::endl;
	cufftComplex *devData;
	t.start();
	CUDA_CALL(cudaMalloc((void **)&devData, dim * sizeof(cufftComplex)));
	CUDA_CALL(cudaMemcpy(devData, CompData, dim * sizeof(CompData), cudaMemcpyHostToDevice));
	t.end();
	std::cout << "malloc devData time: " << t.used_time() << "ms" << std::endl;

	cufftHandle plan;
	t.start();
	cufftPlan2d(&plan, nx, ny, CUFFT_C2C, 1);
	t.end();
	std::cout << "malloc create plan time: " << t.used_time() << "ms" << std::endl;
	t.start();
	cufftExecC2C(plan, devData, devData, CUFFT_FORWARD);
	t.end();
	std::cout << "malloc ExecC2C time: " << t.used_time() << "ms" << std::endl;
	cudaDeviceSynchronize();
	CUDA_CALL(cudaMemcpy(CompData, devData, dim * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	for (int i = 0; i < dim / 2; i++)
	{
		data[2 * i] = CompData[i].x;
		data[2 * i + 1] = CompData[i].y;
	}
	data[1] = CompData[dim / 2].x;
	CUDA_CALL(cufftDestroy(plan));
	CUDA_CALL(cudaFree(devData));
	free(CompData);
}

int main()
{
	CUDA_CALL(cudaSetDevice(4));
	float init[2][1];
	init[0][0] = 1;
	init[1][0] = 1;
	_my_cuda_compute_fft(init, 2, 1);
	for (int i = 0; i < 2; i++)
		printf("\ndata[%d]=%lf", i, init[i]);
	float data[80];
	for (int i = 0; i < 80; i++)
		data[i] = static_cast<float>(1);
	long long int start, end;
	start = getSystemTime();
	_my_cuda_compute_fft(data, 80);
	end = getSystemTime();
	printf("compute fft time: %lldms", end - start);
	for (int i = 0;i < 8; i++)
		printf("\ndata[%d]=%lf", i, data[i]);
	return 0;
}


