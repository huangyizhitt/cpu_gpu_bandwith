#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "config.h"
#include "bench.h"
#include "gpu_bench.h"
#include "helper_cuda.h"


unsigned char *gpu_array_make_uma(unsigned int bytes)
{
	unsigned char *array;

	checkCudaErrors(cudaMallocManaged(&array, bytes));

	return array;
}

float test_host_to_device_uma(unsigned char *array, unsigned int bytes, int loops)
{
	float elapse = 0;
	float bandwidth = 0;
	cudaEvent_t start, stop;
	int l = loops;        
	unsigned char value = rand() % 256;
	
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	gpu_test_status = HOST_TO_DEVICE;
	pthread_barrier_wait(&gpu_barrier);

	checkCudaErrors(cudaEventRecord(start, 0));
	while(l--) {
		for(unsigned int i = 0; i < bytes / sizeof(unsigned char); i++) {
			array[i] = value;
		}
	}

	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventSynchronize(stop));

	gpu_test_status = HOST_TO_DEVICE_COMPLETE;
	pthread_barrier_wait(&gpu_barrier);
	
 	checkCudaErrors(cudaEventElapsedTime(&elapse, start, stop));

	printf("elapse is %fms, bytes: %u\n", elapse, bytes * loops);
	bandwidth =  ((long)bytes * loops / MB) / (elapse / 1000);
	
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));

	return bandwidth;
}

void gpu_array_destroy(unsigned char *array)
{
	cudaFree(array);
}

__global__ void gpu_array_read(unsigned char *array, unsigned int size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned char value;
	if(x < size) {
		value = array[x];
	}
}

float test_device_access(unsigned char *array, unsigned int size)
{
	int threads = 1024;
	int blocks = (size - 1) / threads + 1;
	float elapse = 0;
	float bandwidth = 0;
	cudaEvent_t start, stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	gpu_test_status = DEVICE;
	pthread_barrier_wait(&gpu_barrier);

	checkCudaErrors(cudaEventRecord(start, 0));
	gpu_array_read<<<blocks, threads>>>(array, size);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventSynchronize(stop));
	gpu_test_status = DEVICE_COMPLETE;
	pthread_barrier_wait(&gpu_barrier);
	
	checkCudaErrors(cudaEventElapsedTime(&elapse, start, stop));

	printf("elapse is %fms\n", elapse);
	bandwidth =  ((long)size / MB) / (elapse / 1000);
	
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));

	return bandwidth;
}

extern "C"
void gpu_bench(struct config *con)
{
	unsigned int bytes = con->gpu_array_size * sizeof(unsigned char); 
	float bandwidth;
	gpu_test_status = INIT;
	pthread_barrier_wait(&gpu_barrier);
	unsigned char *array = gpu_array_make_uma(con->gpu_array_size * sizeof(unsigned char));
	bandwidth = test_host_to_device_uma(array, bytes, con->loops);
	printf("Host to device: %.3f MiB/s\n", bandwidth);

	bandwidth = test_device_access(array, bytes);
	printf("device access memory: %.3f MiB/s\n", bandwidth);

	gpu_array_destroy(array);
	gpu_test_status = COMPLETE;
	pthread_barrier_wait(&gpu_barrier);
}


