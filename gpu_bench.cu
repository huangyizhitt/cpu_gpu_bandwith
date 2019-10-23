#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
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

	checkCudaErrors(cudaEventRecord(start, 0));
	while(l--) {
		for(unsigned int i = 0; i < bytes / sizeof(unsigned char); i++) {
			array[i] = value;
		}
	}
	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventSynchronize(stop));
	gpu_done = 1;
 	checkCudaErrors(cudaEventElapsedTime(&elapse, start, stop));
	printf("bytes: %d, elapse: %f, loops: %d\n", bytes, elapse, loops);	
	bandwidth =  ((long)bytes * loops / MB) / (elapse / 1000);
	
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));

	return bandwidth;
}

void gpu_array_destroy(unsigned char *array)
{
	cudaFree(array);
}

extern "C"
void gpu_bench(struct config *con)
{
	unsigned int bytes = con->size * sizeof(long);
	float bandwidth;
	unsigned char *array = gpu_array_make_uma(con->size * sizeof(long));
	bandwidth = test_host_to_device_uma(array, bytes, con->loops);
	printf("Host to device: %.3f MiB/s\n", bandwidth);
	gpu_array_destroy(array);
}

void test()
{

}
