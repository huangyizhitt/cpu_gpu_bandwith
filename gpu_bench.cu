#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
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

	if(use_cpu == FORWORD && use_gpu == FORWORD) {
		ACCESS_ONCE(gpu_test_status) = HOST_TO_DEVICE;
		pthread_barrier_wait(&g_barrier);
	}

	checkCudaErrors(cudaEventRecord(start, 0));
	while(l--) {
		for(unsigned int i = 0; i < bytes / sizeof(unsigned char); i++) {
			array[i] = value;
		}
	}

	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventSynchronize(stop));

	if(use_cpu == FORWORD && use_gpu == FORWORD) {
		ACCESS_ONCE(gpu_test_status) = HOST_TO_DEVICE_COMPLETE;
		pthread_barrier_wait(&g_barrier);
	}
	
 	checkCudaErrors(cudaEventElapsedTime(&elapse, start, stop));

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

	if(use_cpu == FORWORD && use_gpu == FORWORD) {
		ACCESS_ONCE(gpu_test_status) = DEVICE;
		pthread_barrier_wait(&g_barrier);
	}
	
	checkCudaErrors(cudaEventRecord(start, 0));
	gpu_array_read<<<blocks, threads>>>(array, size);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(start));
	checkCudaErrors(cudaEventSynchronize(stop));

	if(use_cpu == FORWORD && use_gpu == FORWORD) {
		ACCESS_ONCE(gpu_test_status) = DEVICE_COMPLETE;
		pthread_barrier_wait(&g_barrier);
	}

	checkCudaErrors(cudaEventElapsedTime(&elapse, start, stop));

	bandwidth =  ((long)size / MB) / (elapse / 1000);
	
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start));

	return bandwidth;
}


void gpu_bench_benckend(struct bench_config *con)
{
	unsigned int bytes = con->gpu_con->size * sizeof(unsigned char); 
	float bandwidth_trans = 0, bandwidth_access = 0;
	int times = 0;
	unsigned char *array = gpu_array_make_uma(con->gpu_con->size * sizeof(unsigned char));
	
	ACCESS_ONCE(gpu_test_status) = INIT;
	pthread_barrier_wait(&g_barrier);

	while(1) {
		unsigned char value = rand() % 256;
		for(int i = 0; i < con->gpu_con->size; i++)
		{
			array[i] = value;
		}
		bandwidth_trans += test_host_to_device_uma(array, bytes, 1);
//		printf("Host to device: %.3f MiB/s\n", bandwidth);

		bandwidth_access += test_device_access(array, bytes);
//		printf("device access memory: %.3f MiB/s\n", bandwidth);
		times++;
		if(bench_all_thread_done(con))	break;
	}
	
	gpu_array_destroy(array);
	printf("Host to device: %.3f MiB/s, device access memory: %.3f MiB/s\n", bandwidth_trans / times, bandwidth_access / times);
	ACCESS_ONCE(gpu_test_status) = COMPLETE;
}

void gpu_bench_forward(struct bench_config *con)
{
	unsigned int bytes = con->gpu_con->size * sizeof(unsigned char); 
	float bandwidth;
	unsigned char *array = gpu_array_make_uma(con->gpu_con->size * sizeof(unsigned char));

	
	ACCESS_ONCE(gpu_test_status) = INIT;
	if(use_cpu)
		pthread_barrier_wait(&g_barrier);

	
	unsigned char value = rand() % 256;
	for(int i = 0; i < con->gpu_con->size; i++)
	{
		array[i] = value;
	}
	bandwidth = test_host_to_device_uma(array, bytes, 1);
	printf("Host to device: %.3f MiB/s\n", bandwidth);

	bandwidth = test_device_access(array, bytes);
	printf("device access memory: %.3f MiB/s\n", bandwidth);
	
	gpu_array_destroy(array);
	ACCESS_ONCE(gpu_test_status) = COMPLETE;
}


void gpu_bench(struct bench_config *con)
{
	
	unsigned int bytes = con->gpu_con->size * sizeof(unsigned char); 
	float bandwidth;
	ACCESS_ONCE(gpu_test_status) = INIT;

	unsigned char *array = gpu_array_make_uma(con->gpu_con->size * sizeof(unsigned char));
	bandwidth = test_host_to_device_uma(array, bytes, DEFAULT_LOOPS);
	printf("Host to device: %.3f MiB/s\n", bandwidth);

	bandwidth = test_device_access(array, bytes);
	printf("device access memory: %.3f MiB/s\n", bandwidth);

	gpu_array_destroy(array);
	ACCESS_ONCE(gpu_test_status) = COMPLETE;
}

void *gpu_bench_func(void *args)
{
	struct bench_config *con = (struct bench_config *) args;
	
	if(use_gpu == FORWORD && use_cpu != FORWORD) {
		gpu_bench_forward(con);
	}

	if(use_gpu == BENCKEND) {
		gpu_bench_benckend(con);
	}

	if(use_gpu == FORWORD && use_cpu == FORWORD) {
		gpu_bench(con);
	}
	pthread_exit((void *)0);
}

extern "C"
bool gpu_bench_init(struct bench_config *con)
{
	if(use_gpu == UNUSED) return true;

	pthread_attr_init(&con->gpu_con->attr);
	cpu_set_t cpu_info;
	CPU_ZERO(&cpu_info);
	CPU_SET(DEFAULT_GPU_TASK_CPU, &cpu_info);

	if(pthread_attr_setaffinity_np(&con->gpu_con->attr, sizeof(cpu_set_t), &cpu_info)) {
		printf("gpu task set affinity failed");
		goto fail_pthread_create;
	}

	pthread_create(&con->gpu_con->tid, NULL, gpu_bench_func, (void *)con);
	return true;
fail_pthread_create:
	pthread_attr_destroy(&con->gpu_con->attr);
	return false;
}

extern "C"
void gpu_bench_finish(struct bench_config *con)
{
	if(use_gpu == UNUSED) return;

	pthread_join(con->gpu_con->tid, NULL);
	pthread_attr_destroy(&con->gpu_con->attr);
}


