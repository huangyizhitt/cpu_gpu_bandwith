#define _GNU_SOURCE

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include "bench.h"
#include "cpu_bench.h"


pthread_barrier_t barrier;				//barrier of memory bandwith test for threads

#define TEST_MEMCPY(d, s, bytes, block_size)		\
{													\
	char *src, *dst;								\
	size_t  remain;									\
	src = s;										\
	dst = d;										\
	for(remain = bytes; remain >= block_size; remain -= block_size, src += block_size) {	\
		dst = memcpy(dst, src, block_size);													\
	}																						\
	if(remain) {																			\
		dst = memcpy(dst, src, remain);														\
	}																						\
}

static inline void test_memcpy(char *d, char *s, size_t bytes, int block_size)
{
	
}

static void test_memcpy_loops(char *d, char *s, size_t bytes, int block_size, int loops)
{
	while(loops--) {
		TEST_MEMCPY(d, s, bytes, block_size);
	}
}

static void test_sequence_write(long *array, size_t bytes, int loops)
{
	long *start = array, *p;
	long *end = start + bytes / sizeof(long);
	long value = rand();
	
	while(loops--) {
		p = start;
		while(p != end) {
			*p++ = value;
		}
	}
}

static void *cpu_co_gpu_worker(void *arg)
{
	struct timeval start, end;
	struct cpu_bench_arg *data = (struct cpu_bench_arg*)arg;
	unsigned long long bytes = data->size * sizeof(long);
	int block_size = data->block_size;
	unsigned long long times = 0;
	double elapse;

	long *a = bench_generate_test_array(data->size);
	if(!a) {
		printf("Fail to generate test array a\n");
		goto fail_a;
	}
	
	long *b = bench_generate_test_array(data->size);
	if(!b) {
		printf("Fail to generate test array b\n");
		goto fail_b;
	}

	pthread_barrier_wait(&barrier);
	gettimeofday(&start, NULL);
	while(1) {
		if(gpu_done) break;
		TEST_MEMCPY((char *)b, (char *)a, bytes, block_size);
		times++;
	}
	gettimeofday(&end, NULL);
	elapse=((double)(end.tv_sec * 1000000 - start.tv_sec * 1000000 + 
			end.tv_usec - start.tv_usec))/1000000;
	elapse /= times;
	return NULL;

fail_b:
	free(a);
fail_a:
	pthread_exit((void *)1);

}

static void *cpu_bench_worker(void *arg)
{
	struct timeval start, end;
	double elapse = 0;
	struct cpu_bench_arg *data = (struct cpu_bench_arg*)arg;
	unsigned long long bytes = data->size * sizeof(long);
	int block_size = data->block_size;
	int loops = data->loops;
	
	long *a = bench_generate_test_array(data->size);
	if(!a) {
		printf("Fail to generate test array a\n");
		goto fail_a;
	}
	
	long *b = bench_generate_test_array(data->size);
	if(!b) {
		printf("Fail to generate test array b\n");
		goto fail_b;
	}

	
	pthread_barrier_wait(&barrier);
	
	gettimeofday(&start, NULL);
	
	test_memcpy_loops((char *)b, (char *)a, bytes, block_size, loops);
//	test_sequence_write(a, bytes, loops);

	gettimeofday(&end, NULL);
	elapse=((double)(end.tv_sec * 1000000 - start.tv_sec * 1000000 + 
			end.tv_usec - start.tv_usec))/1000000;
	elapse /= loops;
	
	bench_print_out(data->core, data->thread, elapse, (double)bytes / MB);
			
	free(b);
	free(a);
	pthread_exit((void *)0);
	return NULL;
fail_b:
	free(a);
fail_a:
	pthread_exit((void *)1);
}

int cpu_bench_init(struct cpu_bench *bench, struct config *con)
{
	int cpu_it, thread_it, i, j;
	int cpu_cores = con->cores;
	int threads_per_core = con->threads_per_core;
	int loops = con->loops;
	int threads = cpu_cores * threads_per_core; 

	pthread_barrier_init(&barrier, NULL, threads);
	
	bench->thread = (struct cpu_bench_thread **)malloc(sizeof(struct cpu_bench_thread *) * cpu_cores);
	if(!bench->thread) {
		printf("Fail to alloc cpu vector!\n");
		goto fail_cpus;
	}

	for(cpu_it = 0; cpu_it < cpu_cores; cpu_it++) {

		bench->thread[cpu_it] = (struct cpu_bench_thread *)malloc(sizeof(struct cpu_bench_thread) * threads_per_core);
		if(!bench->thread[cpu_it]) {			
			printf("Fail to alloc thread vector!\n");
			goto fail_threads;
		}

		for(thread_it = 0; thread_it < threads_per_core; thread_it++) {
			//Init thread argument
			bench->thread[cpu_it][thread_it].arg.loops = loops;
			bench->thread[cpu_it][thread_it].arg.size = con->size;
			bench->thread[cpu_it][thread_it].arg.block_size = con->block_size;
			bench->thread[cpu_it][thread_it].arg.core = cpu_it;
			bench->thread[cpu_it][thread_it].arg.thread = thread_it;
			
			pthread_attr_init(&bench->thread[cpu_it][thread_it].attr);

			cpu_set_t cpu_info;
			CPU_ZERO(&cpu_info);
			CPU_SET(cpu_it, &cpu_info);
			if(pthread_attr_setaffinity_np(&bench->thread[cpu_it][thread_it].attr, sizeof(cpu_set_t), &cpu_info)) {
				printf("set affinity failed");
        		goto fail_pthread_create;
			}
			
			if(pthread_create(&bench->thread[cpu_it][thread_it].tid, NULL, cpu_bench_worker, (void *)&bench->thread[cpu_it][thread_it].arg)) {
				printf("Fail pthread create, thread num is c:%dt:%d\n", cpu_it, thread_it);
				goto fail_pthread_create;
			}
		}
	}

	return 0;

fail_pthread_create:
	
	free(bench->thread[cpu_it]);

fail_threads:
	
	for(i = 0; i < cpu_it; i++) {
		for(j = 0; j < threads_per_core; j++) {
			pthread_cancel(bench->thread[i][j].tid);
		}
		free(bench->thread[i]);
	}

	free(bench->thread);
	
fail_cpus:
	pthread_barrier_destroy(&barrier);
	return -1;
}

void cpu_bench_finish(struct cpu_bench *bench, struct config *con)
{
	int cpu_it, thread_it;
	int cpu_cores = con->cores;
	int threads_per_core = con->threads_per_core;
	int *tret;

	//wait bench thread finish
	for(cpu_it = 0; cpu_it < cpu_cores; cpu_it++) {
		for(thread_it = 0; thread_it < threads_per_core; thread_it++) {
			pthread_join(bench->thread[cpu_it][thread_it].tid, NULL);
		}
	}

	//free thread structure
	for(cpu_it = 0; cpu_it < cpu_cores; cpu_it++) {	
		free(bench->thread[cpu_it]);
	}
	free(bench->thread);
	
	pthread_barrier_destroy(&barrier);
}


