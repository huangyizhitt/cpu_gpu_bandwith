#define _GNU_SOURCE

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <limits.h>
#include "config.h"
#include "bench.h"
#include "cpu_bench.h"


pthread_barrier_t barrier;				//barrier of memory bandwith test for threads

/*#define TEST_MEMCPY(d, s, bytes, block_size)		\
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
}*/

#define TEST_MEMCPY(d, s, size) 							\
{															\
	for(size_t i = 0; i < size; i++) {						\
		d[i] = s[i];										\
	}														\
}

#define TEST_SEQUENCE_WRITE(d, size, value)					\
{															\
	for(size_t i = 0; i < size; i++) {						\
		d[i] = value;										\
	}														\
}

#define TEST_SCALE(d, s, size, scalar)						\
{															\
	for(size_t i = 0; i < size; i++) {						\
		d[i] = scalar * s[i];								\	
	}														\
}

#define TEST_ADD(c, a, b, size)								\
{															\
	for(size_t i = 0; i < size; i++) {						\
		c[i] = a[i] + b[i];									\
	}														\
}

#define TEST_TRIAD(c, a, b, size, scalar)					\
{															\
	for(size_t i = 0; i < size; i++) {						\
		c[i] = a[i] + scalar * b[i];							\
	}														\
}

static inline void test_memcpy(char *d, char *s, size_t bytes, int block_size)
{
	
}

/*static void test_memcpy_loops(char *d, char *s, size_t bytes, int block_size, int loops)
{
	while(loops--) {
		TEST_MEMCPY(d, s, bytes, block_size);
	}
}*/

static void test_sequence_write(long *array, size_t bytes, int loops)
{
	long *start = array, *p;
	long *stop = start + bytes / sizeof(long);
	long value = rand();
	
	while(loops--) {
		p = start;
		while(p != stop) {
			*p++ = value;
		}
	}
}

static inline int get_limit_value()
{
	int limit;
	if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(char))) {
		limit = 7;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(unsigned char))) {
		limit = 15;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(short))) {
		limit = 127;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(unsigned short))) {
		limit = 255;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(int))) {
		limit = 2e15-1;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(unsigned int))) {
		limit = 2e16-1;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(long))) {
		limit = 2e15-1;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(unsigned long))) {
		limit = 2e16-1;
	} else if(__builtin_types_compatible_p(typeof(CPU_DATA_TYPE), typeof(long long))) {
		limit = 2e15-1;
	} else {
		limit = 2e16-1;
	}
	return limit;
}

static void *cpu_co_gpu_worker(void *arg)
{
/*	struct timeval start, stop;
	struct cpu_bench_arg *data = (struct cpu_bench_arg*)arg;
	unsigned long long bytes = data->size * sizeof(long);
	int block_size = data->block_size;
	unsigned long long total_bytes = 0;
	double elapse;
	enum trans_status status;
	char *src, *dst;							
	size_t  remain;	

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

	status = gpu_test_status;
	pthread_barrier_wait(&gpu_barrier);

retry:
	gettimeofday(&start, NULL);
	while(1) {							
		src = (char *)a;										
		dst = (char *)b;										
		for(remain = bytes; remain >= block_size; remain -= block_size, src += block_size) {
			if(status != gpu_test_status) goto out;
			dst = memcpy(dst, src, block_size);		
			total_bytes += block_size;
		}																						
		if(remain) {																			
			dst = memcpy(dst, src, remain);		
			total_bytes += remain;
		}
	}

out:
	gettimeofday(&stop, NULL);
	elapse=((double)(stop.tv_sec * 1000000 - start.tv_sec * 1000000 + 
			stop.tv_usec - start.tv_usec))/1000000;
	
	if(status == HOST_TO_DEVICE || status == DEVICE) {
		bench_print_out(data->core, data->thread, elapse, (double)total_bytes / MB);
	}

	status = gpu_test_status;
	total_bytes = 0;
	pthread_barrier_wait(&gpu_barrier);										//all thread and gpu ready!

	if(gpu_test_status != COMPLETE) goto retry;
	
	return NULL;

fail_b:
	free(a);
fail_a:
	pthread_exit((void *)1);*/

}

void cpu_bench_copy(struct thread *thread)
{
	double elapse = 0, t;
	size_t size = thread->size / sizeof(CPU_DATA_TYPE);
	int align = thread->align;
	int loops = thread->loops;
	const char *name = "CPU Memcpy";

	CPU_DATA_TYPE *dst = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *src = bench_generate_test_array(size, align);
	int limit = get_limit_value();
	
	if(thread->use_cache) {		
		for(size_t i = 0; i < size; i++) {
			src[i] = rand() % limit;
		}

		pthread_barrier_wait(&barrier);										//all thread ready!
		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_MEMCPY(dst, src, size);
		}
		elapse += (bench_second() - t);
	} else {
		for(int i = 0; i < loops; i++) {
			for(size_t i = 0; i < size; i++) {
				src[i] = rand() % limit;
			}

			pthread_barrier_wait(&barrier);	
			t = bench_second();
			TEST_MEMCPY(dst, src, size);
			elapse += (bench_second() - t);
		}
	}

	elapse /= loops;
	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
	bench_destroy_test_array(dst);
	bench_destroy_test_array(src);
}

void cpu_bench_sequence_write(struct thread *thread)
{
	double elapse = 0, t;
	size_t size = thread->size / sizeof(CPU_DATA_TYPE);
	int align = thread->align;
	int loops = thread->loops;
	const char *name = "CPU Sequence Write";

	CPU_DATA_TYPE *arr = bench_generate_test_array(size, align);
	int limit = get_limit_value();

	if(thread->use_cache) {
		CPU_DATA_TYPE value = rand() % limit;
		pthread_barrier_wait(&barrier);										//all thread ready!

		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SEQUENCE_WRITE(arr, size, value);
		}
		elapse += (bench_second() - t);
	} else {
		pthread_barrier_wait(&barrier);	
		for(int i = 0; i < loops; i++) {
			TEST_SEQUENCE_WRITE(arr, size, rand() % limit);
		}
		elapse += (bench_second() - t);
	}

	elapse /= loops;
	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
	bench_destroy_test_array(arr);
}

void cpu_bench_scale(struct thread *thread)
{
	double elapse = 0, t;
	size_t size = thread->size / sizeof(CPU_DATA_TYPE);
	int align = thread->align;
	int loops = thread->loops;
	int limit = get_limit_value();
	const char *name = "CPU Scale";
	
	CPU_DATA_TYPE *d = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *s = bench_generate_test_array(size, align);

	if(thread->use_cache) {
		CPU_DATA_TYPE scalar = rand() % limit;							//may be will cut off result
		pthread_barrier_wait(&barrier);

		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SCALE(d, s, size, scalar);
		}
		elapse += (bench_second() - t);
	} else {
		pthread_barrier_wait(&barrier);

		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SCALE(d, s, size, rand() % limit);
		}
		elapse += (bench_second() - t);
	}

	elapse /= loops;
	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
	bench_destroy_test_array(d);	
	bench_destroy_test_array(s);
}

void cpu_bench_add(struct thread *thread)
{
	double elapse = 0, t;
	size_t size = thread->size / sizeof(CPU_DATA_TYPE);
	int align = thread->align;
	int loops = thread->loops;
	int limit = get_limit_value();
	const char *name = "CPU Add";

	CPU_DATA_TYPE *a = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *b = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *c = bench_generate_test_array(size, align);

	if(thread->use_cache) {
		for(size_t i = 0; i < size; i++) {
			a[i] = rand() % limit;
			b[i] = rand() % limit;
		}
	
		pthread_barrier_wait(&barrier);
		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_ADD(c, a, b, size);
		}
		elapse += (bench_second() - t);
	} else {
		for(int i = 0; i < loops; i++) {
			for(size_t i = 0; i < size; i++) {
				a[i] = rand() % limit;
				b[i] = rand() % limit;
			}

			pthread_barrier_wait(&barrier);
			t = bench_second();
			TEST_ADD(c, a, b, size);
			elapse += (bench_second() - t);
		}
	}

	elapse /= loops;
	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
	bench_destroy_test_array(a);	
	bench_destroy_test_array(b);
	bench_destroy_test_array(c);
}

void cpu_bench_triad(struct thread *thread)
{
	double elapse = 0, t;
	size_t size = thread->size / sizeof(CPU_DATA_TYPE);
	int align = thread->align;
	int loops = thread->loops;
	int limit = get_limit_value();
	const char *name = "CPU Triad";

	CPU_DATA_TYPE *a = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *b = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *c = bench_generate_test_array(size, align);
	CPU_DATA_TYPE scalar = rand() % limit;

	if(thread->use_cache) {
		for(size_t i = 0; i < size; i++) {
			a[i] = rand() % limit;
			b[i] = rand() % limit;
		}

		pthread_barrier_wait(&barrier);
		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_TRIAD(c, a, b, size, scalar);
		}
		elapse += (bench_second() - t);
	} else {
		for(int i = 0; i < loops; i++) {
			for(size_t i = 0; i < size; i++) {
				a[i] = rand() % limit;
				b[i] = rand() % limit;
			}

			pthread_barrier_wait(&barrier);
			t = bench_second();
			TEST_TRIAD(c, a, b, size, scalar);
			elapse += (bench_second() - t);
		}
	}

	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
	bench_destroy_test_array(a);	
	bench_destroy_test_array(b);
	bench_destroy_test_array(c);
}

void cpu_bench_all(struct thread *thread)
{
	cpu_bench_copy(thread);
	cpu_bench_sequence_write(thread);
	cpu_bench_scale(thread);
	cpu_bench_add(thread);
	cpu_bench_triad(thread);
}

static void *cpu_bench_worker(void *arg)
{
	double elapse = 0;
	struct thread *thread = (struct thread*)arg;

	switch(thread->type) {
		case ALL:
			cpu_bench_all(thread);
			break;
		case MEMCPY:
			cpu_bench_copy(thread);
			break;
		case SEQUENTIAL_WRITE:
			cpu_bench_sequence_write(thread);
			break;
		case SCALE:
			cpu_bench_scale(thread);
			break;
		case ADD:
			cpu_bench_add(thread);
			break;
		case TRIAD:
			cpu_bench_triad(thread);
			break;
		default:
			printf("error test type!\n");
			pthread_exit((void *)1);
	}
	
	pthread_exit((void *)0);
}

bool cpu_bench_init(struct bench_config *con)
{
	int cpu_id, thread_id, i, j;
	int threads = con->cpu_con->total_threads;

	pthread_barrier_init(&barrier, NULL, threads);
	
	for(cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
		for(thread_id = 0; thread_id < con->cpu_con->cpus[cpu_id].threads_num; thread_id++) {
			con->cpu_con->cpus[cpu_id].threads[thread_id].loops = con->cpu_con->loops;
			pthread_attr_init(&con->cpu_con->cpus[cpu_id].threads[thread_id].attr);

			if(use_cpu)
				con->cpu_con->cpus[cpu_id].threads[thread_id].thread_func = cpu_bench_worker;

			if(use_cpu && use_gpu) 
				con->cpu_con->cpus[cpu_id].threads[thread_id].thread_func = cpu_co_gpu_worker;

			cpu_set_t cpu_info;
			CPU_ZERO(&cpu_info);
			CPU_SET(cpu_id, &cpu_info);
			if(pthread_attr_setaffinity_np(&con->cpu_con->cpus[cpu_id].threads[thread_id].attr, sizeof(cpu_set_t), &cpu_info)) {
				printf("set affinity failed");
        		goto fail_pthread_create;
			}

			if(pthread_create(&con->cpu_con->cpus[cpu_id].threads[thread_id].tid, NULL, con->cpu_con->cpus[cpu_id].threads[thread_id].thread_func, (void *)&con->cpu_con->cpus[cpu_id].threads[thread_id])) {
					printf("Fail pthread create, thread num is c:%dt:%d\n", cpu_id, thread_id);
					goto fail_pthread_create;
			}

			con->cpu_con->cpus[cpu_id].threads[thread_id].cpu_id = cpu_id;
			con->cpu_con->cpus[cpu_id].threads[thread_id].thread_id = thread_id;
		}
	}

	return true;
fail_pthread_create:
	for(i = 0; i < cpu_id; i++) {
		for(j = 0; j < con->cpu_con->cpus[i].threads_num; j++)
			pthread_attr_destroy(&con->cpu_con->cpus[i].threads[j].attr);
	}

	for(j = 0; j < thread_id; j++) {
		pthread_attr_destroy(&con->cpu_con->cpus[cpu_id].threads[j].attr);
	}

	pthread_barrier_destroy(&barrier);
	return false;
}

void cpu_bench_finish(struct bench_config *con)
{
	struct cpu_config *cpu_config = con->cpu_con;
	int cpu_id, thread_id;

	for(cpu_id = 0; cpu_id < cpu_config->cores; cpu_id++) {
		for(thread_id = 0; thread_id < cpu_config->cpus[cpu_id].threads_num; thread_id++) {
			pthread_join(cpu_config->cpus[cpu_id].threads[thread_id].tid, NULL);
		}
	}
}

void cpu_bench_deinit(struct bench_config *con)
{
	for(int cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
		for(int thread_id = 0; thread_id < con->cpu_con->cpus[cpu_id].threads_num; thread_id++) {
			pthread_attr_destroy(&con->cpu_con->cpus[cpu_id].threads[thread_id].attr);
		}
	}
	pthread_barrier_destroy(&barrier);
}