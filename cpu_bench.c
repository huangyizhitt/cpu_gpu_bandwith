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
		c[i] = a[i] + scalar * b[i];						\
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
	struct thread *thread = (struct thread*)arg;
	double elapse = 0, t;
	int type_size = sizeof(CPU_DATA_TYPE);
	size_t size = thread->size / type_size;
	
	int align = thread->align;
	int loops = thread->loops;
	const char *name = "CPU co GPU";
	enum trans_status status;
	CPU_DATA_TYPE *a = bench_generate_test_array(size, align);
	CPU_DATA_TYPE *b = bench_generate_test_array(size, align);
	CPU_DATA_TYPE limit = get_limit_value();
	size_t i = 0, total = 0;


	status = ACCESS_ONCE(gpu_test_status);						//Force reading from memory, prevent the compiler from optimizing the order of changes
	pthread_barrier_wait(&g_barrier);

retry:
	t = bench_second();
	if(thread->use_cache) {
		CPU_DATA_TYPE value = rand() % limit;
		while(1) {
			if(unlikely(i >= size)) i = 0;
			a[i++] = value;
			total += type_size;
			if(unlikely(status != ACCESS_ONCE(gpu_test_status)))	 goto out;							
		}
	} else {
		while(1) {
			if(unlikely(i >= size)) i = 0;
			a[i++] = rand() % limit;
			total += type_size;
			if(unlikely(status != ACCESS_ONCE(gpu_test_status)))	 goto out;							
		}
	} 

out:
	elapse = (bench_second() - t);
	
	if(status == DEVICE) {
		bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)total / MB);
	}

	
	status = ACCESS_ONCE(gpu_test_status);
	total = 0;
	pthread_barrier_wait(&g_barrier);										//all thread and gpu ready!

	if(gpu_test_status != COMPLETE) goto retry;
	
	return NULL;
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

		pthread_barrier_wait(&g_barrier);										//all thread ready!
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

			pthread_barrier_wait(&g_barrier);	
			t = bench_second();
			TEST_MEMCPY(dst, src, size);
			elapse += (bench_second() - t);
		}
	}

	elapse /= loops;
	thread->elapse = elapse;
	thread->bandwidth = (double)thread->size / MB / elapse;
//	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
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
		pthread_barrier_wait(&g_barrier);										//all thread ready!

		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SEQUENCE_WRITE(arr, size, value);
		}
		elapse += (bench_second() - t);
	} else {
		pthread_barrier_wait(&g_barrier);	
		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SEQUENCE_WRITE(arr, size, rand() % limit);
		}
		elapse += (bench_second() - t);
	}

	elapse /= loops;
	thread->elapse = elapse;
	thread->bandwidth = (double)thread->size / MB / elapse;
//	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
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
		pthread_barrier_wait(&g_barrier);

		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SCALE(d, s, size, scalar);
		}
		elapse += (bench_second() - t);
	} else {
		pthread_barrier_wait(&g_barrier);

		t = bench_second();
		for(int i = 0; i < loops; i++) {
			TEST_SCALE(d, s, size, rand() % limit);
		}
		elapse += (bench_second() - t);
	}

	elapse /= loops;
	thread->elapse = elapse;
	thread->bandwidth = (double)thread->size / MB / elapse;
//	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
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
	
		pthread_barrier_wait(&g_barrier);
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

			pthread_barrier_wait(&g_barrier);
			t = bench_second();
			TEST_ADD(c, a, b, size);
			elapse += (bench_second() - t);
		}
	}

	elapse /= loops;
	thread->elapse = elapse;
	thread->bandwidth = (double)thread->size / MB / elapse;
//	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
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

		pthread_barrier_wait(&g_barrier);
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

			pthread_barrier_wait(&g_barrier);
			t = bench_second();
			TEST_TRIAD(c, a, b, size, scalar);
			elapse += (bench_second() - t);
		}
	}

	thread->elapse = elapse;
	thread->bandwidth = (double)thread->size / MB / elapse;
//	bench_print_out(name, thread->cpu_id, thread->thread_id, elapse, (double)thread->size / MB);
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

//only cpu bench
static void *cpu_forward_worker(void *arg)
{
	double elapse = 0;
	struct thread *thread = (struct thread*)arg;
	bench_set_thread_status(thread->cpu_id, thread->thread_id, BUSY);
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

	bench_set_thread_status(thread->cpu_id, thread->thread_id, DONE);
	pthread_exit((void *)0);
}

//cpu benckend
static void *cpu_benckend_worker(void *arg)
{

}


bool cpu_bench_init(struct bench_config *con)
{
	int cpu_id, thread_id, i, j;

	if(use_cpu == UNUSED) return true;
	
	for(cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
		for(thread_id = 0; thread_id < con->cpu_con->cpus[cpu_id].threads_num; thread_id++) {
			con->cpu_con->cpus[cpu_id].threads[thread_id].loops = con->cpu_con->loops;
			pthread_attr_init(&con->cpu_con->cpus[cpu_id].threads[thread_id].attr);

			if(use_cpu == FORWORD)
				con->cpu_con->cpus[cpu_id].threads[thread_id].thread_func = cpu_forward_worker;

			if(use_cpu == BENCKEND && use_gpu == FORWORD) 
				con->cpu_con->cpus[cpu_id].threads[thread_id].thread_func = cpu_benckend_worker;

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

	return false;
}

void cpu_bench_finish(struct bench_config *con)
{
	if(!use_cpu) return;
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
	if(!use_cpu) return;
	for(int cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
		for(int thread_id = 0; thread_id < con->cpu_con->cpus[cpu_id].threads_num; thread_id++) {
			pthread_attr_destroy(&con->cpu_con->cpus[cpu_id].threads[thread_id].attr);
		}
	}
	pthread_barrier_destroy(&g_barrier);
}
