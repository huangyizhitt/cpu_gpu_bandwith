#ifndef _BENCH_H_
#define _BENCH_H_

#include <stdbool.h>

#define KB	(1024)
#define MB	(1024*1024)

#define READY 	0
#define BUSY	1

#define TRUE	1
#define FALSE	0

struct config{
	int loops;
	int cores;
	int threads_per_core;
	int block_size;
	size_t size;						//CPU array size
	size_t gpu_array_size;				//GPU array size
};

enum trans_status {
	INIT=0,								//init status		
	HOST_TO_DEVICE,						//CPU->GPU by memory
	HOST_TO_DEVICE_COMPLETE,
	DEVICE,								//GPU access memory
	DEVICE_COMPLETE,
	DEVICE_TO_DEVICE,					//GPU memory->GPU another memory
	DEVICE_TO_HOST,						//GPU->CPU
	COMPLETE,							
};

extern int use_gpu;
extern enum trans_status test_status;
extern bool **thread_status;
extern pthread_barrier_t gpu_barrier;
enum trans_status gpu_test_status;

	
long *bench_generate_test_array(size_t size);
void bench_process_input(int argc, char **argv, struct config *con);
void bench_init(struct config *con);
void bench_deinit(struct config *con);
void bench_print_config(struct config *con);
void bench_print_out(int core, int thread, double time, double size);
void bench_default_argument(struct config *con);

static inline void bench_reset_thread_status(int core_id, int thread_id)
{
	thread_status[core_id][thread_id] = READY;
}

static inline void bench_set_thread_status(int core_id, int thread_id)
{
	thread_status[core_id][thread_id] = BUSY;
}

static inline bool bench_all_thread_ready(int cores, int threads_per_core)
{
	int cpu_it, thread_it;
	for(cpu_it = 0; cpu_it < cores; cpu_it++) {
		for(thread_it = 0; thread_it < threads_per_core; thread_it++) {
			if(thread_status[cpu_it][thread_it] == BUSY) return FALSE;
		}
	}
	return TRUE;
}
#endif

