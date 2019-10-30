#ifndef _BENCH_H_
#define _BENCH_H_

#include <stdbool.h>

#define TRUE	1
#define FALSE	0

#define likely(x)       __builtin_expect(!!(x),1)
#define unlikely(x)     __builtin_expect(!!(x),0)
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

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

enum thread_status {
	READY=0,
	BUSY,
	DONE,
};
	
extern enum trans_status test_status;
extern enum thread_status **thread_status;
extern pthread_barrier_t g_barrier;
enum trans_status gpu_test_status;

double bench_second();	
CPU_DATA_TYPE *bench_generate_test_array(size_t size, size_t align);
void bench_destroy_test_array(CPU_DATA_TYPE *arr);
void bench_process_input(int argc, char **argv, struct bench_config *con);
struct bench_config *bench_init(int argc, char **argv);
void bench_deinit(struct bench_config *con);
void bench_print_config(struct bench_config *con);
void bench_print_out(const char *test_name, int core, int thread, double time, double size);
void bench_default_argument(struct bench_config *con);


static inline void bench_set_thread_status(int core_id, int thread_id, enum thread_status status)
{
	ACCESS_ONCE(thread_status[core_id][thread_id]) = status;
}

static inline bool bench_all_thread_done(struct bench_config *con)
{
	int cpu_it, thread_it;
	for(cpu_it = 0; cpu_it < con->cpu_con->cores; cpu_it++) {
		for(thread_it = 0; thread_it < con->cpu_con->cpus[cpu_it].threads_num; thread_it++) {
			if(ACCESS_ONCE(thread_status[cpu_it][thread_it]) != DONE) return FALSE;
		}
	}
	return TRUE;
}
#endif

