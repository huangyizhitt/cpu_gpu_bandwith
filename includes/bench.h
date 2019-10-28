#ifndef _BENCH_H_
#define _BENCH_H_

#include <stdbool.h>

#define KB	(1024)
#define MB	(1024*1024)

#define READY 	0
#define BUSY	1

#define TRUE	1
#define FALSE	0

#define DEFAULT_BLOCK_SIZE					4096
#define DEFALUT_THREADS_NUM_IN_CPU			1
#define DEFAULT_LOOPS 						10
#define DEFAULT_CPU_SIZE					10	
#define DEFAULT_GPU_SIZE					100
#define DEFAULT_CONFIGURATION_FILE			"configs.xml"
#define DEFAULT_CPU_NAME					"CPU"
#define DEFAULT_GPU_NAME					"GPU"
#define DEFAULT_CPU_CORES					sysconf(_SC_NPROCESSORS_ONLN)
#define DEFAULT_TEST_TYPE					MEMCPY

enum test_type {
	MEMCPY = 0,
	SEQUENTIAL_WRITE,
	RANDOM_WRITE,
	UNKNOWN_TYPE,
};

//thread descriptor
struct thread {
	enum test_type type;							//test type
	long long size;									//test data size in bytes
	long long block_size;							//block size in bytes
};

//CPU descriptor
struct cpu {
	int threads_num;								//number of thread in cpu
	struct thread *threads;							//threads in CPU	
};

//GPU descriptor
struct gpu_config {
	char name[20];									//GPU name
	enum test_type type;							//test type
	long long size;									//test data size in bytes
};

struct cpu_config{
	char name[20];									//CPU name
	int cores;										//number of cpu core in test 
	int loops;										//number of run times per test
	struct cpu *cpus;								//CPUs in test	
};

struct config{
	struct cpu_config *cpu_con;
	struct gpu_config *gpu_con;
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

enum device {
	CPU=0,
	GPU,
	FPGA,
	UNKNOWN_DEVICE,
};

extern bool use_gpu;
extern bool use_cpu;
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
bool config_get_from_xml(char *xml, struct config* con);
struct config *config_create();
void config_destroy(struct config *con);
bool config_get_from_default(enum device dev, struct config *con);



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

