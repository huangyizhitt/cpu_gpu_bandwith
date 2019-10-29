#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <stdbool.h>

extern bool use_gpu;
extern bool use_cpu;

#define DEFAULT_ALIGN						4096
#define DEFALUT_THREADS_NUM_IN_CPU			1
#define DEFAULT_LOOPS 						10
#define DEFAULT_CPU_SIZE					10	
#define DEFAULT_GPU_SIZE					100
#define DEFAULT_CONFIGURATION_FILE			"configs.xml"
#define DEFAULT_CPU_NAME					"CPU"
#define DEFAULT_GPU_NAME					"GPU"
#define DEFAULT_CPU_CORES					sysconf(_SC_NPROCESSORS_ONLN)
#define DEFAULT_TEST_TYPE					ALL
#define DEFAULT_CPU_DATA_TYPE				long
#define DEFAULT_GPU_DATA_TYPE				long
#define DEFAULT_USE_CACHE					false

#ifndef CPU_DATA_TYPE
#define CPU_DATA_TYPE						DEFAULT_CPU_DATA_TYPE
#endif

#ifndef GPU_DATA_TYPE
#define GPU_DATA_TYPE						DEFAULT_GPU_DATA_TYPE
#endif

#define KB	(1024)
#define MB	(1024*1024)

//typedef void *(pfunc)(void *);

enum test_type {
	ALL=0,
	MEMCPY,
	SEQUENTIAL_WRITE,
	SCALE,
	ADD,
	TRIAD,
	UNKNOWN_TYPE,
};

//thread descriptor
struct thread {
	//thread configuration
	enum test_type type;							//test type
	long long size;									//test data size in bytes
	long long align;								//memory align
	int bytes_per_element;							//bytes per array element
	int loops;										//number of run times per test
	bool use_cache;									//use cache or no

	//thread info
	pthread_attr_t attr;							//pthread attr
	void* (*thread_func)(void *);					//thread function
	pthread_t tid;									//pthread tid
	int cpu_id;										//cpu id of thread run in;
	int thread_id;									//thread id in bench;
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
	int bytes_per_element;							//bytes per array element
	bool use_cache;									//use cache or no
};

struct cpu_config{
	char name[20];									//CPU name
	int cores;										//number of cpu core in test 
	int loops;										//number of run times per test
	int total_threads;								//number of toatal threads in all cpu
	struct cpu *cpus;								//CPUs in test	
};

struct bench_config{
	struct cpu_config *cpu_con;
	struct gpu_config *gpu_con;
};

enum device {
	CPU=0,
	GPU,
	FPGA,
	UNKNOWN_DEVICE,
};

bool config_get_from_xml(char *xml, struct bench_config* con);
struct bench_config *config_create(void);
void config_destroy(struct bench_config *con);
bool config_get_from_default(enum device dev, struct bench_config *con);

#endif

