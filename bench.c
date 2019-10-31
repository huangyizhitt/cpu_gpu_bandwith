#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include "config.h"
#include "bench.h"

#define VERSION	"1.0"

enum use_status use_cpu = UNUSED;
enum use_status use_gpu = UNUSED;
enum trans_status gpu_test_status;

enum thread_status **thread_status;
pthread_barrier_t g_barrier;				//g_barrier of memory bandwith test for all threads


const char *label[8] = {"All(Memcpy, SequentialWrite, Scale, Add, Triad)", "Memcpy", "SequentialWrite", "Scale", "Add", "Triad", "Unknown"};
const char *use[3] = {"Unused", "Forward", "Benckend"};

void bench_usage()
{
	printf("cpu and gpu memory bandwidth benchmark v%s\n", VERSION);
	printf("Usage: gcmb [-C -G or -f filename]\n");
	printf("Options:\n");
	printf("	-C: use default CPU memory test(cores: %d, thread per core: %d, size: %dMB, align: %d, loops: %d, test type: %d)\n", DEFAULT_CPU_CORES, 
		DEFALUT_THREADS_NUM_IN_CPU, DEFAULT_CPU_SIZE, DEFAULT_ALIGN, DEFAULT_LOOPS, DEFAULT_TEST_TYPE);
	printf("	-G: use default GPU memory test(size: %d, test type: %d)\n", DEFAULT_GPU_SIZE, DEFAULT_TEST_TYPE);
	printf("	-f: use configuration file XML format in memory tests(default: %s)\n", DEFAULT_CONFIGURATION_FILE);
	printf("	-h:	usage help\n");
}

double bench_second()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.0e-6 );
}


CPU_DATA_TYPE *bench_generate_test_array(size_t size, size_t align)
{
	size_t iter;

	CPU_DATA_TYPE *arr = (CPU_DATA_TYPE *)aligned_alloc(align, size * sizeof(CPU_DATA_TYPE));

	if(!arr) {
		errno = ENOMEM;
		perror("Fali to allocate memory!\n");
		exit(1);
	}

	for(iter = 0; iter < size / sizeof(CPU_DATA_TYPE); iter++) {
		arr[iter] = iter;
	}

	return arr;
}

void bench_destroy_test_array(CPU_DATA_TYPE *arr)
{
	if(arr) free(arr);
}

void bench_process_input(int argc, char **argv, struct bench_config *con)
{
	int opt;
	double size;
	if(argc < 2) {
		printf("wrong usage, please ./gcmbw -h know how to use\n");
		exit(1);
	}
	
	while((opt = getopt(argc, argv, "hC:G:f:")) != EOF) {
		switch(opt){
			case 'h':
				bench_usage();
				exit(0);

			case 'C':
				use_cpu=strtoul(optarg, (char **)NULL, 10);
				if(use_cpu) {
//				use_cpu = DEFAULT_CPU_USE;
					if(config_get_from_default(CPU, con) == false) {
						printf("get CPU default config fail\n");
						exit(1);
					}
				}
				break;
			
			case 'G':
				use_gpu = strtoul(optarg, (char **)NULL, 10);
				if(use_gpu) {
					if(config_get_from_default(GPU, con) == false) {
						printf("get GPU default config fail\n");
						exit(1);
					}
				}	
				break;

			case 'f':
				if(config_get_from_xml(optarg, con) == false) {
					printf("get config from xml fail\n");
					exit(1);
				}
				
				break;
				
			default:
				printf("wrong usage, please ./gcmbw -h know how to use\n");
				break;
		}
	}
}

struct bench_config *bench_init(int argc, char **argv)
{
	int threads, cpu_id, thread_id, i, j;
	srand((unsigned)time(NULL));

	struct bench_config *con = config_create();
	if(!con) {
		printf("allocate struct bench_config fail\n");
		exit(1);
	}
	bench_process_input(argc, argv, con);

	if(!use_cpu && !use_gpu) {
		printf("no device used!\n");
		exit(0);
	}

	if(use_cpu == FORWORD) {
		thread_status = (enum thread_status **)malloc(sizeof(enum thread_status *) * con->cpu_con->cores);
		if(!thread_status) {
			printf("allocate thread status fail\n");
			goto fail_cpu;
		}

		for(cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
			thread_status[cpu_id] = (enum thread_status *)malloc(sizeof(enum thread_status) * con->cpu_con->cpus[cpu_id].threads_num);
			if(!thread_status[cpu_id]) {
				printf("allocate thread status fail\n");
				goto fail_thread;
			}	
			for(thread_id = 0; thread_id < con->cpu_con->cpus[cpu_id].threads_num; thread_id++) {
				bench_set_thread_status(cpu_id, thread_id, READY);
			}
		}
	}
	
	if(use_gpu)
		ACCESS_ONCE(gpu_test_status) = INIT;

	if(use_cpu) {
		threads = con->cpu_con->total_threads;

		if(use_gpu)
			pthread_barrier_init(&g_barrier, NULL, threads + 1);				//all cpu thread and gpu task thread

		if(!use_gpu)
			pthread_barrier_init(&g_barrier, NULL, threads);		
	}
	
	return con;
	
fail_thread:
	for(i = 0; i < cpu_id; i++)
	{
		free(thread_status[i]);
	}

fail_cpu:
	free(con);
	return NULL;
}

void bench_deinit(struct bench_config *con)
{
	if(use_cpu)
		pthread_barrier_destroy(&g_barrier);

	if(use_cpu == FORWORD) {
		for(int cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
			free(thread_status[cpu_id]);
		}
		free(thread_status);
	}
	config_destroy(con);
}

void bench_print_config(struct bench_config *con)
{
	if(con) {
		if(con->cpu_con) {
			printf("CPU name: %s, CPUs: %d, loops: %d\n", con->cpu_con->name, con->cpu_con->cores, con->cpu_con->loops);
			for(int i = 0; i < con->cpu_con->cores; i++) {
				printf("\tcpu %d, thread num: %d\n", i, con->cpu_con->cpus[i].threads_num);
				for(int j = 0; j < con->cpu_con->cpus[i].threads_num; j++) {
					printf("\tthread %d, workload size: %.3fMB, memory align: %lld, test type: %s, uses %d bytes per array element, use cache: %d\n", j, 
						(double)con->cpu_con->cpus[i].threads[j].size/MB,
						 con->cpu_con->cpus[i].threads[j].align, label[con->cpu_con->cpus[i].threads[j].type], 
						 con->cpu_con->cpus[i].threads[j].bytes_per_element, con->cpu_con->cpus[i].threads[j].use_cache);
				}
			}
		}

		if(con->gpu_con) {
			printf("GPU name: %s, workload size: %.3fMB, gpu test type: %s, uses %d bytes per array element, use cache: %d\n", 
				con->gpu_con->name, (double)con->gpu_con->size/MB, label[con->gpu_con->type], 
				con->gpu_con->bytes_per_element, con->gpu_con->use_cache);
		}
	}
}

static void bench_cpu_print_out(const char *name, int cpu_id, int thread_id, long long size, double elapse, double bandwidth)
{
	printf("[%s]\t", name);
	printf("Core: %d\t", cpu_id);
	printf("Thread: %d\t", thread_id);
	printf("MiB: %lld\t", size / MB);
	printf("Elapsed: %.5f\t", elapse);
    printf("Copy: %.3f MiB/s\n", bandwidth);
}

static void bench_gpu_print_out(const char *name, long long size, float trans_bandwidth, float access_bandwidth)
{
	printf("[%s]\t", name);
	printf("MiB: %lld\t", size / MB);
	printf("Transmission bandwidth between cpu and gpu: %.3f\t", trans_bandwidth);
    printf("GPU access memory bandwidth: %.3f MiB/s\n", access_bandwidth);
}

void bench_print_out(struct bench_config *con)
{
/*	printf("[%s]\t", name);
	printf("Core: %d\t", core);
	printf("Thread: %d\t", thread);
	printf("Elapsed: %.5f\t", time);
    printf("MiB: %.5f\t", size);
    printf("Copy: %.3f MiB/s\n", size/time);*/

	int cpu_id, thread_id;

	if(con->cpu_con) {
		for(cpu_id = 0; cpu_id < con->cpu_con->cores; cpu_id++) {
			for(thread_id = 0; thread_id < con->cpu_con->cpus[cpu_id].threads_num; thread_id++) {
				bench_cpu_print_out(label[con->cpu_con->cpus[cpu_id].threads[thread_id].type], 
					cpu_id, thread_id, con->cpu_con->cpus[cpu_id].threads[thread_id].size,
					con->cpu_con->cpus[cpu_id].threads[thread_id].elapse,
					con->cpu_con->cpus[cpu_id].threads[thread_id].bandwidth);
			}
		}
	}

	if(con->gpu_con) {
		bench_gpu_print_out("GPU Test", con->gpu_con->size, con->gpu_con->trans_bandwidth, con->gpu_con->access_bandwidth);
	}
}

