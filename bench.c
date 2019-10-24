#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include "bench.h"

#define VERSION	"1.0"

#define DEFALUT_BLOCK_SIZE	4096
#define DEFAULT_CPU_CORES	sysconf(_SC_NPROCESSORS_ONLN)
#define DEFALUT_THREADS_NUM_IN_CPU	1
#define DEFAULT_LOOPS 10
#define DEFAULT_GPU_ARRAY_SIZE	10	

int use_gpu;
enum trans_status gpu_test_status;

bool **thread_status;
pthread_barrier_t gpu_barrier;



void bench_usage()
{
	printf("cpu and gpu memory benchmark v%s\n", VERSION);
	printf("Usage: gcmb [options] array_size_in_MiB\n");
	printf("Options:\n");
	printf("	-n: number of runs per test(default: %d)\n", DEFAULT_LOOPS);
	printf("	-c: number of cpu core in test(default: %ld)\n", DEFAULT_CPU_CORES);
	printf("	-t: number of thread in per cpu core(default: %d)\n", DEFALUT_THREADS_NUM_IN_CPU);
	printf("	-b: block size in bytes(default: %d)\n", DEFALUT_BLOCK_SIZE);
	printf("	-G: use gpu and gpu array size is default %dMB\n", DEFAULT_GPU_ARRAY_SIZE);
	printf("	-g: use gpu and set gpu array size in MB\n");
	printf("\nThe default is to run default config\n");
}

long *bench_generate_test_array(size_t size)
{
	size_t iter;

	long *res = calloc(size, sizeof(long));

	if(!res) {
		errno = ENOMEM;
		perror("Fali to allocate memory!\n");
		exit(1);
	}

	for(iter = 0; iter < size; iter++) {
		res[iter] = LONG_MAX;
	}

	return res;
}

void bench_process_input(int argc, char **argv, struct config *con)
{
	int opt;
	double size;
	while((opt = getopt(argc, argv, "hGg:n:c:t:b:")) != EOF) {
		switch(opt){
			case 'h':
				bench_usage();
				exit(0);

			case 'G':
				use_gpu = 1;
				con->gpu_array_size = DEFAULT_GPU_ARRAY_SIZE * MB / sizeof(unsigned char);
				break;

			case 'g':
				use_gpu = 1;
				con->gpu_array_size = strtoul(optarg, (char **)NULL, 10) * MB / sizeof(unsigned char);
				if(con->gpu_array_size < 0) {
					printf("Error: gpu array size must >0\n");
					exit(1);
				}
				break;

			case 'n':
				con->loops = strtoul(optarg, (char **)NULL, 10);
				if(con->loops <= 0) {
					printf("Error: run test must >=1\n");
					exit(1);
				}
				break;

			case 'c':
				con->cores = strtoul(optarg, (char **)NULL, 10);
				if(con->cores < 0) {
					printf("Error: core number must >=1\n");
					exit(1);
				}

			case 't':
				con->threads_per_core = strtoul(optarg, (char **)NULL, 10);
				if(con->threads_per_core < 0) {
					printf("Error: thread number must >=1\n");
					exit(1);
				}
				break;

			case 'b':
				con->block_size = strtoul(optarg, (char **)NULL, 10);
				if(con->block_size <= 0) {
					printf("Error: memory block size must >=1\n");
					exit(1);
				}
				break;

			default:
				break;
		}
	}

	if(optind < argc) {
        size = strtoul(argv[optind++], (char **)NULL, 10);
    } else {
        printf("Error: no array size given!\n");
        exit(1);
    }

	con->size = size * MB / sizeof(long);
}

void bench_init(struct config *con)
{
	int cpu_it, thread_it; 
	int threads = con->cores * con->threads_per_core;
	srand((unsigned)time(NULL));

	gpu_test_status = INIT;
	thread_status = (bool **) malloc(con->cores * sizeof(bool *));
	for(cpu_it = 0; cpu_it < con->cores; cpu_it++) {
		thread_status[cpu_it] = (bool *)malloc(con->threads_per_core * sizeof(bool));
		for(thread_it = 0; thread_it < con->threads_per_core; thread_it++) {
			thread_status[cpu_it][thread_it] = 0;
		}
	}

	if(use_gpu) {
		pthread_barrier_init(&gpu_barrier, NULL, threads+1);
	}
}

void bench_deinit(struct config *con)
{
	int cpu_it, thread_it; 
	for(cpu_it = 0; cpu_it < con->cores; cpu_it++) {
		free(thread_status[cpu_it]);
	}
	free(thread_status);

	if(use_gpu) {
		pthread_barrier_destroy(&gpu_barrier);
	}
}

void bench_default_argument(struct config *con)
{
	con->loops = DEFAULT_LOOPS;
	con->cores = DEFAULT_CPU_CORES;
	con->threads_per_core = DEFALUT_THREADS_NUM_IN_CPU;
	con->block_size = DEFALUT_BLOCK_SIZE;
	use_gpu = 0;
}

void bench_print_config(struct config *con)
{
	printf("Configuration information:\n");
	printf("	Testing times: 				%d\n", con->loops);
	printf("	Testing CPUs:  				%d\n", con->cores);
	printf("	Testing threads per CPU:		%d\n", con->threads_per_core);
	printf("	Testing memory block size:		%d\n", con->block_size);
}

void bench_print_out(int core, int thread, double time, double size)
{
	printf("Core: %d\t", core);
	printf("Thread: %d\t", thread);
	printf("Elapsed: %.5f\t", time);
    printf("MiB: %.5f\t", size);
    printf("Copy: %.3f MiB/s\n", size/time);
}

