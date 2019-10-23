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

int gpu_done;

void bench_usage()
{
	printf("cpu and gpu memory benchmark v%s\n", VERSION);
	printf("Usage: gcmb [options] array_size_in_MiB\n");
	printf("Options:\n");
	printf("	-n: number of runs per test(default: %d)\n", DEFAULT_LOOPS);
	printf("	-c: number of cpu core in test(default: %ld)\n", DEFAULT_CPU_CORES);
	printf("	-t: number of thread in per cpu core(default: %d)\n", DEFALUT_THREADS_NUM_IN_CPU);
	printf("	-b: block size in bytes(default: %d)\n", DEFALUT_BLOCK_SIZE);
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
	while((opt = getopt(argc, argv, "hn:c:t:b:")) != EOF) {
		switch(opt){
			case 'h':
				bench_usage();
				exit(0);

			case 'n':
				con->loops = strtoul(optarg, (char **)NULL, 10);
				if(con->loops <= 0) {
					printf("Error: run test must >=1\n");
					exit(1);
				}
				break;

			case 'c':
				con->cores = strtoul(optarg, (char **)NULL, 10);
				if(con->cores <= 0) {
					printf("Error: core number must >=1\n");
					exit(1);
				}

			case 't':
				con->threads_per_core = strtoul(optarg, (char **)NULL, 10);
				if(con->threads_per_core <= 0) {
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
	srand((unsigned)time(NULL));
	gpu_done = 0;
}

void bench_default_argument(struct config *con)
{
	con->loops = DEFAULT_LOOPS;
	con->cores = DEFAULT_CPU_CORES;
	con->threads_per_core = DEFALUT_THREADS_NUM_IN_CPU;
	con->block_size = DEFALUT_BLOCK_SIZE;
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

