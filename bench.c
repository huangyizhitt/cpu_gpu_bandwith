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

bool use_cpu = false;
bool use_gpu = false;
enum trans_status gpu_test_status;

bool **thread_status;
pthread_barrier_t gpu_barrier;

static char *label[8] = {"All(Memcpy, SequentialWrite, Scale, Add, Triad)", "Memcpy", "SequentialWrite", "Scale", "Add", "Triad", "Unknown"};

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
	
	while((opt = getopt(argc, argv, "hCGf:")) != EOF) {
		switch(opt){
			case 'h':
				bench_usage();
				exit(0);

			case 'C':
				use_cpu = true;
				if(config_get_from_default(CPU, con) == false) {
					printf("get CPU default config fail\n");
					exit(1);
				}
				break;
			
			case 'G':
				use_gpu = true;
				if(config_get_from_default(GPU, con) == false) {
					printf("get GPU default config fail\n");
					exit(1);
				}
				break;

			case 'f':
				if(config_get_from_xml(optarg, con) == false) {
					printf("get config from xml fail\n");
					exit(1);
				}
				
				if(con->cpu_con) {
					use_cpu = true;
				}
						
				if(con->gpu_con) {
					use_gpu = true;
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
	srand((unsigned)time(NULL));

	struct bench_config *con = config_create();
	if(!con) {
		printf("allocate struct bench_config fail\n");
		exit(1);
	}
	bench_process_input(argc, argv, con);

	if(use_gpu)
		gpu_test_status = INIT;

	if(use_gpu && use_cpu) {
		pthread_barrier_init(&gpu_barrier, NULL, con->cpu_con->total_threads+1);
	}

	return con;
}

void bench_deinit(struct bench_config *con)
{
	if(use_cpu && use_gpu) {
		pthread_barrier_destroy(&gpu_barrier);
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

void bench_print_out(const char *name, int core, int thread, double time, double size)
{
	printf("[%s]\t", name);
	printf("Core: %d\t", core);
	printf("Thread: %d\t", thread);
	printf("Elapsed: %.5f\t", time);
    printf("MiB: %.5f\t", size);
    printf("Copy: %.3f MiB/s\n", size/time);
}

