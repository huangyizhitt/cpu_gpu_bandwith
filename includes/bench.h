#ifndef _BENCH_H_
#define _BENCH_H_

#define KB	(1024)
#define MB	(1024*1024)


struct config{
	int loops;
	int cores;
	int threads_per_core;
	int block_size;
	size_t size;
};

extern int gpu_done;
extern int use_gpu;

long *bench_generate_test_array(size_t size);
void bench_process_input(int argc, char **argv, struct config *con);
void bench_init(struct config *con);
void bench_print_config(struct config *con);
void bench_print_out(int core, int thread, double time, double size);
void bench_default_argument(struct config *con);

#endif

