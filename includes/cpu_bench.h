#ifndef _CPU_BENCH_H_
#define _CPU_BENCH_H_

#include <pthread.h>

struct cpu_bench_arg {
	int loops;
	int block_size;
	int size;
};


struct cpu_bench_thread {
	struct cpu_bench_arg	arg;
	pthread_t tid;
};

struct cpu_bench {
	struct cpu_bench_thread **thread;
};

int cpu_bench_init(struct cpu_bench *bench, struct config *con);
void cpu_bench_finish(struct cpu_bench *bench, struct config *con);

#endif

