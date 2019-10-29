#ifndef _CPU_BENCH_H_
#define _CPU_BENCH_H_

#include <pthread.h>

bool cpu_bench_init(struct bench_config *con);
void cpu_bench_finish(struct bench_config *con);
void cpu_bench_deinit(struct bench_config *con);

#endif

