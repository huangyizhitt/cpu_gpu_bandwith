#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "config.h"
#include "bench.h"
#include "cpu_bench.h"
#include "gpu_bench.h"

/*int test1(int argc, char **argv)
{
	struct config con;
	struct cpu_bench bench;
	int cpu_it, thread_it;

	bench_default_argument(&con);
	bench_process_input(argc, argv, &con);
	bench_init(&con);
	bench_print_config(&con);

	if(con.cores > 0 && con.threads_per_core > 0) {
	cpu_bench_init(&bench, &con);
	}

#if (GPU == 1)
	if(use_gpu)
		gpu_bench(&con);
#endif
	cpu_bench_finish(&bench, &con);

	bench_deinit(&con);
	return 0;
}*/

int main(int argc, char **argv)
{
	struct bench_config *con = bench_init(argc, argv);
	bench_print_config(con);

	cpu_bench_init(con);
	gpu_bench_init(con);
	gpu_bench_finish(con);
	cpu_bench_finish(con);
	cpu_bench_deinit(con);
	bench_print_out(con);
	bench_deinit(con);
	return 0;
}
