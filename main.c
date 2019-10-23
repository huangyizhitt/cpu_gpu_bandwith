#include <stdio.h>
#include <stdlib.h>
#include "bench.h"
#include "cpu_bench.h"
#include "gpu_bench.h"

int main(int argc, char **argv)
{
	struct config con;
	struct cpu_bench bench;
	int cpu_it, thread_it;

	bench_default_argument(&con);

	bench_process_input(argc, argv, &con);

	bench_print_config(&con);

	bench_init(&con);

	if(con.cores > 0 && con.threads_per_core > 0) {
		cpu_bench_init(&bench, &con);
		cpu_bench_finish(&bench, &con);
	}

#if (GPU == 1)
	gpu_bench(&con);
#endif

	return 0;
}
