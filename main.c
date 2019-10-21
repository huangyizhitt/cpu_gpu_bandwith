#include <stdio.h>
#include <stdlib.h>
#include "bench.h"
#include "cpu_bench.h"

int main(int argc, char **argv)
{
	struct config con;
	struct cpu_bench bench;
	int cpu_it, thread_it;

	bench_default_argument(&con);

	bench_process_input(argc, argv, &con);

	bench_print_config(&con);

	cpu_bench_init(&bench, &con);
	cpu_bench_finish(&bench, &con);

	return 0;
}
