#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include "bench.h"

static int xmlStrchar(xmlChar *str, char c)
{
	int i = 0;
	while(str[i] != '\0') {
		if(str[i] == c) {
			return i;
		}
		i++;
	}
	return -1;
}

static unsigned long long to_bytes(xmlNodePtr cur_node)
{
	xmlChar		*key;
	int index;
	char str[64];
	long long size;

	key = xmlNodeGetContent(cur_node);
	if((index = xmlStrchar(key, 'M')) != -1) {
		strncpy(str, (char *)key, index);
		size = atoll(str) * MB;
	} else if((index = xmlStrchar(key, 'K')) != -1) {
		strncpy(str, (char *)key, index);
		size = atoll(str) * KB;
	} else {
		size = atoll((char *)key);
	}
	xmlFree(key);
	return size;
}

static bool thread_attributes_get_from_xml(xmlNodePtr thread_node, struct thread *thread, int thread_id)
{
	xmlNodePtr cur_node = thread_node->children;
	xmlChar		*key;
	int index;
	char str[64];
	
	while(cur_node) {
		if(!xmlStrcmp(cur_node->name, BAD_CAST"type")) {
			key = xmlNodeGetContent(cur_node);
			thread[thread_id].type = atoi(key);
			xmlFree(key);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"size")) {
			thread[thread_id].size = to_bytes(cur_node);
		} else if (!xmlStrcmp(cur_node->name, BAD_CAST"block_size")) {
			thread[thread_id].block_size = to_bytes(cur_node);
		} else {
//			printf("wrong thread attributes xml format!\n");
//			return false;
		}

		cur_node = cur_node->next;
	}

	return true;
}

static bool cpu_attributes_get_from_xml(xmlNodePtr cpu_node, struct cpu *cpu, int cpu_id)
{
	xmlNodePtr cur_node = cpu_node->children;
	xmlChar		*key;
	int i = 0;

	while(cur_node) {
		if(!xmlStrcmp(cur_node->name, BAD_CAST"thread_num")) {
			key = xmlNodeGetContent(cur_node);
			cpu[cpu_id].threads_num = atoi(key);
			xmlFree(key);
			cpu[cpu_id].threads = (struct thread *) malloc(sizeof(struct thread) * cpu[cpu_id].threads_num);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"thread")) {
			if(i > cpu[cpu_id].threads_num) {
				printf("wrong thread xml format!\n");
				goto fail_format;
			}

			if(thread_attributes_get_from_xml(cur_node, cpu[cpu_id].threads, i) == false) {
				printf("thread attributes get from xml fail!\n");
				goto fail_format;
			}
			i++;
			
		} else {
//			printf("wrong cpu attributes xml format!\n");
//			goto fail_format;
		}

		cur_node = cur_node->next;
	}

	return true;

fail_format:
	free(cpu[cpu_id].threads);
	return false;
}

static struct cpu_config *cpu_config_create_from_xml(xmlNodePtr cpu_node)
{
	struct cpu_config *con = (struct cpu_config *)malloc(sizeof(*con));
	if(!con) {
		printf("cpu config struct allocate fail!\n");
		return NULL;
	}
	int i = 0;
	xmlChar		*key;
	xmlNodePtr cur_node = cpu_node->children;
	while(cur_node != NULL) {
		if(!xmlStrcmp(cur_node->name, BAD_CAST"name")) {
			key = xmlNodeGetContent(cur_node);
			strcpy(con->name, (char *)key);
			xmlFree(key);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"num")) {
			key = xmlNodeGetContent(cur_node);
			con->cores = atoi((char *)key);
			xmlFree(key);
			con->cpus = (struct cpu *)malloc(sizeof(struct cpu) * con->cores);
			if(!con->cpus) {
				printf("create cpu struct fail\n");
				goto fail_cpu;
			}
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"loops")) {
			key = xmlNodeGetContent(cur_node);
			con->loops = atoi((char *)key);
			xmlFree(key);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"CPU")) {
			if(i > con->cores) {
				printf("wrong xml format!\n");
				goto fail_format;
			}
			
			if(cpu_attributes_get_from_xml(cur_node, con->cpus, i) == false) {
				printf("cpu attributes get from xml fail!\n");
				goto fail_format;
			}
			
			i++;
		} else {
//			printf("wrong xml format!\n");
//			goto fail_format;
		}

		cur_node = cur_node->next;
	}

	return con;

fail_format:
	free(con->cpus);

fail_cpu:
	free(con);
	return NULL;
}

static struct gpu_config *gpu_config_create_from_xml(xmlNodePtr gpu_node)
{
	struct gpu_config *con = (struct gpu_config *)malloc(sizeof(*con));
	if(!con) {
		printf("gpu_config struct allocate fail!\n");
		return NULL;
	}

	xmlNodePtr cur_node = gpu_node->children;
	xmlChar		*key;
	int index;
	char str[64];

	while(cur_node) {
		if(!xmlStrcmp(cur_node->name, BAD_CAST"name")){
			key = xmlNodeGetContent(cur_node);
			strcpy(con->name, (char *)key);
			xmlFree(key);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"size")) {
			con->size = to_bytes(cur_node);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"type")){
			key = xmlNodeGetContent(cur_node);
			con->type = atoi(key);
			xmlFree(key);
		}
		cur_node = cur_node->next;
	}

	return con;
fail_format:
	free(con);
	return NULL;
}

bool config_get_from_xml(char *xml, struct config* con)
{
	xmlDocPtr	doc;
	xmlNodePtr	cur_node;

	doc = xmlParseFile(xml);
	if(!doc) {
		printf("xml file open fail\n");
		goto fail_xml;
	}

	cur_node = xmlDocGetRootElement(doc);
	if(!cur_node) {
		printf("empty xml file\n");
		goto fail_type;
	}

	if(xmlStrcmp(cur_node->name, BAD_CAST"bench")) {
		printf("xml of the wrong type, root node != bench\n");
		goto fail_type;
	}
	printf("cur_node name: %s\n", cur_node->name);

	cur_node = cur_node->children;

	while(cur_node) {
		if(!xmlStrcmp(cur_node->name, BAD_CAST"CPUbench")) {
			con->cpu_con = cpu_config_create_from_xml(cur_node);
		} else if(!xmlStrcmp(cur_node->name, BAD_CAST"GPUbench")) {
			con->gpu_con = gpu_config_create_from_xml(cur_node);
		} else {

		}
		cur_node = cur_node->next;
	}

	if(!con->gpu_con && !con->cpu_con) {
		printf("create cpu config and gpu config fail!\n");
		goto fail_type;
	}

	xmlFreeDoc(doc);
	return true;

fail_type:
	xmlFreeDoc(doc);

fail_xml:
	free(con);
	return false;
}

static struct cpu_config *cpu_config_from_default()
{
	struct cpu_config *con = (struct cpu_config *)malloc(sizeof(*con));
	if(!con) return NULL;
	int cpu_id, thread_id;

	strcpy(con->name, DEFAULT_CPU_NAME);
	con->cores = DEFAULT_CPU_CORES;
	con->loops = DEFAULT_LOOPS;
	con->cpus = (struct cpu *)malloc(sizeof(struct cpu) * con->cores);
	if(!con->cpus) {
		printf("con->cpus allocate fail\n");
		goto fail_cpus;
	}

	for(cpu_id = 0; cpu_id < con->cores; cpu_id++) {
		con->cpus[cpu_id].threads_num = DEFALUT_THREADS_NUM_IN_CPU;
		con->cpus[cpu_id].threads = (struct thread *)malloc(sizeof(struct thread) * con->cpus[cpu_id].threads_num);
		if(!con->cpus[cpu_id].threads) {
			printf("cpu %d: threads allocate fail\n", cpu_id);
			goto fail_threads;
		}
		
		for(thread_id = 0; thread_id < con->cpus[cpu_id].threads_num; thread_id++) {
			con->cpus[cpu_id].threads[thread_id].type = DEFAULT_TEST_TYPE;	
			con->cpus[cpu_id].threads[thread_id].size = DEFAULT_CPU_SIZE * MB;
			con->cpus[cpu_id].threads[thread_id].block_size = DEFAULT_BLOCK_SIZE;
		}
	}

	return con;
	
fail_threads:
	do {
		free(con->cpus[--cpu_id].threads);
	}while(cpu_id);
	free(con->cpus);
	
fail_cpus:
	free(con);
	return NULL;
}

static struct gpu_config *gpu_config_from_default()
{
	struct gpu_config *con = (struct gpu_config *)malloc(sizeof(*con));
	if(!con) {
		printf("gpu_config struct allocate fail\n");
		return NULL;
	}

	strcpy(con->name, DEFAULT_GPU_NAME);
	con->type = DEFAULT_TEST_TYPE;
	con->size = DEFAULT_GPU_SIZE * MB;

	return con;
}

bool config_get_from_default(enum device dev, struct config *con)
{
	switch(dev) {
		case CPU:
			con->cpu_con = cpu_config_from_default();
			if(!con->cpu_con) return false;
			break;

		case GPU:
			con->gpu_con = gpu_config_from_default();
			if(!con->gpu_con) return false;
			break;

		case FPGA:
			break;

		default:
			break;
	}
	return true;
}

struct config *config_create()
{
	struct config *con = (struct config *)malloc(sizeof(*con));
	if(!con)
		return NULL;
	return con;
}

void config_destroy(struct config *con)
{
	if(con) {
		if(con->cpu_con) {
			if(con->cpu_con->cpus) {
				if(con->cpu_con->cpus->threads) {
					free(con->cpu_con->cpus->threads);
				}
				free(con->cpu_con->cpus);
			}
			free(con->cpu_con);
		}
		
		if(con->gpu_con) {
			free(con->gpu_con);
		}

		free(con);
	}
}

int test()
{
	struct config *con = config_create();
	config_get_from_xml("configs.xml", con);

	if(con) {
		if(con->cpu_con) {
			printf("CPU name: %s, CPUs: %d, loops: %d\n", con->cpu_con->name, con->cpu_con->cores, con->cpu_con->loops);
			for(int i = 0; i < con->cpu_con->cores; i++) {
				printf("\tcpu %d, thread num: %d\n", i, con->cpu_con->cpus[i].threads_num);
				for(int j = 0; j < con->cpu_con->cpus[i].threads_num; j++) {
					printf("\tthread %d, workload type: %d, size: %lld, block_size: %lld\n", j, con->cpu_con->cpus[i].threads[j].type, \
						con->cpu_con->cpus[i].threads[j].size, con->cpu_con->cpus[i].threads[j].block_size);
				}
			}
		}

		if(con->gpu_con) {
			printf("GPU name: %s, workload size: %lld, gpu test type: %d\n", con->gpu_con->name, con->gpu_con->size, con->gpu_con->type);
		}
	}

	config_destroy(con);
}
