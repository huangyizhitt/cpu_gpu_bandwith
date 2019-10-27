#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include "bench.h"

int xmlStrchar(xmlChar *str, char c)
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

bool thread_attributes_get_from_xml(xmlNodePtr thread_node, struct thread *thread, int thread_id)
{
	xmlNodePtr cur_node = thread_node->children;
	xmlChar		*key;
	int index;
	char str[64];
	
	while(cur_node) {
		if(xmlStrcmp(cur_node->name, BAD_CAST"type")) {
			key = xmlNodeGetContent(cur_node);
			thread[thread_id].type = atoi(key);
			xmlFree(key);
		} else if(xmlStrcmp(cur_node->name, BAD_CAST"size")) {
			key = xmlNodeGetContent(cur_node);
			if((index = xmlStrchar(key, 'M')) != -1) {
				strncpy(str, (char *)key, index);
				thread[thread_id].size = atoll(str) * MB;
			} else if((index = xmlStrchar(key, 'K')) != -1) {
				strncpy(str, (char *)key, index);
				thread[thread_id].size = atoll(str) * KB;
			} else {
				thread[thread_id].size = atoll((char *)key);
			}
			xmlFree(key);
		} else if (xmlStrcmp(cur_node->name, BAD_CAST"block_size")) {
			key = xmlNodeGetContent(cur_node);
			if((index = xmlStrchar(key, 'M')) != -1) {
				strncpy(str, (char *)key, index);
				thread[thread_id].size = atoll(str) * MB;
			} else if((index = xmlStrchar(key, 'K')) != -1) {
				strncpy(str, (char *)key, index);
				thread[thread_id].size = atoll(str) * KB;
			} else {
				thread[thread_id].size = atoll((char *)key);
			}
			xmlFree(key);
		} else {
			printf("wrong thread attributes xml format!\n");
			return false;
		}

		cur_node = cur_node->next;
	}

	return true;
}

bool cpu_attributes_get_from_xml(xmlNodePtr cpu_node, struct cpu *cpu, int cpu_id)
{
	xmlNodePtr cur_node = cpu_node->children;
	xmlChar		*key;
	int i = 0;

	while(cur_node) {
		if(xmlStrcmp(cur_node->name, BAD_CAST"thread_num")) {
			key = xmlNodeGetContent(cur_node);
			cpu[cpu_id].threads_num = atoi(key);
			xmlFree(key);
			cpu[cpu_id].threads = (struct thread *) malloc(sizeof(struct thread) * cpu[cpu_id].threads_num);
		} else if(xmlStrcmp(cur_node->name, BAD_CAST"thread")) {
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
			printf("wrong cpu attributes xml format!\n");
			goto fail_format;
		}

		cur_node = cur_node->next;
	}

	return true;

fail_format:
	free(cpu[cpu_id].threads);
	return false;
}

struct cpu_config *cpu_config_create_from_xml(xmlNodePtr cpu_node, bool *flags)
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
		if(xmlStrcmp(cur_node->name, BAD_CAST"name")) {
			key = xmlNodeGetContent(cur_node);
			strcpy(con->name, (char *)key);
			xmlFree(key);
		} else if(xmlStrcmp(cur_node->name, BAD_CAST"num")) {
			key = xmlNodeGetContent(cur_node);
			con->cores = atoi((char *)key);
			xmlFree(key);
			con->cpus = (struct cpu *)malloc(sizeof(struct cpu) * con->cores);
			if(!con->cpus) {
				printf("create cpu struct fail\n");
				goto fail_cpu;
			}
		} else if(xmlStrcmp(cur_node->name, BAD_CAST"loops")) {
			key = xmlNodeGetContent(cur_node);
			con->loops = atoi((char *)key);
			xmlFree(key);
		} else if(xmlStrcmp(cur_node->name, BAD_CAST"CPU")) {
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
			printf("wrong xml format!\n");
			goto fail_format;
		}

		*flags = true;
		cur_node = cur_node->next;
	}

	return con;

fail_format:
	free(con->cpus);

fail_cpu:
	free(con);
	return NULL;
}

struct gpu_config *gpu_config_create_from_xml(xmlNodePtr gpu_node, bool *flags)
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
		if(xmlStrcmp(cur_node->name, BAD_CAST"size")) {
			key = xmlNodeGetContent(cur_node);
			if((index = xmlStrchar(key, 'M')) != -1) {
				strncpy(str, (char *)key, index);
				con->size = atoll(str) * MB;
			} else if((index = xmlStrchar(key, 'K')) != -1) {
				strncpy(str, (char *)key, index);
				con->size = atoll(str) * KB;
			} else {
				con->size = atoll((char *)key);
			}
			xmlFree(key);
		} else {
			printf("gpu config xml format wrong!\n");
			goto fail_format;
		}
		cur_node = cur_node->next;
	}

	*flags = true;
	return con;
fail_format:
	free(con);
	return NULL;
}

struct config *config_create_from_xml(char *xml)
{
	xmlDocPtr	doc;
	xmlNodePtr	cur_node;
	
	bool 		success = false;

	struct config *con = (struct config *)malloc(sizeof(*con));
	if(!con)
		return NULL;

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

	if(xmlStrcasecmp(cur_node->name, BAD_CAST"xmlinfo")) {
		printf("xml of the wrong type, root node != bench\n");
		goto fail_type;
	}
	printf("cur_node name: %s\n", cur_node->name);

	cur_node = cur_node->children;

	if (!xmlStrcasecmp( cur_node->name, BAD_CAST"version" ) ) {
		xmlChar *key = xmlNodeGetContent(cur_node);
		printf("version:%s\n", key);
		xmlFree(key); 
	}
	
	cur_node = cur_node->next;
	if(!xmlStrcasecmp(cur_node->name, BAD_CAST"CPUbench")) {
		con->cpu_con = cpu_config_create_from_xml(cur_node, &success);
	}

	cur_node = cur_node->next;
	if(!xmlStrcasecmp(cur_node->name, BAD_CAST"GPUbench")) {
		con->gpu_con = gpu_config_create_from_xml(cur_node, &success);
	}

	if((!con->gpu_con && !con->cpu_con) || (success == false))	{
		printf("xml cpu or gpu config wrong type!\n");
		goto fail_type;
	}

	xmlFreeDoc(doc);
	return con;

fail_type:
	xmlFreeDoc(doc);

fail_xml:
	free(con);
	return NULL;
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

int main()
{
	struct config *con = config_create_from_xml("test.xml");

	if(con) {
		printf("CPU name: %s, CPUs: %d, loops: %d\n", con->cpu_con->name, con->cpu_con->cores, con->cpu_con->loops);
	}

	config_destroy(con);
}