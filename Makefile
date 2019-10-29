#Makefile for gcmbw

CC = gcc
NVCC = nvcc
CFLAGS = -O2 -g -std=gnu11
RM = rm -rf
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
INCLUDES = -I./includes -I/usr/include/libxml2
LIBS = -lpthread -lxml2
CUDA_LIBS_DIR = /usr/local/cuda/lib
CUDA_LIBS = -lcuda -lcudart
HAVE_NVCC = $(shell $(NVCC) -V | if [ $?==0 ]; then echo "1"; else echo "0"; fi)

#ifeq ($(HAVE_NVCC),1)
#	DEFINES = -DGPU=1 
#	CUDA_DEFINES = -D_FORCE_INLINES -gencode arch=compute_32,code=compute_32
#	OBJS += $(CUDA_SRCS:.cu=.o)
#	CUDA_SRCS = $(wildcard *.cu)
#else
#	DEFINES = 
#endif

HAVE_NVCC = 0

ifeq ($(HAVE_NVCC),1)
gcmbw: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(DEFINES) $(INCLUDES) $(LIBS) -L$(CUDA_LIBS_DIR) $(CUDA_LIBS)
else
gcmbw: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(DEFINES) $(INCLUDES) $(LIBS)
endif

ifeq ($(HAVE_NVCC),1)
%.o: %.cu
	$(NVCC) -o $@ -c $^ $(CFLAGS) $(DEFINES) $(CUDA_DEFINES) $(INCLUDES)
endif

%.o: %.c 
	$(CC) -o $@ -c $^ $(CFLAGS) $(DEFINES) $(INCLUDES)

.PHONY:	clean
clean:
	$(RM) $(OBJS) gcmbw
