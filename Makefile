#Makefile for gcmbw

CC = gcc
NVCC = nvcc
CFLAGS = -O2
RM = rm -rf
SRCS = $(wildcard *.c)
CUDA_SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.c=.o)
OBJS += $(CUDA_SRCS:.cu=.o)
INCLUDES = ./includes
LIBS = pthread
CUDA_LIBS_DIR = /usr/local/cuda/lib
CUDA_LIBS = -lcuda -lcudart
HAVE_NVCC = $(shell $(NVCC) -V | if [ $?==0 ]; then echo "1"; else echo "0"; fi)

ifeq ($(HAVE_NVCC),1)
	DEFINES = -DGPU=1 
	CUDA_DEFINES = -D_FORCE_INLINES -gencode arch=compute_32,code=compute_32
else
	DEFINES = 
endif

ifeq ($(HAVE_NVCC),1)
gcmbw: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(DEFINES) -I$(INCLUDES) -l$(LIBS) -L$(CUDA_LIBS_DIR) $(CUDA_LIBS)
else
gcmbw: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(DEFINES) -I$(INCLUDES) -l$(LIBS)
endif

ifeq ($(HAVE_NVCC),1)
%.o: %.cu
	$(NVCC) -o $@ -c $^ $(CFLAGS) $(DEFINES) $(CUDA_DEFINES) -I$(INCLUDES)
endif

%.o: %.c 
	$(CC) -o $@ -c $^ $(CFLAGS) $(DEFINES) -I$(INCLUDES) -l$(LIBS)

.PHONY:	clean
clean:
	$(RM) $(OBJS) gcmbw
