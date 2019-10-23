#Makefile for gcmbw

CC = gcc
NVCC = nvcc
CFLAGS = -O2 -g 
RM = rm -rf
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
INCLUDES = ./includes
LIBS = pthread
HAVE_NVCC = $(shell if [ -x $(NVCC) ]; then echo "1"; else echo "0"; fi;)

ifeq ($(HAVE_NVCC),1)
	DEFINES = -DGPU=1
else
	DEFINES = 
endif

ifeq ($(HAVE_NVCC),1)
gcmbw: $(OBJS)
	$(NVCC) -o $@ $^ $(CFLAGS) $(DEFINES) -I$(INCLUDES) -l$(LIBS)
else
gcmbw: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(DEFINES) -I$(INCLUDES) -l$(LIBS)
endif

%.o: %.c 
	$(CC) -o $@ -c $^ $(CFLAGS) $(DEFINES) -I$(INCLUDES) -l$(LIBS)

ifeq ($(HAVE_NVCC),1)
%.o: %.cu
	$(NVCC) -o $@ -c $(CFLAGS) $(DEFINES) -I$(INCLUDES)
endif

.PHONY:	clean
clean:
	$(RM) $(OBJS) gcmbw