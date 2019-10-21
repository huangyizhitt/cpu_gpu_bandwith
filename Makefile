#Makefile for gcmbw

CC = gcc
CFLAGS = -O2 -g 
RM = rm -rf
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
INCLUDES = ./includes
LIBS = pthread

gcmbw: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) -I$(INCLUDES) -l$(LIBS)

%.o: %.c 
	$(CC) -o $@ -c $^ $(CFLAGS) -I$(INCLUDES) -l$(LIBS)

.PHONY:	clean
clean:
	$(RM) $(OBJS) gcmbw