CC = gcc
CFLAGS = -fPIC -std=c99 -O3 -mavx2 -march=native -Wall -Wextra -pedantic -Wshadow -I. -I./SIMDxorshift/include
PFLAGS = -lpthread

TARGET = pi.out
SOURCES = pi.c SIMDxorshift/src/simdxorshift128plus.c SIMDxorshift/src/xorshift128plus.c
HEADERS = include/simdxorshift128plus.h include/xorshift128plus.h
OBJECTS = ${SOURCES:.c=.o}

all: ${TARGET}

${TARGET}:${OBJECTS}
	${CC} ${CFLAGS} $^ -o $@ ${PFLAGS}

%.o:%.c
	${CC} ${CFLAGS} -c $^ -o $@ ${PFLAGS}

.PHONY: clean

clean:
	rm -f *.o *.out