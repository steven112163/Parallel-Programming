SRC = main.cc matmul_impl.cc

all: $(SRC)
	mpicxx $(SRC) -o matmul -O3

.PHONY: clean

clean:
	rm -f matmul