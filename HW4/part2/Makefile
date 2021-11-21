SRC = main.cc matmul_impl.cc

all: $(SRC)
	mpicxx $(SRC) -o matmul

.PHONY: clean

clean:
	rm -f matmul