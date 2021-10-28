TARGET=pi

all: $(TARGET).c
	make clean
	gcc -o $(TARGET).out $(TARGET).c -lpthread

clean:
	rm -f $(TARGET).out