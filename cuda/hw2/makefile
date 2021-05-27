all:
	gcc cpu.cpp -o cpu.out
	nvcc gpu.cu -o gpu.out
run:
	gcc cpu.cpp -o cpu.out
	nvcc gpu.cu -o gpu.out
	./cpu.out
	./gpu.out
clean:
	rm *.out
	rm *.bmp