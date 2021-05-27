all:
	nvcc RBFKernel.cu -o baseline.out
	nvcc RBFKernel-opt.cu -o opt_sharedmem.out
	nvcc RBFKernel-opt2.cu -o opt_cat.out
	nvcc RBFKernel-opt3.cu -o opt_seq_thread.out
	nvcc RBFKernel-opt4.cu -o opt_seq_block.out
	nvcc RBFKernel-opt5.cu -o opt_seq_nidle.out
	nvcc RBFKernel-opt6.cu -o opt_reduct6.out
clean:
	rm *.out