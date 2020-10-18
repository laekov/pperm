#!/bin/bash
for i in {8..15}
do
	for j in smart_recr lex sjt
	do
		mpirun -n 128 ./pperm -l $i ${j}_cpu_mpi_simd | tee logs/l_${i}_algo_mpi_simd_$j.log
	done
done
