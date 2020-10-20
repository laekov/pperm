#!/bin/bash

gen() {
	for t in {8..15}
	do
		for algo in recr lex sjt
		do
			echo $t $algo
		done
	done
}

mkdir -p logs

gen | xargs -P 8 -n 2 --process-slot-var=WORKER_ID \
	bash -c 'CUDA_VISIBLE_DEVICES=$WORKER_ID \
	numactl -C $(expr $WORKER_ID \* 4) \
	./pperm -l $0 $1_gpu | tee logs/l_$0_algo_$1_gpu.log'

for i in {8..15}
do
	mv logs/l_${i}_algo_recr_gpu.log logs/l_${i}_algo_smart_recr_gpu.log
done
