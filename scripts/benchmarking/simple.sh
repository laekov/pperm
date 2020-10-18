#!/bin/bash

gen() {
	for t in {8..15}
	do
		for algo in smart_recr lex sjt
		do
			echo $t $algo
		done
	done
}

gen | xargs -P 128 -n 2 --process-slot-var=WORKER_ID \
	bash -c 'numactl -C $(expr $WORKER_ID) \
	./pperm -l $0 $1_cpu_simple | tee logs/l_$0_algo_$1_simple.log'
