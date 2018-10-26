#!/bin/bash

configs=$(ls configs/)
for config in $configs; do
	echo $config >> results.txt
	for i in `seq 1 10`; do
		result=$(python test_agent.py -a demo_agent+DemoAgent -c configs/$config)
		echo $result >> results.txt
	done
done