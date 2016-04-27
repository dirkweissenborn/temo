#!/bin/sh
for cell in "MORU" "GRU"; do
	for k in 1 2 4 8 16 32 64 128 256; do
		echo "Running $cell $k"
     	python3 train_logic.py $cell $k
	done
done
