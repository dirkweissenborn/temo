#!/bin/sh
for cell in "MORU" "GRU"; do
	for k in 2 4 8 16; do	  
     	python3 train_logic.py $cell $k
	done
done
