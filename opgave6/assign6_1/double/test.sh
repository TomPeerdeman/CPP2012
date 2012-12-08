#!/bin/bash

for t in 10 100 1000 
do
	echo Start $t
	tar -cf result$t.tar test.sh

	for i in 1000 10000 100000 1000000 10000000
	do
		echo $t-$i
		touch res$i.txt
		echo $i-$t-512 > res$i.txt

		for n in {1..10..1}
		do
			./assign6_1 $i $t 512 sin >> res$i.txt
		done
		mv result.txt result$i.txt
		tar -rf result$t.tar result$i.txt res$i.txt
		rm -f result$i.txt res$i.txt
	done
	
done

tar -cvzf results.tar.gz result10.tar result100.tar result1000.tar
rm result*.tar