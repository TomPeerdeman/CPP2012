#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>

#include "seq_max.h"
#include "timer.h"

using namespace std;

float computeMaxSeq(int length, float *list){
	if(length <= 0){
		return 0.0;
	}
	timer timerMax("Seq max timer");
	
	timerMax.start();
	
	float max = list[0];
	int i;
	for(i = 1; i < length; i++){
		if(list[i] > max){
			max = list[i];
		}
	}
	
	timerMax.stop();
	
	cout << timerMax;

	return max;
}
