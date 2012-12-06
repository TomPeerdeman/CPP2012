#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#include "file.h"
#include "generatedata.h"
#include "wave_kernel.h"

float sin(float x){
	return (float) sin((double) x);
}

int main(int argc, char **argv){
	float *old, *cur, *next, *ret;
	int i_max, t_max;
	
	if(argc < 3){
		printf("Usage: %s i_max t_max [initial_data]\n", argv[0]);
		printf(" - i_max: number of discrete amplitude points, should be >2\n");
		printf(" - t_max: number of discrete timesteps, should be >=1\n");
		printf(" - initial_data: select what data should be used for the first "
		  "two generation.\n");
		printf("   Available options are:\n");
		printf("    * sin: one period of the sinus function at the start.\n");
		printf("    * sinfull: entire data is filled with the sinus.\n");
		printf("    * gauss: a single gauss-function at the start.\n");
		printf("    * file <2 filenames>: allows you to specify a file with on "
		  "each line a float for both generations.\n");

		return EXIT_FAILURE;
	}

	i_max = atoi(argv[1]);
	t_max = atoi(argv[2]);

	if(i_max < 3){
		printf("argument error: i_max should be >2.\n");
		return EXIT_FAILURE;
	}
	if(t_max < 0){
		printf("argument error: t_max should be >=1.\n");
		return EXIT_FAILURE;
	}
	
	old = new float[i_max];
    cur = new float[i_max];
    next = new float[i_max];
	
	memset(old, 0, i_max * sizeof(float));	
	memset(cur, 0, i_max * sizeof(float));	
	memset(next, 0, i_max * sizeof(float));	
	
	if(argc > 3){
		if(strcmp(argv[3], "sin") == 0){
			fill(old, 1, i_max/4, 0, 2*3.14, sin);
			fill(cur, 2, i_max/4, 0, 2*3.14, sin);
		}else if(strcmp(argv[3], "sinfull") == 0){
			fill(old, 1, i_max-2, 0, 10*3.14, sin);
			fill(cur, 2, i_max-3, 0, 10*3.14, sin);
		}else if(strcmp(argv[3], "gauss") == 0){
			fill(old, 1, i_max/4, -3, 3, gauss);
			fill(cur, 2, i_max/4, -3, 3, gauss);
		}else if(strcmp(argv[3], "file") == 0){
			if(argc < 6){
				printf("No files specified!\n");
				return EXIT_FAILURE;
			}
			file_read_float_array(argv[4], old, i_max);
			file_read_float_array(argv[5], cur, i_max);
		}else{
			printf("Unknown initial mode: %s.\n", argv[3]);
			return EXIT_FAILURE;
		}
	}else{
		/* Default to sinus. */
		fill(old, 1, i_max/4, 0, 2*3.14, sin);
		fill(cur, 2, i_max/4, 0, 2*3.14, sin);
	}
	
	ret = computeWaveCuda(i_max, t_max, 128, old, cur, next);
	
	file_write_float_array("result.txt", ret, i_max);
	
	delete[] old;
	delete[] cur;
	delete[] next;
	
	return 0;
}