/*
  A program to generate an image of the Mandelbrot set. 

  Usage: ./mandelbrot > output
         where "output" will be a binary image, 1 byte per pixel
         The program will print instructions on stderr as to how to
         process the output to produce a JPG file.

  Michael Ashley / UNSW / 13-Mar-2003
*/

// Define the range in x and y here:

const double yMin = -1.0;
const double yMax = +1.0;
const double xMin = -2.0;
const double xMax = +0.5;

// And here is the resolution:

const double dxy = 0.005;

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "omp.h"

#include "timer.h"

int main(void) {
	timer_start();
	
	omp_set_num_threads(8);
	
    double cx, cy;
    double zx, zy, new_zx;
	double time;
    int nx, ny;
	int i;
	unsigned char *buf;
	
	// Now calculate the image dimensions. We use exactly the same
    // for loops as above, to guard against any potential rounding errors.

    nx = 0;
    ny = 0;
    for (cx = xMin; cx < xMax; cx += dxy) {
		nx++;
		
    }
    for (cy = yMin; cy < yMax; cy += dxy) {
        ny++;
    }
	
	buf = malloc(ny * nx * sizeof(unsigned char));
	if(buf == NULL){
		fprintf(stderr, "Could not allocate enough memory, aborting.\n");
		return 1;
	}
	
    // The Mandelbrot calculation is to iterate the equation
    // z = z*z + c, where z and c are complex numbers, z is initially
    // zero, and c is the coordinate of the point being tested. If
    // the magnitude of z remains less than 2 for ever, then the point
    // c is in the Mandelbrot set. We write out the number of iterations
    // before the magnitude of z exceeds 2, or UCHAR_MAX, whichever is
    // smaller.

	int j, idx, n;
	//#pragma omp parallel for private(cy, cx, i, j, zx, zy, n)
    for (i = 0; i < ny; i++) {
		cy = i * dxy + yMin;
		
        for (j = 0; j < nx; j++) {
			idx = ny * i + j;
			cx = j * dxy + xMin;
            zx = 0.0; 
            zy = 0.0; 
            n = 0;
            while ((zx*zx + zy*zy < 4.0) && (n != UCHAR_MAX)) {
                new_zx = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
				zx = new_zx;
				n++;
            }
			buf[idx] = n;
        }
    }
	
	// Write the result to stdout
	fwrite(buf, sizeof(unsigned char), nx * ny, stdout);

	free(buf);

    fprintf (stderr, "To process the image: convert -depth 8 -size %dx%d gray:output out.jpg\n",
	     nx, ny);
	
	time = timer_end();
    fprintf(stderr, "Took %g seconds\n", time);
    return 0;
}

