NVCC		= nvcc
CC		= g++
#CU_FLAGS	= -O3 -g -deviceemu
#~ CU_FLAGS	= -O3 -g --ptxas-options=-v -arch compute_20 -code sm_20
CU_FLAGS	= -O3 -g --ptxas-options=-v -arch compute_10 -code sm_10
#~ CC_FLAGS	= -O3 -m64 -Wall
CC_FLAGS	= -O3 -Wall

CU_SOURCES	= wave_kernel.cu
CC_SOURCES	= assign6_1.cc timer.cc file.cc generatedata.cc

CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o: %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(CU_FLAGS) --ptx $< -o $@

assign6_1:	$(CU_OBJECTS) $(CC_OBJECTS)
	$(NVCC) $^ -o $@

ptx: $(CU_PTX) 

plot: result.txt
	gnuplot plot.gnp
	display plot.png

run: assign6_1
	prun -v -np 1 -native ='-l gpu=GTX480' wave

clean:
	rm -f *.o assign6_1 *~ plot.png result.txt
