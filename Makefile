all:
	g++ -c sector.cpp udpbroadcast.cpp floats.c tcp.cpp
	nvcc -o process gpu_1fp_streamcasc.cu -lfftw3f -lcufft -lm sector.o udpbroadcast.o floats.o tcp.o
