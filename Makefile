all:
	g++ -c sector.cpp udpbroadcast.cpp floats.c tcp.cpp dimension.cpp
	nvcc -o process gpu_1fp_streamcasc.cu -lfftw3f -lcufft -lm sector.o udpbroadcast.o floats.o tcp.o
	nvcc -o rpv2 rpv2.cu -lfftw3f -lcufft -lm -lzmq floats.o dimension.o -I msgpack-c/include

