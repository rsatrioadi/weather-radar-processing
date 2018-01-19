#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>

using namespace std;

__global__ void __apply_hamming(cuDoubleComplex *a, double *b, int m, int n) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double real, imag;
    real = a[i].x;
    imag = a[i].y;
    a[i] = make_cuDoubleComplex(b[i%(m*n)]*real, b[i%(m*n)]*imag);
}

/*cuDoubleComplex *pcmul_gpu(cuDoubleComplex *a, double *b, int m, int n) {

    // host
    cuDoubleComplex *h_res;
    // device
    cuDoubleComplex *d_a;
    double *d_b;

    // struct timeval tb, te;
    // unsigned long long bb, e;

    h_res = new cuDoubleComplex[2*m*n];

    cudaMalloc(&d_a, 2*m*n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_b, m*n*sizeof(double));

    // gettimeofday(&tb, NULL);

    cudaMemcpy(d_a, a, 2*m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m*n*sizeof(double), cudaMemcpyHostToDevice);

    // gettimeofday(&te, NULL);
    // bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    // e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    // cout << "copy to device " << e-bb << endl;

    // gettimeofday(&tb, NULL);
    pcmul_kernel<<<2*m,n>>>(d_a, d_b, m, n);
    // gettimeofday(&te, NULL);
    // bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    // e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    // cout << "kernel compute " << e-bb << endl;

    // gettimeofday(&tb, NULL);
    cudaMemcpy(h_res, d_a, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // gettimeofday(&te, NULL);
    // bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    // e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    // cout << "copy to host " << e-bb << endl;

    cudaFree(d_a);
    cudaFree(d_b);

    return h_res;
}*/
