#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>

using namespace std;

cuFloatComplex *pcmul(cuFloatComplex *a, float *b, int m, int n) {
    cuFloatComplex *res = new cuFloatComplex[m*n];

    float real, imag;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            real = cuCrealf(a[i*n+j]);
            imag = cuCimagf(a[i*n+j]);
            res[i*n+j] = make_cuFloatComplex(b[i*n+j]*real, b[i*n+j]*imag);
        }
    }

    return res;
}

__global__ void pcmul_kernel(cuFloatComplex *res, cuFloatComplex *a, float *b, int m, int n) {

    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<m*n) {
        float real, imag;
        real = cuCrealf(a[i]);
        imag = cuCimagf(a[i]);
        res[i] = make_cuFloatComplex(b[i]*real, b[i]*imag);
    }
}

cuFloatComplex *pcmul_gpu(cuFloatComplex *a, float *b, int m, int n) {

    // host
    cuFloatComplex *h_res;
    // device
    cuFloatComplex *d_a, *d_res;
    float *d_b;

    struct timeval tb, te;
    unsigned long long bb, e;

    h_res = new cuFloatComplex[m*n];

    cudaMalloc(&d_res, m*n*sizeof(cuFloatComplex));
    cudaMalloc(&d_a, m*n*sizeof(cuFloatComplex));
    cudaMalloc(&d_b, m*n*sizeof(float));

    gettimeofday(&tb, NULL);

    cudaMemcpy(d_a, a, m*n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m*n*sizeof(float), cudaMemcpyHostToDevice);

    gettimeofday(&te, NULL);
    bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    cout << "copy to device " << e-bb << endl;

    gettimeofday(&tb, NULL);
    pcmul_kernel<<<m,n>>>(d_res, d_a, d_b, m, n);
    gettimeofday(&te, NULL);
    bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    cout << "kernel compute " << e-bb << endl;

    gettimeofday(&tb, NULL);
    cudaMemcpy(h_res, d_res, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    gettimeofday(&te, NULL);
    bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    cout << "copy to host " << e-bb << endl;

    cudaFree(d_res);
    cudaFree(d_a);
    cudaFree(d_b);

    return h_res;
}

int main(int argc, char **argv) {

    int m, n;
    cuFloatComplex *iq, *mul;
    float *h;

    m = 1024;
    n = 512;

    iq = new cuFloatComplex[m*n];
    h = new float[m*n];

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            iq[i*n+j] = make_cuFloatComplex((float)i, (float)j);
            h[i*n+j] = i;
            //cout << "(" << cuCrealf(iq[i*n+j]) << "," << cuCimagf(iq[i*n+j]) << ") ";
        }
        //cout << endl;
    }

    /*for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << h[i*n+j] << " ";
        }
        cout << endl;
    }*/

    struct timeval tb, te;
    unsigned long long b, e;

    gettimeofday(&tb, NULL);
    mul = pcmul(iq, h, m, n);
    gettimeofday(&te, NULL);
    b = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    cout << e-b << endl;

    gettimeofday(&tb, NULL);
    mul = pcmul_gpu(iq, h, m, n);
    gettimeofday(&te, NULL);
    b = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    cout << e-b << endl;

    /*for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << "(" << cuCrealf(mul[i*n+j]) << "," << cuCimagf(mul[i*n+j]) << ") ";
        }
        cout << endl;
    }*/

    delete[] iq;
    delete[] h;
    delete[] mul;

    return 0;
}
