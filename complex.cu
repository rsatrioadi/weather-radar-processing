#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <complex.h>

using namespace std;

complex<float> *pcmul(complex<float> *a, float *b, int m, int n) {
    complex<float> *res = new complex<float>[m*n];

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            res[i*n+j] = b[i*n+j] * a[i*n+j];
        }
    }

    return res;
}

complex<float> *pcmul_gpu(complex<float> *a, float *b, int m, int n) {
    
    complex<float> *res = new complex<float>[m*n];

    cuFloatComplex *d_a, *d_res, *h_a, *h_res;
    float *d_b;

    h_a = new cuFloatComplex[m*n];
    h_res = new cuFloatComplex[m*n];
    cudaMalloc(&d_a, m*n*sizeof(cuFloatComplex));
    cudaMalloc(&d_b, m*n*sizeof(float));
    cudaMalloc(&d_res, m*n*sizeof(cuFloatComplex));

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            h_a[i*n+j] = make_cuFloatComplex(a[i*n+j].real(), a[i*n+j].imag());
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    delete[] h_res;

    return res;
}

int main(int argc, char **argv) {

    int m, n;
    complex<float> *iq, *mul;
    float *h;

    m = 8;
    n = 4;

    iq = new complex<float>[m*n];
    h = new float[m*n];

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            iq[i*n+j] = {(float)i,(float)j};
            h[i*n+j] = i;
            cout << iq[i*n+j] << " ";
        }
        cout << endl;
    }

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << h[i*n+j] << " ";
        }
        cout << endl;
    }

    mul = pcmul(iq, h, m, n);

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << mul[i*n+j] << " ";
        }
        cout << endl;
    }

    delete[] iq;
    delete[] h;
    delete[] mul;

    return 0;
}
