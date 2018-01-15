#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;

float *cf_hamming_cpu(int m, int n) {

    // dict
    float pr = .0;
    float pd = .0;
    float k = -1 / (16383.5 * m * n * sqrt(50));
    float *M;

    // init
    M = (float*)malloc(m*n*sizeof(float));

    // calc pr
    for (int i=0; i<m; i++) {
        pr += pow(0.53836 - 0.46164 * cos(2 * M_PI * (i) / (m - 1)), 2);
    }
    pr /= m;
    cout << pr << endl;

    // calc pd
    for (int j=0; j<n; j++) {
        pd += pow(0.53836 - 0.46164 * cos(2 * M_PI * (j) / (n - 1)), 2);
    }
    pd /= n;
    cout << pd << endl;

    // populate M
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            M[i*n+j] = (0.53836 - 0.46164 * cos(2 * M_PI * (i) / (m - 1))) 
            * (0.53836 - 0.46164 * cos(2 * M_PI * (j) / (n - 1))) 
            * k / sqrt(pr * pd);
        }
    }

    return M;
}

__global__ void seq_kernel(float *s, float *s2, int n) {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) {
        float t = 0.53836 - 0.46164*cos(2*M_PI*idx/(n - 1));
        s[idx] = t;
        s2[idx] = t*t;
    }
}

__global__ void red_kernel(float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void seqred_kernel(float *s, float *s2, float *o, int n) {

    extern __shared__ float fdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float t = 0.53836 - 0.46164 * cos(2 * M_PI * (i) / (n - 1));
        s[i] = t;
        s2[i] = t*t;
    }

    fdata[tid] = s2[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            fdata[tid] += fdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) o[blockIdx.x] = fdata[0];
}

float *cf_hamming_gpu(int m, int n) {

    float *h_sm, *h_sm2, *h_sn, *h_sn2, *h_om, *h_on;
    float *d_sm, *d_sm2, *d_sn, *d_sn2, *d_om, *d_on;

    h_sm = (float*)malloc(m*sizeof(float));
    h_sm2 = (float*)malloc(m*sizeof(float));
    // h_om = (float*)malloc(m*sizeof(float));

    cudaMalloc(&d_sm, m*sizeof(float)); 
    cudaMalloc(&d_sm2, m*sizeof(float));
    // cudaMalloc(&d_om, m*sizeof(float)); 

    h_sn = (float*)malloc(n*sizeof(float));
    h_sn2 = (float*)malloc(n*sizeof(float));
    // h_on = (float*)malloc(n*sizeof(float));

    cudaMalloc(&d_sn, n*sizeof(float)); 
    cudaMalloc(&d_sn2, n*sizeof(float));
    // cudaMalloc(&d_on, n*sizeof(float)); 

    seqred_kernel<<<1,m>>>(d_sm, d_sm2, m);
    cudaMemcpy(h_sm, d_sm, m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sm2, d_sm2, m*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_om, d_om, m*sizeof(float), cudaMemcpyDeviceToHost);

    seq_kernel<<<1,n>>>(d_sn, d_sn2, n);
    cudaMemcpy(h_sn, d_sn, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sn2, d_sn2, n*sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i=0; i<m; i++) {
        sum += h_sm2[i];
    }
    float pr = h_om[0] / m;

    float sum = 0;
    for (int j=0; j<n; j++) {
        sum += h_sn2[j];
    }
    float pd = h_on[0] / n;

    cout << pr << endl << pd << endl;

    cudaFree(h_sm);
    cudaFree(h_sm2);
    cudaFree(h_sn);
    cudaFree(h_sn2);
}

int main()
{
    // dict
    int m = 1024;
    int n = 512;
    float *cpu_hamming;
    //float *gpu_hamming;

    // init
    cpu_hamming = (float*)malloc(m*n*sizeof(float));
    gpu_hamming = (float*)malloc(m*n*sizeof(float));

    //cout << "cpu" << endl;
    cpu_hamming = cf_hamming_cpu(m,n);
    //cout << "gpu" << endl;
    gpu_hamming = cf_hamming_gpu(m,n);

    // free
    free(cpu_hamming);
    free(gpu_hamming);

    /*for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << cpu_hamming[i*n+j] << " ";
        }
        cout << endl;
    }*/
        /*
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
        int N = 16;
        int SIZE = N*N;

    // Allocate memory on the host
        vector<float> h_A(SIZE);
        vector<float> h_B(SIZE);
        vector<float> h_C(SIZE);

    // Initialize matrices on the host
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                h_A[i*N+j] = sin(i);
                h_B[i*N+j] = cos(j);
            }
        }

    // Allocate memory on the device
        dev_array<float> d_A(SIZE);
        dev_array<float> d_B(SIZE);
        dev_array<float> d_C(SIZE);

        d_A.set(&h_A[0], SIZE);
        d_B.set(&h_B[0], SIZE);

        matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
        cudaDeviceSynchronize();

        d_C.get(&h_C[0], SIZE);
        cudaDeviceSynchronize();

        float *cpu_C;
        cpu_C=new float[SIZE];

    // Now do the matrix multiplication on the CPU
        float sum;
        for (int row=0; row<N; row++){
            for (int col=0; col<N; col++){
                sum = 0.f;
                for (int n=0; n<N; n++){
                    sum += h_A[row*N+n]*h_B[n*N+col];
                }
                cpu_C[row*N+col] = sum;
            }
        }

        double err = 0;
    // Check the result and make sure it is correct
        for (int ROW=0; ROW < N; ROW++){
            for (int COL=0; COL < N; COL++){
                err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
            }
        }

        cout << "Error: " << err << endl;
*/
        return 0;
    }