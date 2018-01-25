#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>

using namespace std;

__global__ void __sumcomplex(cuDoubleComplex *g_idata, cuDoubleComplex *g_odata) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    g_odata[i] = make_cuDoubleComplex(g_idata[i].x, g_idata[i].y);
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            g_odata[i] = make_cuDoubleComplex(g_odata[i].x+g_odata[i + s].x, g_odata[i].y+g_odata[i + s].y);
        }
        __syncthreads();
    }
}

__global__ void __sum_inplace(cuDoubleComplex *g_idata) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            g_idata[i] = make_cuDoubleComplex(g_idata[i].x+g_idata[i + s].x, 0);
        }
        __syncthreads();
    }
}

int main() {

	const int m=16, n=8;

	cuDoubleComplex *h_data = new cuDoubleComplex[m*n];
	cuDoubleComplex *h_sum = new cuDoubleComplex[m*n];
	cuDoubleComplex *d_data, *d_sum;

    cudaMalloc(&d_data, m*n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_sum, m*n*sizeof(cuDoubleComplex));

    for (int i=0; i<m; i++) {
	    for (int j=0; j<n; j++) {
	    	h_data[i*n+j] = make_cuDoubleComplex(i+j, 0);
	    }
    }

    cout << "in:" << endl;
    for (int i=0; i<m; i++) {
	    for (int j=0; j<n; j++) {
			cout << "(" << h_data[i*n+j].x << "," << h_data[i*n+j].y << ") ";
		}
	    cout << endl;
    }

    cudaMemcpy(d_data, h_data, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    __sum_inplace<<<m,n>>>(d_data);
    cudaMemcpy(h_data, d_data, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cout << "out:" << endl;
    for (int i=0; i<m; i++) {
	    for (int j=0; j<n; j++) {
	    	cout << "(" << h_data[i*n+j].x << "," << h_data[i*n+j].y << ") ";
	    }
	    cout << endl;   
    }

	return 0;
}
