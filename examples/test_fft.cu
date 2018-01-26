#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <cufft.h>

using namespace std;

#define M 8
#define N 4

int main() {

    cuDoubleComplex *a, *d_a;

    a = new cuDoubleComplex[2*M*N];
    cudaMalloc(&d_a, 2*M*N*sizeof(cuDoubleComplex));

    for (int i=0; i<2*M*N; i++) {
        a[i] = make_cuDoubleComplex(i+N, 0);
    }

    cout << endl;
    for (int i=0; i<2*M; i++) {
        for (int j=0; j<N; j++) {
            cout << "(" << a[i*N+j].x << "," << a[i*N+j].y << ") ";
        }
        cout << endl;
    }

    fftw_complex *fft_doppler_buffer;
    fftw_plan fft_doppler_plan;
    fft_doppler_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fft_doppler_plan = fftw_plan_dft_1d(N, fft_doppler_buffer, fft_doppler_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i=0; i<2*M; i++) {

        for (int j=0; j<N; j++) {
            fft_doppler_buffer[j][0] = a[i*N+j].x;
            fft_doppler_buffer[j][1] = a[i*N+j].y;
        }
        fftw_execute(fft_doppler_plan);
        for (int j=0; j<N; j++) {
            a[i*N+j] = make_cuDoubleComplex(fft_doppler_buffer[j][0], fft_doppler_buffer[j][1]);
        }
    }
    fftw_destroy_plan(fft_doppler_plan);
    fftw_free(fft_doppler_buffer);

    cout << endl;
    for (int i=0; i<2*M; i++) {
        for (int j=0; j<N; j++) {
            cout << "(" << a[i*N+j].x << "," << a[i*N+j].y << ") ";
        }
        cout << endl;
    }

    for (int i=0; i<2*M*N; i++) {
        a[i] = make_cuDoubleComplex(i+N, 0);
    }

    cout << endl;
    for (int i=0; i<2*M; i++) {
        for (int j=0; j<N; j++) {
            cout << "(" << a[i*N+j].x << "," << a[i*N+j].y << ") ";
        }
        cout << endl;
    }

    cudaMemcpy(d_a, a, 2*M*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int nn[] = { N };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = N, odist = N; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = 2*M;                      // --- Number of batched executions
    // cufftPlanMany(&handle, rank, nn, 
    //               inembed, istride, idist,
    //               onembed, ostride, odist, CUFFT_Z2Z, batch);

    cufftPlan1d(&handle, N, CUFFT_Z2Z, 2*M);
    cufftExecZ2Z(handle,  d_a, d_a, CUFFT_FORWARD);
    cudaMemcpy(a, d_a, 2*M*N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cout << endl;
    for (int i=0; i<2*M; i++) {
        for (int j=0; j<N; j++) {
            cout << "(" << a[i*N+j].x << "," << a[i*N+j].y << ") ";
        }
        cout << endl;
    }

    return 0;
}
