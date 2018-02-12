#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <cufft.h>
#include <sys/time.h>

using namespace std;

#define k_rangeres 30
#define k_calib 1941.05

#define RESULT_SIZE 2

double *generate_hamming_coef(int m, int n) {

    // Calculate normalization power on range cell
    double p_range=0;
    for(int i=0; i < m; i++) {
        p_range=p_range+pow(0.53836-0.46164*cos(2*M_PI*(i)/(m-1)), 2.0);
    }
    p_range=p_range/m;

    // Calculate normalization power on Doppler cell
    double p_doppler=0;
    for(int j=0; j < n; j++) {
        p_doppler=p_doppler+pow(0.53836-0.46164*cos(2*M_PI*(j)/(n-1)), 2.0);
    }
    p_doppler=p_doppler/n;

    // Constant since FFT is not normalized and the power is computed w.r.t. 50ohm
    const double K_wind = -1/(16383.5*m*n*sqrt(50));
    const double c = K_wind/sqrt(p_range*p_doppler);

    // Generate elements
    double *_hamming_coef= new double[m*n];
    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            _hamming_coef[i*n+j] = (0.53836-0.46164*cos(2*M_PI*(i)/(m-1))) * (0.53836-0.46164*cos(2*M_PI*(j)/(n-1))) * c;
        }
    }

    return _hamming_coef;
}

double *generate_ma_coef(int n){
    double *_ma_coef = new double[n];
    double _sum = 0.0;
    for(int i=0; i < n; i++) {
        _ma_coef[i]=exp(-(pow(i-((n-1)/2), 2.0))/2);
        _sum += _ma_coef[i];
    }
    for(int i=0; i < n; i++){
        _ma_coef[i] = _ma_coef[i]/_sum;
    }
    return _ma_coef;
}

__global__ void __apply_hamming(cuDoubleComplex *a, double *b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double real, imag;
    real = a[i].x;
    imag = a[i].y;
    a[i] = make_cuDoubleComplex(b[i]*real, b[i]*imag);
}

__global__ void __apply_ma(cuDoubleComplex *inout, cuDoubleComplex *macoef) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

    double ii, qq, mi, mq;
    ii = inout[i*n+j].x;
    qq = inout[i*n+j].y;
    mi = macoef[j].x;
    mq = macoef[j].y;

    inout[i*n+j] = make_cuDoubleComplex(ii*mi-qq*mq, ii*mq+qq*mi);
}

__global__ void __conjugate(cuDoubleComplex *a) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = make_cuDoubleComplex(a[i].x, a[i].y * -1);
}

__global__ void __shift(cuDoubleComplex *inout, int n) {
    unsigned int i = blockIdx.x, j = threadIdx.x;

    cuDoubleComplex temp = inout[i*n+j];
    inout[i*n+j] = inout[i*n+(j+n/2)];
    inout[i*n+(j+n/2)] = temp;
}

__global__ void __trim(cuDoubleComplex *inout, int n) {
    unsigned int i = blockIdx.x, j = n-threadIdx.x-1;
    inout[i*n+j] = make_cuDoubleComplex(0, 0);
}

__global__ void __abssqr(cuDoubleComplex *inout, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double real, imag;
    real = inout[i].x;
    imag = inout[i].y;
    inout[i] = make_cuDoubleComplex(real*real+imag*imag, 0);    
}

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

__global__ void __avgconj(cuDoubleComplex *in, cuDoubleComplex *sum) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

    double avgx = sum[i*n].x/n;
    double avgy = sum[i*n].y/n;
    in[i*n+j] = make_cuDoubleComplex(in[i*n+j].x-avgx, (in[i*n+j].y-avgy)*-1);
}

__global__ void __scale_real(cuDoubleComplex *inout) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

    inout[i*n+j] = make_cuDoubleComplex(inout[i*n+j].x/n, 0);
}

__global__ void __calcresult(cuDoubleComplex *hh, cuDoubleComplex *vv, double *out, int n) {
    unsigned int i = blockIdx.x;

    double z = pow(i*k_rangeres, 2.0) * k_calib * hh[i*n].x;
    double zdb = 10 * log10(z);
    double zdr = 10 * (log10(hh[i*n].x)-log10(vv[i*n].x));
    out[i*RESULT_SIZE+0] = zdb;
    out[i*RESULT_SIZE+1] = zdr;
}

void tick(timeval *begin) {
    gettimeofday(begin, NULL);
}

void tock(timeval *begin, timeval *end, string caption) {
    unsigned long long bb, e;

    gettimeofday(end, NULL);
    bb = (unsigned long long)(begin->tv_sec) * 1000000 + (unsigned long long)(begin->tv_usec) / 1;
    e = (unsigned long long)(end->tv_sec) * 1000000 + (unsigned long long)(end->tv_usec) / 1;

    cout << caption << ": " << e-bb << endl;
}

int main(int argc, char **argv) {

    struct timeval tb, te;

    tick(&tb);

    cuDoubleComplex *iqhh, *iqvv;
    double *result;

    const int m = 1024; // cell
    const int n = 512;  // sweep

    const int ma_count = 7;

    iqhh = new cuDoubleComplex[m*n];
    iqvv = new cuDoubleComplex[m*n];
    result = new double[(m/2)*RESULT_SIZE];

    double a, b;

    // Generate Hamming coefficients
    const double *hamming_coef = generate_hamming_coef(m, n);

    // Generate MA coefficients
    double *ma_coef = generate_ma_coef(ma_count);
    fftw_complex *_fft_ma = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan fft_ma_plan = fftw_plan_dft_1d(n, _fft_ma, _fft_ma, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<ma_count; j++) {
        _fft_ma[j][0] = ma_coef[j];
        _fft_ma[j][1] = 0;
    }
    for (int j=ma_count; j<n; j++) {
        _fft_ma[j][0] = 0;
        _fft_ma[j][1] = 0;
    }
    fftw_execute(fft_ma_plan);
    fftw_destroy_plan(fft_ma_plan);
    cuDoubleComplex *fft_ma;
    fft_ma = new cuDoubleComplex[n];
    for (int j=0; j<n; j++) {
        fft_ma[j] = make_cuDoubleComplex(_fft_ma[j][0], _fft_ma[j][1]);
    }
    fftw_free(_fft_ma);

    // Device buffers
    /*__constant__*/ double *d_hamming;
    /*__constant__*/ cuDoubleComplex *d_ma;
    cuDoubleComplex *d_iqhh, *d_iqvv;
    cuDoubleComplex *d_sum;
    double *d_result;
    //double *d_powhh, *d_powvv;

    cudaMalloc(&d_hamming, m*n*sizeof(double));
    cudaMalloc(&d_ma, n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_iqhh, m*n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_iqvv, m*n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_sum, m*n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_result, (m/2)*RESULT_SIZE*sizeof(double));

    cudaMemcpy(d_hamming, hamming_coef, m*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma, fft_ma, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // CUFFT initialization
    cufftHandle fft_range_handle;
    cufftHandle fft_doppler_handle;
    cufftHandle fft_pdop_handle;

    int rank = 1;                   // --- 1D FFTs
    int nn[] = { m };               // --- Size of the Fourier transform
    int istride = n, ostride = n;   // --- Distance between two successive input/output elements
    int idist = 1, odist = 1;       // --- Distance between batches
    int inembed[] = { 0 };          // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };          // --- Output size with pitch (ignored for 1D transforms)
    int batch = n;                  // --- Number of batched executions

    cufftPlanMany(&fft_range_handle, rank, nn, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_Z2Z, batch);
    cufftPlan1d(&fft_doppler_handle, n, CUFFT_Z2Z, m);
    cufftPlan1d(&fft_pdop_handle, n, CUFFT_Z2Z, m/2);

    // Read 1 sector data
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cin >> a >> b;
            iqhh[i*n+j] = make_cuDoubleComplex(a, b);
        }
    }
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cin >> a >> b;
            iqvv[i*n+j] = make_cuDoubleComplex(a, b);
        }
    }

    tock(&tb, &te, "initialization");

    tick(&tb);

    cudaMemcpy(d_iqhh, iqhh, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iqvv, iqvv, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    tock(&tb, &te, "memcpy to device");

    tick(&tb);

    // apply Hamming coefficients
    __apply_hamming<<<m,n>>>(d_iqhh, d_hamming);
    __apply_hamming<<<m,n>>>(d_iqvv, d_hamming);

    tock(&tb, &te, "apply hamming");

    tick(&tb);

    // FFT range profile
    cufftExecZ2Z(fft_range_handle, d_iqhh, d_iqhh, CUFFT_FORWARD);
    cufftExecZ2Z(fft_range_handle, d_iqvv, d_iqvv, CUFFT_FORWARD);

    tock(&tb, &te, "fft range");

    tick(&tb);

    // FFT+shift Doppler profile
    __sumcomplex<<<m,n>>>(d_iqhh, d_sum);
    __avgconj<<<m,n>>>(d_iqhh, d_sum);
    __sumcomplex<<<m,n>>>(d_iqvv, d_sum);
    __avgconj<<<m,n>>>(d_iqvv, d_sum);

    tock(&tb, &te, "sum reduction & average conjugate");

    tick(&tb);

    cufftExecZ2Z(fft_doppler_handle, d_iqhh, d_iqhh, CUFFT_FORWARD);
    cufftExecZ2Z(fft_doppler_handle, d_iqvv, d_iqvv, CUFFT_FORWARD);

    tock(&tb, &te, "fft doppler");

    tick(&tb);

    __conjugate<<<m,n>>>(d_iqhh);
    __conjugate<<<m,n>>>(d_iqvv);

    tock(&tb, &te, "conjugate");

    tick(&tb);

    __shift<<<m,n/2>>>(d_iqhh, n);
    __shift<<<m,n/2>>>(d_iqvv, n);

    tock(&tb, &te, "ffstshift");

    tick(&tb);

    __trim<<<m,2>>>(d_iqhh, n);
    __trim<<<m,2>>>(d_iqvv, n);

    tock(&tb, &te, "clipping");

    tick(&tb);

    // Get absolute value
    __abssqr<<<m/2,n>>>(d_iqhh, n);
    __abssqr<<<m/2,n>>>(d_iqvv, n);

    tock(&tb, &te, "absloute squared");

    tick(&tb);

    // FFT PDOP
    cufftExecZ2Z(fft_pdop_handle, d_iqhh, d_iqhh, CUFFT_FORWARD);
    cufftExecZ2Z(fft_pdop_handle, d_iqvv, d_iqvv, CUFFT_FORWARD);

    tock(&tb, &te, "fft pdop");

    tick(&tb);

    // Apply MA coefficients
    __apply_ma<<<m/2,n>>>(d_iqhh, d_ma);
    __apply_ma<<<m/2,n>>>(d_iqvv, d_ma);

    tock(&tb, &te, "apply ma");

    tick(&tb);

    // Inverse FFT
    cufftExecZ2Z(fft_pdop_handle, d_iqhh, d_iqhh, CUFFT_INVERSE);
    cufftExecZ2Z(fft_pdop_handle, d_iqvv, d_iqvv, CUFFT_INVERSE);

    tock(&tb, &te, "ifft");

    tick(&tb);

    __scale_real<<<m/2,n>>>(d_iqhh);
    __scale_real<<<m/2,n>>>(d_iqvv);

    tock(&tb, &te, "ifft rescale");

    tick(&tb);

    // Sum
    __sum_inplace<<<m/2,n>>>(d_iqhh);
    __sum_inplace<<<m/2,n>>>(d_iqvv);

    tock(&tb, &te, "sum reduction");

    tick(&tb);

    // cudaMemcpy(iqhh, d_iqhh, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // cudaMemcpy(iqvv, d_iqvv, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // for (int i=0; i<m/2; i++) {
    //     double z = pow(i*k_rangeres, 2.0) * k_calib * iqhh[i*n].x;
    //     double zdb = 10 * log10(z);
    //     double zdr = 10 * (log10(iqhh[i*n].x)-log10(iqvv[i*n].x));
    //     cout << zdb << " " << zdr << endl;
    // }
    // exit(0);

    // Calculate ZdB, Zdr
    __calcresult<<<m/2,1>>>(d_iqhh, d_iqvv, d_result, n);

    tock(&tb, &te, "result calc");

    tick(&tb);

    cudaMemcpy(result, d_result, (m/2)*RESULT_SIZE*sizeof(double), cudaMemcpyDeviceToHost);

    tock(&tb, &te, "memcpy to host");

    // for (int i=0; i<m/2; i++) {
    //     for (int j=0; j<RESULT_SIZE; j++) {
    //         cout << result[i*RESULT_SIZE+j] << " ";
    //     }
    //     cout << endl;
    // }

    cudaFree(d_hamming);
    cudaFree(d_ma);
    cudaFree(d_iqhh);
    cudaFree(d_iqvv);

    delete iqhh;
    delete iqvv;

    return 0;
}

    // cudaMemcpy(iqhh, d_iqhh, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // cudaMemcpy(iqvv, d_iqvv, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // for (int i=0; i<m; i++) {
    //     for (int j=0; j<n; j++) {
    //         cout << "(" << iqhh[i*n+j].x << "," << iqhh[i*n+j].y << ") ";
    //     }
    //     cout << endl;
    // }
    // // for (int i=0; i<m; i++) {
    // //     for (int j=0; j<n; j++) {
    // //         cout << iqvv[i*n+j].x << " ";
    // //     }
    // //     cout << endl;
    // // }
    // exit(0);
