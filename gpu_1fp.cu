#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <cufft.h>
#include <sys/time.h>
#include <assert.h>

using namespace std;

#define k_rangeres 30
#define k_calib 1941.05

#define RESULT_SIZE 2

#define DEBUG

inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

float *generate_hamming_coef(int m, int n) {

    // Calculate normalization power on range cell
    float p_range=0;
    for(int i=0; i < m; i++) {
        p_range=p_range+pow(0.53836-0.46164*cos(2*M_PI*(i)/(m-1)), 2.0);
    }
    p_range=p_range/m;

    // Calculate normalization power on Doppler cell
    float p_doppler=0;
    for(int j=0; j < n; j++) {
        p_doppler=p_doppler+pow(0.53836-0.46164*cos(2*M_PI*(j)/(n-1)), 2.0);
    }
    p_doppler=p_doppler/n;

    // Constant since FFT is not normalized and the power is computed w.r.t. 50ohm
    const float K_wind = -1/(16383.5*m*n*sqrt(50));
    const float c = K_wind/sqrt(p_range*p_doppler);

    // Generate elements
    float *_hamming_coef= new float[m*n];
    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            _hamming_coef[i*n+j] = (0.53836-0.46164*cos(2*M_PI*(i)/(m-1))) * (0.53836-0.46164*cos(2*M_PI*(j)/(n-1))) * c;
        }
    }

    return _hamming_coef;
}

float *generate_ma_coef(int n){
    float *_ma_coef = new float[n];
    float _sum = 0.0;
    for(int i=0; i < n; i++) {
        _ma_coef[i]=exp(-(pow(i-((n-1)/2), 2.0))/2);
        _sum += _ma_coef[i];
    }
    for(int i=0; i < n; i++){
        _ma_coef[i] = _ma_coef[i]/_sum;
    }
    return _ma_coef;
}

__global__ void __apply_hamming(cuFloatComplex *a, float *b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // float real, imag;
    // real = a[i].x;
    // imag = a[i].y;
    a[i] = make_cuFloatComplex(b[i]*cuCrealf(a[i]), b[i]*cuCimagf(a[i]));
}

__global__ void __apply_ma(cuFloatComplex *inout, cuFloatComplex *macoef) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

    // float ii, qq, mi, mq;
    // ii = inout[i*n+j].x;
    // qq = inout[i*n+j].y;
    // mi = macoef[j].x;
    // mq = macoef[j].y;

    // inout[i*n+j] = make_cuFloatComplex(ii*mi-qq*mq, ii*mq+qq*mi);
    inout[i*n+j] = cuCmulf(inout[i*n+j], macoef[j]);
}

__global__ void __conjugate(cuFloatComplex *a) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // a[i] = make_cuFloatComplex(a[i].x, a[i].y * -1);
    a[i].y *= -1;
}

__global__ void __shift(cuFloatComplex *inout, int n) {
    unsigned int i = blockIdx.x, j = threadIdx.x;

    cuFloatComplex temp = inout[i*n+j];
    inout[i*n+j] = inout[i*n+(j+n/2)];
    inout[i*n+(j+n/2)] = temp;
}

__global__ void __trim(cuFloatComplex *inout, int n) {
    unsigned int i = blockIdx.x, j = n-threadIdx.x-1;
    inout[i*n+j] = make_cuFloatComplex(0, 0);
}

__global__ void __abssqr(cuFloatComplex *inout, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float real, imag;
    real = cuCrealf(inout[i]);
    imag = cuCimagf(inout[i]);
    inout[i] = make_cuFloatComplex(real*real+imag*imag, 0);    
}

__global__ void __sumcomplex(cuFloatComplex *g_idata, cuFloatComplex *g_odata) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;
    g_odata[i*n+j] = make_cuFloatComplex(g_idata[i*n+j].x, g_idata[i*n+j].y);
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (j < s) {
            // g_odata[i*n+j] = make_cuFloatComplex(g_odata[i*n+j].x+g_odata[i*n+j + s].x, g_odata[i*n+j].y+g_odata[i*n+j + s].y);
            g_odata[i*n+j] = cuCaddf(g_odata[i*n+j], g_odata[i*n+j+s]);
        }
        __syncthreads();
    }
}

__global__ void __sum_inplace(cuFloatComplex *g_idata) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            // g_idata[i] = make_cuFloatComplex(g_idata[i].x+g_idata[i + s].x, 0);
            g_idata[i] = cuCaddf(g_idata[i], g_idata[i+s]);
        }
        __syncthreads();
    }
}

__global__ void __avgconj(cuFloatComplex *in, cuFloatComplex *sum) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

    float avgx = sum[i*n].x/n;
    float avgy = sum[i*n].y/n;
    in[i*n+j] = make_cuFloatComplex(in[i*n+j].x-avgx, (in[i*n+j].y-avgy)*-1);
}

__global__ void __scale_real(cuFloatComplex *inout) {
    unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

    inout[i*n+j] = make_cuFloatComplex(inout[i*n+j].x/n, 0);
}

__global__ void __calcresult(cuFloatComplex *hh, cuFloatComplex *vv, float *out, int n) {
    unsigned int i = blockIdx.x;

    float z = pow(i*k_rangeres, 2.0) * k_calib * hh[i*n].x;
    float zdb = 10 * log10(z);
    float zdr = 10 * (log10(hh[i*n].x)-log10(vv[i*n].x));
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
    ios_base::sync_with_stdio(false);

    struct timeval tb, te;

    tick(&tb);

    cuFloatComplex *iqhh, *iqvv;
    float *result;
    int sector_id;

    const int m = 1024; // cell
    const int n = 512;  // sweep

    const int ma_count = 7;

    iqhh = new cuFloatComplex[m*n];
    iqvv = new cuFloatComplex[m*n];
    result = new float[(m/2)*RESULT_SIZE];

    float a, b;

    // Generate Hamming coefficients
    const float *hamming_coef = generate_hamming_coef(m, n);

    // Generate MA coefficients
    float *ma_coef = generate_ma_coef(ma_count);
    fftwf_complex *_fft_ma = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    fftwf_plan fft_ma_plan = fftwf_plan_dft_1d(n, _fft_ma, _fft_ma, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<ma_count; j++) {
        _fft_ma[j][0] = ma_coef[j];
        _fft_ma[j][1] = 0;
    }
    for (int j=ma_count; j<n; j++) {
        _fft_ma[j][0] = 0;
        _fft_ma[j][1] = 0;
    }
    fftwf_execute(fft_ma_plan);
    fftwf_destroy_plan(fft_ma_plan);
    cuFloatComplex *fft_ma;
    fft_ma = new cuFloatComplex[n];
    for (int j=0; j<n; j++) {
        fft_ma[j] = make_cuFloatComplex(_fft_ma[j][0], _fft_ma[j][1]);
    }
    fftwf_free(_fft_ma);

    // Device buffers
    /*__constant__*/ float *d_hamming;
    /*__constant__*/ cuFloatComplex *d_ma;
    cuFloatComplex *d_iqhh, *d_iqvv;
    cuFloatComplex *d_sum;
    float *d_result;
    //float *d_powhh, *d_powvv;

    cudaMalloc(&d_hamming, m*n*sizeof(float));
    cudaMalloc(&d_ma, n*sizeof(cuFloatComplex));
    cudaMalloc(&d_iqhh, m*n*sizeof(cuFloatComplex));
    cudaMalloc(&d_iqvv, m*n*sizeof(cuFloatComplex));
    cudaMalloc(&d_sum, m*n*sizeof(cuFloatComplex));
    cudaMalloc(&d_result, (m/2)*RESULT_SIZE*sizeof(float));

    cudaMemcpy(d_hamming, hamming_coef, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma, fft_ma, n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

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
                  onembed, ostride, odist, CUFFT_C2C, batch);
    cufftPlan1d(&fft_doppler_handle, n, CUFFT_C2C, m);
    cufftPlan1d(&fft_pdop_handle, n, CUFFT_C2C, m/2);

    tock(&tb, &te, "initialization");

    float ms; // elapsed time in milliseconds

    sector_id = 0;

    // create events and streams
    cudaEvent_t startEvent, stopEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // cudaEventCreate(&dummyEvent);

    cudaEventRecord(startEvent,0);

    while(sector_id < 126) {

        tick(&tb);

        // Read 1 sector data
        cin >> sector_id;
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                cin >> a >> b;
                iqhh[i*n+j] = make_cuFloatComplex(a, b);
            }
        }
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                cin >> a >> b;
                iqvv[i*n+j] = make_cuFloatComplex(a, b);
            }
        }

        tock(&tb, &te, "read");

        cout << "Processing sector " << sector_id << endl;


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
        
        tick(&tb);

        cudaMemcpy(d_iqhh, iqhh, m*n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_iqvv, iqvv, m*n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // tock(&tb, &te, "memcpy to device");

        // tick(&tb);

        // apply Hamming coefficients
        __apply_hamming<<<m,n>>>(d_iqhh, d_hamming);
        __apply_hamming<<<m,n>>>(d_iqvv, d_hamming);

        // tock(&tb, &te, "apply hamming");

        // tick(&tb);

        // FFT range profile
        cufftExecC2C(fft_range_handle, d_iqhh, d_iqhh, CUFFT_FORWARD);
        cufftExecC2C(fft_range_handle, d_iqvv, d_iqvv, CUFFT_FORWARD);

        // tock(&tb, &te, "fft range");

        // tick(&tb);

        // FFT+shift Doppler profile
        __sumcomplex<<<m,n>>>(d_iqhh, d_sum);
        __avgconj<<<m,n>>>(d_iqhh, d_sum);
        __sumcomplex<<<m,n>>>(d_iqvv, d_sum);
        __avgconj<<<m,n>>>(d_iqvv, d_sum);

        // tock(&tb, &te, "sum reduction & average conjugate");

        // tick(&tb);

        cufftExecC2C(fft_doppler_handle, d_iqhh, d_iqhh, CUFFT_FORWARD);
        cufftExecC2C(fft_doppler_handle, d_iqvv, d_iqvv, CUFFT_FORWARD);

        // tock(&tb, &te, "fft doppler");

        // tick(&tb);

        __conjugate<<<m,n>>>(d_iqhh);
        __conjugate<<<m,n>>>(d_iqvv);

        // tock(&tb, &te, "conjugate");

        // tick(&tb);

        __shift<<<m,n/2>>>(d_iqhh, n);
        __shift<<<m,n/2>>>(d_iqvv, n);

        // tock(&tb, &te, "ffstshift");

        // tick(&tb);

        __trim<<<m,2>>>(d_iqhh, n);
        __trim<<<m,2>>>(d_iqvv, n);

        // tock(&tb, &te, "clipping");

        // tick(&tb);

        // Get absolute value
        __abssqr<<<m/2,n>>>(d_iqhh, n);
        __abssqr<<<m/2,n>>>(d_iqvv, n);

        // tock(&tb, &te, "absloute squared");

        // tick(&tb);

        // FFT PDOP
        cufftExecC2C(fft_pdop_handle, d_iqhh, d_iqhh, CUFFT_FORWARD);
        cufftExecC2C(fft_pdop_handle, d_iqvv, d_iqvv, CUFFT_FORWARD);

        // tock(&tb, &te, "fft pdop");

        // tick(&tb);

        // Apply MA coefficients
        __apply_ma<<<m/2,n>>>(d_iqhh, d_ma);
        __apply_ma<<<m/2,n>>>(d_iqvv, d_ma);

        // tock(&tb, &te, "apply ma");

        // tick(&tb);

        // Inverse FFT
        cufftExecC2C(fft_pdop_handle, d_iqhh, d_iqhh, CUFFT_INVERSE);
        cufftExecC2C(fft_pdop_handle, d_iqvv, d_iqvv, CUFFT_INVERSE);

        // tock(&tb, &te, "ifft");

        // tick(&tb);

        __scale_real<<<m/2,n>>>(d_iqhh);
        __scale_real<<<m/2,n>>>(d_iqvv);

        // tock(&tb, &te, "ifft rescale");

        // tick(&tb);

        // Sum
        __sum_inplace<<<m/2,n>>>(d_iqhh);
        __sum_inplace<<<m/2,n>>>(d_iqvv);

        // tock(&tb, &te, "sum reduction");

        // tick(&tb);

        // cudaMemcpy(iqhh, d_iqhh, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        // cudaMemcpy(iqvv, d_iqvv, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        // for (int i=0; i<m/2; i++) {
        //     float z = pow(i*k_rangeres, 2.0) * k_calib * iqhh[i*n].x;
        //     float zdb = 10 * log10(z);
        //     float zdr = 10 * (log10(iqhh[i*n].x)-log10(iqvv[i*n].x));
        //     cout << zdb << " " << zdr << endl;
        // }
        // exit(0);

        // Calculate ZdB, Zdr
        __calcresult<<<m/2,1>>>(d_iqhh, d_iqvv, d_result, n);

        // cudaDeviceSynchronize();
        // tock(&tb, &te, "processing");

        // tick(&tb);

        cudaMemcpy(result, d_result, (m/2)*RESULT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

        tock(&tb, &te, "time");

        for (int i=0; i<m/2; i++) {
            for (int j=0; j<RESULT_SIZE; j++) {
                cout << result[i*RESULT_SIZE+j] << " ";
            }
            cout << endl;
        }
        // exit(0);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time for sequential transfer and execute (ms): %f\n", ms);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(d_hamming);
    cudaFree(d_ma);
    cudaFree(d_iqhh);
    cudaFree(d_iqvv);

    delete[] iqhh;
    delete[] iqvv;

    return 0;
}

    // cudaMemcpy(iqhh, d_iqhh, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    // cudaMemcpy(iqvv, d_iqvv, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

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
