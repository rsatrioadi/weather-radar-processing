#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <cufft.h>

using namespace std;

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

__global__ void __apply_hamming(cuDoubleComplex *a, double *b, int m, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double real, imag;
    real = a[i].x;
    imag = a[i].y;
    a[i] = make_cuDoubleComplex(b[i%(m*n)]*real, b[i%(m*n)]*imag);
}

__global__ void __shift(cuDoubleComplex *out, cuDoubleComplex *in, int n) {
    unsigned int i = blockIdx.x, j = threadIdx.x;

    out[i*n+j] = make_cuDoubleComplex(in[i*n+(n/2-j)].x, in[i*n+(n/2-j)].y);
    out[i*n+j+n/2] = make_cuDoubleComplex(in[i*n+(n-j)].x, in[i*n+(n-j)].y);
}

int main(int argc, char **argv) {

    cuDoubleComplex *iq;
    double *pow_;

    const int m = 1024; // cell
    const int n = 512;  // sweep

    const int ma_count = 7;

    const double k_rangeres = 30;
    const double k_calib = 1941.05;

    iq = new cuDoubleComplex[2*m*n];
    pow_ = new double[m*n];

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
    cuDoubleComplex *d_iq;
    cuDoubleComplex *d_shift;
    //double *d_pow;

    cudaMalloc(&d_hamming, m*n*sizeof(double));
    cudaMemcpy(d_hamming, hamming_coef, m*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_ma, n*sizeof(cuDoubleComplex));
    cudaMemcpy(d_ma, fft_ma, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMalloc(&d_iq, 2*m*n*sizeof(cuDoubleComplex));
    cudaMalloc(&d_shift, 2*m*n*sizeof(cuDoubleComplex));

    // Read 1 sector data
    for (int i=0; i<m*2; i++) {
        for (int j=0; j<n; j++) {
            cin >> a >> b;
            iq[i*n+j] = make_cuDoubleComplex(a, b);
        }
    }

    // apply Hamming coefficients
    cudaMemcpy(d_iq, iq, 2*m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    __apply_hamming<<<2*m,n>>>(d_iq, d_hamming, m, n);
    cudaMemcpy(iq, d_iq, 2*m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // FFT range profile
    fftw_complex *fft_range_buffer;
    fftw_plan fft_range_plan;
    fft_range_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m);
    fft_range_plan = fftw_plan_dft_1d(m, fft_range_buffer, fft_range_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<n; j++) {

        // HH
        for (int i=0; i<m; i++) {
            fft_range_buffer[i][0] = iq[i*n+j].x;
            fft_range_buffer[i][1] = iq[i*n+j].y;
        }
        fftw_execute(fft_range_plan);
        for (int i=0; i<m; i++) {
            iq[i*n+j] = make_cuDoubleComplex(fft_range_buffer[i][0], fft_range_buffer[i][1]);
        }

        // VV
        for (int i=0; i<m; i++) {
            fft_range_buffer[i][0] = iq[(i+m)*n+j].x;
            fft_range_buffer[i][1] = iq[(i+m)*n+j].y;
        }
        fftw_execute(fft_range_plan);
        for (int i=0; i<m; i++) {
            iq[(i+m)*n+j] = make_cuDoubleComplex(fft_range_buffer[i][0], fft_range_buffer[i][1]);
        }
    }
    fftw_destroy_plan(fft_range_plan);
    fftw_free(fft_range_buffer);

    // FFT Doppler profile
    cudaMemcpy(d_iq, iq, 2*m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cufftHandle handle;
    int rank = 1;                     // --- 1D FFTs
    int nn[] = { n };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;     // --- Distance between two successive input/output elements
    int idist = n, odist = n;         // --- Distance between batches
    int inembed[] = { 0 };            // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };            // --- Output size with pitch (ignored for 1D transforms)
    int batch = 2*m;                  // --- Number of batched executions
    cufftPlanMany(&handle, rank, nn, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_Z2Z, batch);
    //cufftPlan1d(&handle, n, CUFFT_Z2Z, 2*m);
    cufftExecZ2Z(handle,  d_iq, d_iq, CUFFT_FORWARD);
    __shift<<<2*m,n/2>>>(d_shift, d_iq, n);
    cudaMemcpy(iq, d_shift, 2*m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i=0; i<2*m; i++) {
        iq[i*n+(n-1)] = make_cuDoubleComplex(0,0);
        iq[i*n+(n-2)] = make_cuDoubleComplex(0,0);
    }

    // for (int i=0; i<m*2; i++) {
    //     for (int j=0; j<n; j++) {
    //         cout << "(" << iq[i*n+j].x << "," << iq[i*n+j].y << ") ";
    //     }
    //     cout << endl;
    // }
    // exit(0);

    cuDoubleComplex *iqhalf;
    iqhalf = new cuDoubleComplex[m*n];
    for (int i=0; i<m/2; i++) {
        for (int j=0; j<n; j++) {
            iqhalf[i*n+j] = make_cuDoubleComplex(iq[i*n+j].x, iq[i*n+j].y);
            iqhalf[(i+m/2)*n+j] = make_cuDoubleComplex(iq[(i+m)*n+j].x, iq[(i+m)*n+j].y);
        }
    }

    // PDOP
    fftw_complex *fft_pdop_buffer;
    fft_pdop_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *fft_mult_buffer;
    fft_mult_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan fft_pdop_plan;
    fft_pdop_plan = fftw_plan_dft_1d(n, fft_pdop_buffer, fft_pdop_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan ifft_conv_plan;
    ifft_conv_plan = fftw_plan_dft_1d(n, fft_mult_buffer, fft_mult_buffer, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (int i=0; i<m; i++) {

        for (int j=0; j<n; j++) {
            fft_pdop_buffer[j][0] = iqhalf[i*n+j].x * iqhalf[i*n+j].x + iqhalf[i*n+j].y * iqhalf[i*n+j].y;
            fft_pdop_buffer[j][1] = 0;
            //cout << fft_pdop_buffer[j][0] << " ";
        }
        //cout << endl;
        fftw_execute(fft_pdop_plan);
        for (int j=0; j<n; j++) {
            fft_mult_buffer[j][0] = fft_pdop_buffer[j][0] * fft_ma[j].x - fft_pdop_buffer[j][1] * fft_ma[j].y;
            fft_mult_buffer[j][1] = fft_pdop_buffer[j][0] * fft_ma[j].y + fft_pdop_buffer[j][1] * fft_ma[j].x;
            //cout << "(" << fft_mult_buffer[j][0] << "," << fft_mult_buffer[j][1] << ") ";
        }
        //cout << endl;
        fftw_execute(ifft_conv_plan);
        for (int j=0; j<n; j++) {
            pow_[i*n+j] = fft_mult_buffer[j][0]/n;
            //cout << pow_[i*n+j] << " ";
        }
        //cout << endl;
    }
    fftw_destroy_plan(ifft_conv_plan);
    fftw_destroy_plan(fft_pdop_plan);

    // Reflectivity
    double *z, *zdb, *zdr;
    z = new double[m];
    zdb = new double[m];
    zdr = new double[m/2];
    for (int i=0; i<m; i++) {
        for (int j=1; j<n; j++) {
            pow_[i*n] += pow_[i*n+j];
        }
        //cout << pow_[i*n] << endl;
    }
    for (int i=0; i<m/2; i++) {
        z[i] = pow(i*k_rangeres, 2.0) * k_calib * pow_[i*n];
        z[i+m/2] = pow(i*k_rangeres, 2.0) * k_calib * pow_[(i+m/2)*n];
        zdb[i] = 10 * log10(z[i]);
        //zdb[i+m/2] = 10 * log10(z[i+m/2]);
        zdr[i] = 10 * (log10(pow_[i*n])-log10(pow_[(i+m/2)*n]));
        cout << zdb[i] << " " << zdr[i] << endl;
    }

    cudaFree(d_hamming);
    //cudaFree(d_ma);
    cudaFree(d_iq);
    cudaFree(d_shift);

    delete iqhalf;

    delete zdr;
    delete zdb;
    delete z;

    fftw_free(fft_mult_buffer);
    fftw_free(fft_pdop_buffer);

    delete pow_;
    delete iq;

    return 0;
}
