#include <iostream>
#include <stdlib.h>
#include <cuComplex.h>
#include <fftw3.h>

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
    double *ham_coef= new double[m*n];
    for(int i=0; i < m; i++) {
        for(int j=0; j < n; j++) {
            ham_coef[i*n+j] = (0.53836-0.46164*cos(2*M_PI*(i)/(m-1))) * (0.53836-0.46164*cos(2*M_PI*(j)/(n-1))) * c;
        }
    }

    return ham_coef;
}

int main(int argc, char **argv) {

    int m, n;
    cuDoubleComplex *iqhh, *iqvv;

    m = 1024; // cell
    n = 512;  // sweep

    iqhh = new cuDoubleComplex[m*n];
    iqvv = new cuDoubleComplex[m*n];

    double a, b, c, d;

    // Read 1 sector data (file is transposed, actual index should be i*n+j)
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
        	cin >> a >> b >> c >> d;
            iqhh[j*n+i] = make_cuDoubleComplex(a, b);
            iqvv[j*n+i] = make_cuDoubleComplex(c, d);
        }
    }

    // Generate Hamming coefficients
    double *hamming_coef = generate_hamming_coef(m,n);

    // apply Hamming coefficients
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            double real = iqhh[i*n+j].x;
            double imag = iqhh[i*n+j].y;
            iqhh[i*n+j] = make_cuDoubleComplex(real*hamming_coef[i*n+j], imag*hamming_coef[i*n+j]);
        }
    }

    // FFT range profile
    fftw_complex *fft_range_buffer;
    fftw_plan fft_range_plan;
    fft_range_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m);
    fft_range_plan = fftw_plan_dft_1d(m, fft_range_buffer, fft_range_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<n; j++) {
        for (int i=0; i<m; i++) {
            fft_range_buffer[i][0] = iqhh[i*n+j].x;
            fft_range_buffer[i][1] = iqhh[i*n+j].y;
        }
        fftw_execute(fft_range_plan);
        for (int i=0; i<m; i++) {
            iqhh[i*n+j] = make_cuDoubleComplex(fft_range_buffer[i][0], fft_range_buffer[i][1]);
        }
    }
    fftw_destroy_plan(fft_range_plan);
    fftw_free(fft_range_buffer);

    // FFT Doppler profile
    fftw_complex *fft_doppler_buffer;
    fftw_plan fft_doppler_plan;
    fft_doppler_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fft_doppler_plan = fftw_plan_dft_1d(n, fft_doppler_buffer, fft_doppler_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            fft_doppler_buffer[j][0] = iqhh[i*n+j].x;
            fft_doppler_buffer[j][1] = iqhh[i*n+j].y;
        }
        fftw_execute(fft_doppler_plan);
        for (int j=0; j<n; j++) {
            iqhh[i*n+j] = make_cuDoubleComplex(fft_doppler_buffer[j][0], fft_doppler_buffer[j][1]);
        }
        iqhh[i*n+(-1)] = make_cuDoubleComplex(0.0,0.0);
        iqhh[i*n+(-2)] = make_cuDoubleComplex(0.0,0.0);
    }
    fftw_destroy_plan(fft_doppler_plan);
    fftw_free(fft_doppler_buffer);

    double *pdophh = new double[n];
    for (int i=0; i<m/2; i++) {
        for (int j=0; j<n; j++) {
            pdophh[j] = iqhh[i*n+j].x * iqhh[i*n+j].x + iqhh[i*n+j].y * iqhh[i*n+j].y;
            cout << pdophh[j] << " ";
        }
        cout << endl;
    }
    delete pdophh;



/*
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << "(" << iqhh[i*n+j].x << "," << iqhh[i*n+j].y << ") ";
        }
        cout << endl;
    }
*/
}
