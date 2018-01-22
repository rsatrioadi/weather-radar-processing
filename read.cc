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

int main(int argc, char **argv) {

    //cuDoubleComplex *iqhh, *iqvv;
    fftw_complex *iqhh, *iqvv;
    double *powhh, *powvv;

    const int m = 1024; // cell
    const int n = 512;  // sweep

    const int ma_count = 7;

    const double k_rangeres = 30;
    const double k_calib = 1941.05;

    //iqhh = new cuDoubleComplex[m*n];
    //iqvv = new cuDoubleComplex[m*n];
    iqhh = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m*n);
    iqvv = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m*n);

    powhh = new double[(m/2)*n];
    powvv = new double[(m/2)*n];

    double a, b, c, d;

    // Generate Hamming coefficients
    const double *hamming_coef = generate_hamming_coef(m, n);

    // Generate MA coefficients
    double *ma_coef = generate_ma_coef(ma_count);
    fftw_complex *fft_ma = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan fft_ma_plan = fftw_plan_dft_1d(n, fft_ma, fft_ma, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<ma_count; j++) {
        fft_ma[j][0] = ma_coef[j];
        fft_ma[j][1] = 0;
    }
    for (int j=ma_count; j<n; j++) {
        fft_ma[j][0] = 0;
        fft_ma[j][1] = 0;
    }
    fftw_execute(fft_ma_plan);
    fftw_destroy_plan(fft_ma_plan);
    /*for (int j=7; j<n; j++) {
        cout << "(" << fft_ma[j][0] << "," << fft_ma[j][1] << ") ";
    }
    cout << endl;
    exit(0);*/

    // Read 1 sector data (hh)
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
        	cin >> a >> b;
            //iqhh[i*n+j] = make_cuDoubleComplex(a, b);
            //iqvv[i*n+j] = make_cuDoubleComplex(c, d);
            iqhh[i*n+j][0] = a;
            iqhh[i*n+j][1] = b;
        }
    }

    // Read 1 sector data (vv)
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cin >> a >> b;
            iqvv[i*n+j][0] = a;
            iqvv[i*n+j][1] = b;
        }
    }

    // apply Hamming coefficients
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            //double real = iqhh[i*n+j].x;
            //double imag = iqhh[i*n+j].y;
            //iqhh[i*n+j] = make_cuDoubleComplex(real*hamming_coef[i*n+j], imag*hamming_coef[i*n+j]);

            // HH
            iqhh[i*n+j][0] *= hamming_coef[i*n+j];
            iqhh[i*n+j][1] *= hamming_coef[i*n+j];

            // VV
            iqvv[i*n+j][0] *= hamming_coef[i*n+j];
            iqvv[i*n+j][1] *= hamming_coef[i*n+j];
        }
    }

    // FFT range profile
    fftw_complex *fft_range_buffer;
    fftw_plan fft_range_plan;
    fft_range_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m);
    fft_range_plan = fftw_plan_dft_1d(m, fft_range_buffer, fft_range_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<n; j++) {

        // HH
        for (int i=0; i<m; i++) {
            //fft_range_buffer[i][0] = iqhh[i*n+j].x;
            //fft_range_buffer[i][1] = iqhh[i*n+j].y;
            fft_range_buffer[i][0] = iqhh[i*n+j][0];
            fft_range_buffer[i][1] = iqhh[i*n+j][1];
        }
        fftw_execute(fft_range_plan);
        for (int i=0; i<m; i++) {
            //iqhh[i*n+j] = make_cuDoubleComplex(fft_range_buffer[i][0], fft_range_buffer[i][1]);
            iqhh[i*n+j][0] = fft_range_buffer[i][0];
            iqhh[i*n+j][1] = fft_range_buffer[i][1];
        }

        // VV
        for (int i=0; i<m; i++) {
            fft_range_buffer[i][0] = iqvv[i*n+j][0];
            fft_range_buffer[i][1] = iqvv[i*n+j][1];
        }
        fftw_execute(fft_range_plan);
        for (int i=0; i<m; i++) {
            iqvv[i*n+j][0] = fft_range_buffer[i][0];
            iqvv[i*n+j][1] = fft_range_buffer[i][1];
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

        // HH
        for (int j=0; j<n; j++) {
            //fft_doppler_buffer[j][0] = iqhh[i*n+j].x;
            //fft_doppler_buffer[j][1] = iqhh[i*n+j].y;
            fft_doppler_buffer[j][0] = iqhh[i*n+j][0];
            fft_doppler_buffer[j][1] = iqhh[i*n+j][1];
        }
        fftw_execute(fft_doppler_plan);
        for (int j=0; j<n/2; j++) {
            //iqhh[i*n+j] = make_cuDoubleComplex(fft_doppler_buffer[j][0], fft_doppler_buffer[j][1]);
            iqhh[i*n+j][0] = fft_doppler_buffer[n/2-j][0];
            iqhh[i*n+j][1] = fft_doppler_buffer[n/2-j][1];
            iqhh[i*n+j+n/2][0] = fft_doppler_buffer[n-j][0];
            iqhh[i*n+j+n/2][1] = fft_doppler_buffer[n-j][1];
        }
        //iqhh[i*n+(n-1)] = make_cuDoubleComplex(0.0,0.0);
        //iqhh[i*n+(n-2)] = make_cuDoubleComplex(0.0,0.0);
        iqhh[i*n+(n-1)][0] = 0;
        iqhh[i*n+(n-1)][1] = 0;
        iqhh[i*n+(n-2)][0] = 0;
        iqhh[i*n+(n-2)][1] = 0;

        // VV
        for (int j=0; j<n; j++) {
            fft_doppler_buffer[j][0] = iqvv[i*n+j][0];
            fft_doppler_buffer[j][1] = iqvv[i*n+j][1];
        }
        fftw_execute(fft_doppler_plan);
        for (int j=0; j<n/2; j++) {
            iqvv[i*n+j][0] = fft_doppler_buffer[n/2-j][0];
            iqvv[i*n+j][1] = fft_doppler_buffer[n/2-j][1];
            iqvv[i*n+j+n/2][0] = fft_doppler_buffer[n-j][0];
            iqvv[i*n+j+n/2][1] = fft_doppler_buffer[n-j][1];
        }
        iqvv[i*n+(n-1)][0] = 0;
        iqvv[i*n+(n-1)][1] = 0;
        iqvv[i*n+(n-2)][0] = 0;
        iqvv[i*n+(n-2)][1] = 0;
    }
    fftw_destroy_plan(fft_doppler_plan);
    fftw_free(fft_doppler_buffer);

    // for (int i=0; i<m; i++) {
    //     for (int j=0; j<n; j++) {
    //         cout << "(" << iqhh[i*n+j][0] << "," << iqhh[i*n+j][1] << ") ";
    //     }
    //     cout << endl;
    // }
    // for (int i=0; i<m; i++) {
    //     for (int j=0; j<n; j++) {
    //         cout << "(" << iqvv[i*n+j][0] << "," << iqvv[i*n+j][1] << ") ";
    //     }
    //     cout << endl;
    // }
    // exit(0);

    // PDOP
    fftw_complex *fft_pdop_buffer;
    fft_pdop_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *fft_mult_buffer;
    fft_mult_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan fft_pdop_plan;
    fft_pdop_plan = fftw_plan_dft_1d(n, fft_pdop_buffer, fft_pdop_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan ifft_conv_plan;
    ifft_conv_plan = fftw_plan_dft_1d(n, fft_mult_buffer, fft_mult_buffer, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (int i=0; i<m/2; i++) {

        // HH
        for (int j=0; j<n; j++) {
            fft_pdop_buffer[j][0] = iqhh[i*n+j][0] * iqhh[i*n+j][0] + iqhh[i*n+j][1] * iqhh[i*n+j][1];
            fft_pdop_buffer[j][1] = 0;
            //cout << fft_pdop_buffer[j][0] << " ";
        }
        //cout << endl;
        fftw_execute(fft_pdop_plan);
        for (int j=0; j<n; j++) {
            fft_mult_buffer[j][0] = fft_pdop_buffer[j][0] * fft_ma[j][0] - fft_pdop_buffer[j][1] * fft_ma[j][1];
            fft_mult_buffer[j][1] = fft_pdop_buffer[j][0] * fft_ma[j][1] + fft_pdop_buffer[j][1] * fft_ma[j][0];
            //cout << "(" << fft_mult_buffer[j][0] << "," << fft_mult_buffer[j][1] << ") ";
        }
        //cout << endl;
        fftw_execute(ifft_conv_plan);
        for (int j=0; j<n; j++) {
            powhh[i*n+j] = fft_mult_buffer[j][0]/n;
            //cout << powhh[i*n+j] << " ";
        }
        //cout << endl;

        // VV
        for (int j=0; j<n; j++) {
            fft_pdop_buffer[j][0] = iqvv[i*n+j][0] * iqvv[i*n+j][0] + iqvv[i*n+j][1] * iqvv[i*n+j][1];
            fft_pdop_buffer[j][1] = 0;
        }
        fftw_execute(fft_pdop_plan);
        for (int j=0; j<n; j++) {
            fft_mult_buffer[j][0] = fft_pdop_buffer[j][0] * fft_ma[j][0] - fft_pdop_buffer[j][1] * fft_ma[j][1];
            fft_mult_buffer[j][1] = fft_pdop_buffer[j][0] * fft_ma[j][1] + fft_pdop_buffer[j][1] * fft_ma[j][0];
        }
        fftw_execute(ifft_conv_plan);
        for (int j=0; j<n; j++) {
            powvv[i*n+j] = fft_mult_buffer[j][0]/n;
            //cout << powvv[i*n+j] << " ";
        }
        //cout << endl;
    }
    fftw_destroy_plan(ifft_conv_plan);
    fftw_destroy_plan(fft_pdop_plan);
    //fftw_free(fft_pdop_buffer);

    // Reflectivity
    double *z, *zdb, *zdr;
    z = new double[m/2];
    zdb = new double[m/2];
    zdr = new double[m/2];
    for (int i=0; i<m/2; i++) {
        for (int j=1; j<n; j++) {
            powhh[i*n] += powhh[i*n+j];
            powvv[i*n] += powvv[i*n+j];
        }
        //cout << powhh[i*n] << endl;
        z[i] = pow(i*k_rangeres, 2.0) * k_calib * powhh[i*n];
        zdb[i] = 10 * log10(z[i]);
        zdr[i] = 10 * (log10(powhh[i*n])-log10(powvv[i*n]));
        cout << zdb[i] << " " << zdr[i] << endl;
    }


    delete zdr;
    delete zdb;
    delete z;

    fftw_free(fft_mult_buffer);
    fftw_free(fft_pdop_buffer);
    fftw_free(fft_ma);

    delete powhh;

    //delete iqvv;
    //delete iqhh;
    fftw_free(iqvv);
    fftw_free(iqhh);

    return 0;
}
