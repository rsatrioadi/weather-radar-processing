#include <iostream>
#include <stdlib.h>
#include <cuComplex.h>

using namespace std;

double *generate_hamming_coef(int m, int n) {
    // Determine the Hamming window coeff

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
    double K_wind = -1/(16383.5*m*n*sqrt(50));
    double c = K_wind/sqrt(p_range*p_doppler);

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

    m = 1024;
    n = 512;

    iqhh = new cuDoubleComplex[m*n];
    iqvv = new cuDoubleComplex[m*n];

    double a, b, c, d;

    // read 1 sector data (file is transposed, actual index should be i*n+j)
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
        	cin >> a >> b >> c >> d;
            iqhh[j*n+i] = make_cuDoubleComplex((double)a, (double)b);
            iqvv[j*n+i] = make_cuDoubleComplex((double)c, (double)d);
        }
    }

    // generate Hamming coefficients
    double *hamming_coef = generate_hamming_coef(m,n);

    // apply Hamming coefficients
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            double real = iqhh[i*n+j].x;
            double imag = iqhh[i*n+j].y;
            iqhh[i*n+j] = make_cuDoubleComplex(real*hamming_coef[i*n+j], imag*hamming_coef[i*n+j]);
        }
    }
}