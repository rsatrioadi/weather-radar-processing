#include <iostream>
#include <stdlib.h>
#include <fftw3.h>
#include <sys/time.h>    
#include <math.h>
// #include <fstream>

#include "udpbroadcast.h"
#include "sector.h"
#include "floats.h"

using namespace std;
using namespace udpbroadcast;

#define NUM_BYTES_PER_SAMPLE (3*2*2)

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

int main(int argc, char **argv) {
    ios_base::sync_with_stdio(false);

    struct timeval tb, te;
    unsigned long long bb, e;

    float *z, *zdb, *zdr;

    gettimeofday(&tb, NULL);

    //cuDoubleComplex *iqhh, *iqvv;
    fftwf_complex *iqhh, *iqvv, *iqvh;
    float *powhh, *powvv, *powhv;
    int sector_id;

    const int m = 1024; // NUM_SWEEPS
    const int n = 512;  // NUM_SAMPLES

    const int ma_count = 7;

    const float k_rangeres = 30;
    const float k_calib = 1941.05;

    //iqhh = new cuDoubleComplex[m*n];
    //iqvv = new cuDoubleComplex[m*n];
    iqhh = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m*n);
    iqvv = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m*n);
    iqvh = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m*n);

    powhh = new float[(m/2)*n];
    powvv = new float[(m/2)*n];
    powhv = new float[(m/2)*n];

    //float a, b;

    // Generate Hamming coefficients
    const float *hamming_coef = generate_hamming_coef(m, n);

    // Generate MA coefficients
    float *ma_coef = generate_ma_coef(ma_count);
    fftwf_complex *fft_ma = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    fftwf_plan fft_ma_plan = fftwf_plan_dft_1d(n, fft_ma, fft_ma, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int j=0; j<ma_count; j++) {
        fft_ma[j][0] = ma_coef[j];
        fft_ma[j][1] = 0;
    }
    for (int j=ma_count; j<n; j++) {
        fft_ma[j][0] = 0;
        fft_ma[j][1] = 0;
    }
    fftwf_execute(fft_ma_plan);
    fftwf_destroy_plan(fft_ma_plan);
    /*for (int j=7; j<n; j++) {
        cout << "(" << fft_ma[j][0] << "," << fft_ma[j][1] << ") ";
    }
    cout << endl;
    exit(0);*/

    gettimeofday(&te, NULL);
    bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
    e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

    // cout << "initialization: " << e-bb << endl;

    udpserver server(19001); // receive raw data
    udpclient zdbClient(19002); // send zdb result
    udpclient zdrClient(19003); // send zdr result

    // ofstream myFile;
    // myFile.open("out/cpu.bin", ios::out | ios::binary);
    sector_id = -1;

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    while(sector_id < 142) {

        // gettimeofday(&tb, NULL);

        // Read 1 sector data
        // cin >> sector_id;
        sector_id++;

        cout << "receiving sector " << sector_id << endl;
        char *buff = new char[NUM_BYTES_PER_SAMPLE*n*m];
        for (int j=0; j<m; j++) {
            server.recv(&buff[j*(NUM_BYTES_PER_SAMPLE*n)],NUM_BYTES_PER_SAMPLE*n);
        }
        //cout << "done!" << endl;
        Sector s(m,n);
        s.fromByteArray(buff);
        // s.print();
        // exit(0);
        delete [] buff;

        int a, b;

        int idx=0;
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
            	// cin >> a >> b;
                //iqhh[i*n+j] = make_cuDoubleComplex(a, b);
                a = idx++;
                b = idx++;
                iqhh[i*n+j][0] = s.hh[a];
                iqhh[i*n+j][1] = s.hh[b];
                iqvv[i*n+j][0] = s.vv[a];
                iqvv[i*n+j][1] = s.vv[b];
                iqvh[i*n+j][0] = s.vh[a];
                iqvh[i*n+j][1] = s.vh[b];
            }
        }

        // cout << "hh:" << endl;
        // for (int i=0; i<m; i++) {
        //     for (int j=0; j<n; j++) {
        //         cout << iqhh[i*n+j][0] << " " << iqhh[i*n+j][1] << " ";
        //     }
        // }
        // cout << endl;

        // cout << "vv:" << endl;
        // for (int i=0; i<m; i++) {
        //     for (int j=0; j<n; j++) {
        //         cout << iqvv[i*n+j][0] << " " << iqvv[i*n+j][1] << " ";
        //     }
        // }
        // cout << endl;

        // cout << "vh:" << endl;
        // for (int i=0; i<m; i++) {
        //     for (int j=0; j<n; j++) {
        //         cout << iqvh[i*n+j][0] << " " << iqvh[i*n+j][1] << " ";
        //     }
        // }
        // cout << endl;
        // exit(0);
        // gettimeofday(&te, NULL);
        // bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
        // e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

        // cout << "read: " << e-bb << endl;

        // cout << "Processing sector " << sector_id << endl;

        // for (int i=0; i<m; i++) {
        //     for (int j=0; j<n; j++) {
        //         cout << "(" << iqhh[i*n+j][0] << "," << iqhh[i*n+j][1] << ") ";
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
        
        // gettimeofday(&tb, NULL);

        // apply Hamming coefficients
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                //float real = iqhh[i*n+j].x;
                //float imag = iqhh[i*n+j].y;
                //iqhh[i*n+j] = make_cuDoubleComplex(real*hamming_coef[i*n+j], imag*hamming_coef[i*n+j]);

                // HH
                iqhh[i*n+j][0] *= hamming_coef[i*n+j];
                iqhh[i*n+j][1] *= hamming_coef[i*n+j];

                // VV
                iqvv[i*n+j][0] *= hamming_coef[i*n+j];
                iqvv[i*n+j][1] *= hamming_coef[i*n+j];

                // HV
                iqvh[i*n+j][0] *= hamming_coef[i*n+j];
                iqvh[i*n+j][1] *= hamming_coef[i*n+j];
            }
        }

        // FFT range profile
        fftwf_complex *fft_range_buffer;
        fftwf_plan fft_range_plan;
        fft_range_buffer = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * m);
        fft_range_plan = fftwf_plan_dft_1d(m, fft_range_buffer, fft_range_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
        for (int j=0; j<n; j++) {

            // HH
            for (int i=0; i<m; i++) {
                //fft_range_buffer[i][0] = iqhh[i*n+j].x;
                //fft_range_buffer[i][1] = iqhh[i*n+j].y;
                fft_range_buffer[i][0] = iqhh[i*n+j][0];
                fft_range_buffer[i][1] = iqhh[i*n+j][1];
            }
            fftwf_execute(fft_range_plan);
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
            fftwf_execute(fft_range_plan);
            for (int i=0; i<m; i++) {
                iqvv[i*n+j][0] = fft_range_buffer[i][0];
                iqvv[i*n+j][1] = fft_range_buffer[i][1];
            }

            // VV
            for (int i=0; i<m; i++) {
                fft_range_buffer[i][0] = iqvh[i*n+j][0];
                fft_range_buffer[i][1] = iqvh[i*n+j][1];
            }
            fftwf_execute(fft_range_plan);
            for (int i=0; i<m; i++) {
                iqvh[i*n+j][0] = fft_range_buffer[i][0];
                iqvh[i*n+j][1] = fft_range_buffer[i][1];
            }
        }
        fftwf_destroy_plan(fft_range_plan);
        fftwf_free(fft_range_buffer);

        // FFT Doppler profile
        fftwf_complex *fft_doppler_buffer;
        fftwf_plan fft_doppler_plan;
        fft_doppler_buffer = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
        fft_doppler_plan = fftwf_plan_dft_1d(n, fft_doppler_buffer, fft_doppler_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
        for (int i=0; i<m; i++) {

            // HH
            float avgi = 0, avgq = 0;
            for (int j=0; j<n; j++) {
                avgi += iqhh[i*n+j][0];
                avgq += iqhh[i*n+j][1];
            }
            avgi /= n;
            avgq /= n;
            for (int j=0; j<n; j++) {
                //fft_doppler_buffer[j][0] = iqhh[i*n+j].x;
                //fft_doppler_buffer[j][1] = iqhh[i*n+j].y;
                fft_doppler_buffer[j][0] = (iqhh[i*n+j][0] - avgi);
                fft_doppler_buffer[j][1] = (iqhh[i*n+j][1] - avgq) * -1;
            }
            fftwf_execute(fft_doppler_plan);

            // for (int j=0; j<n; j++) {
            //     cout << "(" << fft_doppler_buffer[j][0] << "," << fft_doppler_buffer[j][1] << ") ";
            // }
            // cout << endl;
            for (int j=0; j<n/2; j++) {
                //iqhh[i*n+j] = make_cuDoubleComplex(fft_doppler_buffer[j][0], fft_doppler_buffer[j][1]);
                iqhh[i*n+j][0] = fft_doppler_buffer[j+n/2][0];
                iqhh[i*n+j][1] = fft_doppler_buffer[j+n/2][1] * -1;
                iqhh[i*n+j+n/2][0] = fft_doppler_buffer[j][0];
                iqhh[i*n+j+n/2][1] = fft_doppler_buffer[j][1] * -1;
            }
            //iqhh[i*n+(n-1)] = make_cuDoubleComplex(0.0,0.0);
            //iqhh[i*n+(n-2)] = make_cuDoubleComplex(0.0,0.0);
            iqhh[i*n+(n-1)][0] = 0;
            iqhh[i*n+(n-1)][1] = 0;
            iqhh[i*n+(n-2)][0] = 0;
            iqhh[i*n+(n-2)][1] = 0;

            // VV
            avgi = 0; avgq = 0;
            for (int j=0; j<n; j++) {
                avgi += iqvv[i*n+j][0];
                avgq += iqvv[i*n+j][1];
            }
            avgi /= n;
            avgq /= n;
            for (int j=0; j<n; j++) {
                fft_doppler_buffer[j][0] = (iqvv[i*n+j][0] - avgi);
                fft_doppler_buffer[j][1] = (iqvv[i*n+j][1] - avgq) * -1;
            }
            fftwf_execute(fft_doppler_plan);

            // for (int j=0; j<n; j++) {
            //     cout << "(" << fft_doppler_buffer[j][0] << "," << fft_doppler_buffer[j][1] << ") ";
            // }
            // cout << endl;
            for (int j=0; j<n/2; j++) {
                iqvv[i*n+j][0] = fft_doppler_buffer[j+n/2][0];
                iqvv[i*n+j][1] = fft_doppler_buffer[j+n/2][1] * -1;
                iqvv[i*n+j+n/2][0] = fft_doppler_buffer[j][0];
                iqvv[i*n+j+n/2][1] = fft_doppler_buffer[j][1] * -1;
            }
            iqvv[i*n+(n-1)][0] = 0;
            iqvv[i*n+(n-1)][1] = 0;
            iqvv[i*n+(n-2)][0] = 0;
            iqvv[i*n+(n-2)][1] = 0;

            // HV
            avgi = 0; avgq = 0;
            for (int j=0; j<n; j++) {
                avgi += iqvh[i*n+j][0];
                avgq += iqvh[i*n+j][1];
            }
            avgi /= n;
            avgq /= n;
            for (int j=0; j<n; j++) {
                fft_doppler_buffer[j][0] = (iqvh[i*n+j][0] - avgi);
                fft_doppler_buffer[j][1] = (iqvh[i*n+j][1] - avgq) * -1;
            }
            fftwf_execute(fft_doppler_plan);

            // for (int j=0; j<n; j++) {
            //     cout << "(" << fft_doppler_buffer[j][0] << "," << fft_doppler_buffer[j][1] << ") ";
            // }
            // cout << endl;
            for (int j=0; j<n/2; j++) {
                iqvh[i*n+j][0] = fft_doppler_buffer[j+n/2][0];
                iqvh[i*n+j][1] = fft_doppler_buffer[j+n/2][1] * -1;
                iqvh[i*n+j+n/2][0] = fft_doppler_buffer[j][0];
                iqvh[i*n+j+n/2][1] = fft_doppler_buffer[j][1] * -1;
            }
            iqvh[i*n+(n-1)][0] = 0;
            iqvh[i*n+(n-1)][1] = 0;
            iqvh[i*n+(n-2)][0] = 0;
            iqvh[i*n+(n-2)][1] = 0;
        }
        fftwf_destroy_plan(fft_doppler_plan);
        fftwf_free(fft_doppler_buffer);

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
        fftwf_complex *fft_pdop_buffer;
        fft_pdop_buffer = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
        fftwf_complex *fft_mult_buffer;
        fft_mult_buffer = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
        fftwf_plan fft_pdop_plan;
        fft_pdop_plan = fftwf_plan_dft_1d(n, fft_pdop_buffer, fft_pdop_buffer, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_plan ifft_conv_plan;
        ifft_conv_plan = fftwf_plan_dft_1d(n, fft_mult_buffer, fft_mult_buffer, FFTW_BACKWARD, FFTW_ESTIMATE);
        for (int i=0; i<m/2; i++) {

            // HH
            for (int j=0; j<n; j++) {
                fft_pdop_buffer[j][0] = iqhh[i*n+j][0] * iqhh[i*n+j][0] + iqhh[i*n+j][1] * iqhh[i*n+j][1];
                fft_pdop_buffer[j][1] = 0;
                // cout << fft_pdop_buffer[j][0] << " ";
            }
            // cout << endl;
            fftwf_execute(fft_pdop_plan);
            for (int j=0; j<n; j++) {
                fft_mult_buffer[j][0] = fft_pdop_buffer[j][0] * fft_ma[j][0] - fft_pdop_buffer[j][1] * fft_ma[j][1];
                fft_mult_buffer[j][1] = fft_pdop_buffer[j][0] * fft_ma[j][1] + fft_pdop_buffer[j][1] * fft_ma[j][0];
                // cout << "(" << fft_mult_buffer[j][0] << "," << fft_mult_buffer[j][1] << ") ";
            }
            // cout << endl;
            fftwf_execute(ifft_conv_plan);
            for (int j=0; j<n; j++) {
                powhh[i*n+j] = fft_mult_buffer[j][0]/n;
                // cout << powhh[i*n+j] << " ";
            }
            // cout << endl;
        }
        for (int i=0; i<m/2; i++) {
            // VV
            for (int j=0; j<n; j++) {
                fft_pdop_buffer[j][0] = iqvv[i*n+j][0] * iqvv[i*n+j][0] + iqvv[i*n+j][1] * iqvv[i*n+j][1];
                fft_pdop_buffer[j][1] = 0;
                // cout << fft_pdop_buffer[j][0] << " ";
            }
            // cout << endl;
            fftwf_execute(fft_pdop_plan);
            for (int j=0; j<n; j++) {
                fft_mult_buffer[j][0] = fft_pdop_buffer[j][0] * fft_ma[j][0] - fft_pdop_buffer[j][1] * fft_ma[j][1];
                fft_mult_buffer[j][1] = fft_pdop_buffer[j][0] * fft_ma[j][1] + fft_pdop_buffer[j][1] * fft_ma[j][0];
                // cout << "(" << fft_pdop_buffer[j][0] << "," << fft_pdop_buffer[j][1] << ") ";
            }
            // cout << endl;
            fftwf_execute(ifft_conv_plan);
            for (int j=0; j<n; j++) {
                powvv[i*n+j] = fft_mult_buffer[j][0]/n;
                //cout << powvv[i*n+j] << " ";
            }
            //cout << endl;
        }
        for (int i=0; i<m/2; i++) {
            // HV
            for (int j=0; j<n; j++) {
                fft_pdop_buffer[j][0] = iqvh[i*n+j][0] * iqvh[i*n+j][0] + iqvh[i*n+j][1] * iqvh[i*n+j][1];
                fft_pdop_buffer[j][1] = 0;
                // cout << fft_pdop_buffer[j][0] << " ";
            }
            // cout << endl;
            fftwf_execute(fft_pdop_plan);
            for (int j=0; j<n; j++) {
                fft_mult_buffer[j][0] = fft_pdop_buffer[j][0] * fft_ma[j][0] - fft_pdop_buffer[j][1] * fft_ma[j][1];
                fft_mult_buffer[j][1] = fft_pdop_buffer[j][0] * fft_ma[j][1] + fft_pdop_buffer[j][1] * fft_ma[j][0];
                // cout << "(" << fft_pdop_buffer[j][0] << "," << fft_pdop_buffer[j][1] << ") ";
            }
            // cout << endl;
            fftwf_execute(ifft_conv_plan);
            for (int j=0; j<n; j++) {
                powhv[i*n+j] = fft_mult_buffer[j][0]/n;
                //cout << powhv[i*n+j] << " ";
            }
            //cout << endl;
        }
        fftwf_destroy_plan(ifft_conv_plan);
        fftwf_destroy_plan(fft_pdop_plan);
        //fftwf_free(fft_pdop_buffer);

        // Reflectivity
        z = new float[m/2];
        zdb = new float[m/2];
        zdr = new float[m/2];
        for (int i=0; i<m/2; i++) {
            for (int j=1; j<n; j++) {
                powhh[i*n] += powhh[i*n+j];
                powvv[i*n] += powvv[i*n+j];
                powhv[i*n] += powhv[i*n+j];
            }
            //cout << powhh[i*n] << endl;
            z[i] = pow(i*k_rangeres, 2.0) * k_calib * powhh[i*n];
            zdb[i] = 10 * log10(z[i]);
            zdr[i] = 10 * (log10(powhh[i*n])-log10(powvv[i*n]));
            // myFile.write((char*)&zdb[i], sizeof(float));
            // myFile.write((char*)&zdr[i], sizeof(float));
            // cout << zdb[i] << " " << zdr[i] << endl;
        }

        cout << "zdb: ";
        for (int i=0; i<m/2; i++) {
            cout << zdb[i] << " ";
        }
        cout << endl;

        unsigned char* zdbBuff = new unsigned char[sizeof(float)*(m/2)+2];
        unsigned char* zdrBuff = new unsigned char[sizeof(float)*(m/2)+2];
        zdbBuff[0] = (sector_id>>8)&0xff;
        zdbBuff[1] = (sector_id)&0xff;
        zdrBuff[0] = (sector_id>>8)&0xff;
        zdrBuff[1] = (sector_id)&0xff;
        aftoab(zdb,(m/2),&zdbBuff[2]);
        aftoab(zdr,(m/2),&zdrBuff[2]);

        zdbClient.send((const char*)zdbBuff, (m/2)*sizeof(float)+2);
        zdrClient.send((const char*)zdrBuff, (m/2)*sizeof(float)+2);


        fftwf_free(fft_mult_buffer);
        fftwf_free(fft_pdop_buffer);

        // gettimeofday(&te, NULL);
        // bb = (unsigned long long)(tb.tv_sec) * 1000000 + (unsigned long long)(tb.tv_usec) / 1;
        // e = (unsigned long long)(te.tv_sec) * 1000000 + (unsigned long long)(te.tv_usec) / 1;

        // cout << "time: " << e-bb << endl;
    }
    // myFile.close();

    gettimeofday(&t2, NULL);
    bb = (unsigned long long)(t1.tv_sec) * 1000000 + (unsigned long long)(t1.tv_usec) / 1;
    e = (unsigned long long)(t2.tv_sec) * 1000000 + (unsigned long long)(t2.tv_usec) / 1;

    cout << "All (us): " << e-bb << endl;

    delete zdr;
    delete zdb;
    delete z;

    fftwf_free(fft_ma);

    delete powhh;
    delete powvv;
    delete powhv;

    //delete iqvv;
    //delete iqhh;
    fftwf_free(iqvh);
    fftwf_free(iqvv);
    fftwf_free(iqhh);

    return 0;
}
