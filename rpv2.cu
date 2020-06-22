#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <cufft.h>
#include <sys/time.h>
#include <assert.h>
#include "sector.h"
#include "udpbroadcast.h"
#include "floats.h"

using namespace std;
using namespace udpbroadcast;

const int 
  n_sectors, 
  n_sweeps, 
  n_samples, 
  n_elevations;
static const int k_range_resolution = 30;
static constexpr float k_calibration = 1941.05;
static const int ma_count = 7;
const int n_cuda_streams;
int
  current_sector = 0,
  current_sweep = 0,
  current_sample = 0,
  current_elevation = 0,
  current_stream = 0;
const int o_types = 2;

int
  hh_index_start,
  vv_index_start,
  vh_index_start,
  input_stream_index_offset;

// host
cuFloatComplex *p_iq;
float *result;
float *hamming_coef;
cuFloatComplex *fft_ma;

// device
float *d_hamming;
cuFloatComplex *d_iq;
cuFloatComplex *d_sum;
float *d_result;

// cufft
cufftHandle 
  *fft_range_handle,
  *fft_doppler_handle,
  *fft_pdop_handle;

cudaStream_t *streams;

udpserver server;
udpclient zdb_client, zdr_client;


__constant__ cuFloatComplex d_ma[512];

__global__ void __apply_hamming(cuFloatComplex *a, float *b, int b_length, int offset) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float factor = b[idx % b_length];
  a[offset + idx] = make_cuFloatComplex( factor * cuCrealf( a[offset + idx] ),
                                         factor * cuCimagf( a[offset + idx] ));
}



void setup_ports() {
  server = udpserver( 19001 );
  zdb_client = udpclient( 19002 );
  zdr_client = udpclient( 19003 );
}

void generate_constants() {
  generate_hamming_coefficients( n_samples, n_sweeps );
  generate_ma_coefficients( ma_count );
}

void generate_hamming_coefficients(int m, int n) {
  cout << "Generating Hamming coefficients..." << endl;
  // Calculate normalization power on range cell
  float p_range = 0;
  for (int i = 0; i < m; i++) {
    p_range = p_range + pow( 0.53836 - 0.46164 * cos( 2 * M_PI * (i) / (m - 1)), 2.0 );
  }
  p_range = p_range / m;

  // Calculate normalization power on Doppler cell
  float p_doppler = 0;
  for (int j = 0; j < n; j++) {
    p_doppler = p_doppler + pow( 0.53836 - 0.46164 * cos( 2 * M_PI * (j) / (n - 1)), 2.0 );
  }
  p_doppler = p_doppler / n;

  // Constant since FFT is not normalized and the power is computed w.r.t. 50ohm
  const float k_wind = -1 / (16383.5 * m * n * sqrt( 50 ));
  const float c = k_wind / sqrt( p_range * p_doppler );

  // Generate elements
  hamming_coef = new float[m * n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      hamming_coef[i * n + j] =
          (0.53836 - 0.46164 * cos( 2 * M_PI * (i) / (m - 1))) * (0.53836 - 0.46164 * cos( 2 * M_PI * (j) / (n - 1))) *
          c;
    }
  }
}

void generate_ma_coefficients(int n) {
  cout << "Generating MA coefficients..." << endl;
  float *ma_coef = new float[n];
  float _sum = 0.0;
  for (int i = 0; i < n; i++) {
    ma_coef[i] = exp( -(pow( i - ((n - 1) / 2), 2.0 )) / 2 );
    _sum += ma_coef[i];
  }
  for (int i = 0; i < n; i++) {
    ma_coef[i] = ma_coef[i] / _sum;
  }

  fftwf_complex *_fft_ma = (fftwf_complex *) fftwf_malloc( sizeof( fftwf_complex ) * n );
  fftwf_plan fft_ma_plan = fftwf_plan_dft_1d( n, _fft_ma, _fft_ma, FFTW_FORWARD, FFTW_ESTIMATE );
  for (int j = 0; j < n; j++) {
    _fft_ma[j][0] = ma_coef[j];
    _fft_ma[j][1] = 0;
  }
  for (int j = n; j < n; j++) {
    _fft_ma[j][0] = 0;
    _fft_ma[j][1] = 0;
  }
  fftwf_execute( fft_ma_plan );
  fftwf_destroy_plan( fft_ma_plan );
  fft_ma = new cuFloatComplex[n];
  for (int j = 0; j < n; j++) {
    fft_ma[j] = make_cuFloatComplex( _fft_ma[j][0], _fft_ma[j][1] );
  }
  fftwf_free( _fft_ma );
}

int main(int argc, char **argv) {

  ios_base::sync_with_stdio( false );

  int num_streams = 1;
  if (argc > 1) {
    num_streams = atoi( argv[1] );
    num_streams = num_streams < 1 ? 1 : num_streams;
  }

  setup_ports();

  generate_constants();
  prepare_arys();
  initialize_streams();
  do_process();
  destroy_streams();
  destroy_arrays();

  return 0;

}