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
#include "dimension.h"

#define NUM_BYTES_PER_SAMPLE (3*2*2)

using namespace std;
using namespace udpbroadcast;

const int 
  n_sectors=143, 
  n_sweeps=1024, 
  n_samples=512, 
  n_elevations=9;
static const int k_range_resolution = 30;
static constexpr float k_calibration = 1941.05;
static const int ma_count = 7;
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

udpserver *server;
udpclient *zdb_client, *zdr_client;


__constant__ cuFloatComplex d_ma[512];

//__global__ void __apply_hamming(cuFloatComplex *a, float *b, int b_length, int offset) {
//  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  float factor = b[idx % b_length];
//  a[offset + idx] = make_cuFloatComplex( factor * cuCrealf( a[offset + idx] ),
//                                         factor * cuCimagf( a[offset + idx] ));
//}


__global__ void __apply_hamming(cuFloatComplex *a, float *b, int offset) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[offset+idx] = make_cuFloatComplex(b[idx]*cuCrealf(a[offset+idx]), b[idx]*cuCimagf(a[offset+idx]));
}



void setup_ports() {
  server = new udpserver( 19001 );
  zdb_client = new udpclient( 19002 );
  zdr_client = new udpclient( 19003 );
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

void generate_constants(Dimension4 dim, int ma_count) {
  cout << "Generating constants..." << endl;
  generate_hamming_coefficients( dim.width, dim.height );
  generate_ma_coefficients( ma_count );
}

void prepare_host_arys(Dimension4 idim, Dimension4 odim) {
  cout << "Preparing host arrays..." << endl;
  // 3 is for hh, vv, and vh
  cudaMallocHost((void **) &p_iq, idim.total_size * sizeof( cuFloatComplex ));
  result = new float[odim.total_size];
}

void prepare_device_arys(Dimension4 idim, Dimension4 odim) {
  cout << "Preparing device arrays..." << endl;
  cudaMalloc( &d_hamming, idim.m_size * sizeof( float ));
  cudaMalloc( &d_iq, idim.total_size * sizeof( cuFloatComplex ));
  cudaMalloc( &d_sum, idim.total_size * sizeof( cuFloatComplex ));
  cudaMalloc( &d_result, odim.total_size * sizeof( float ));

  cudaMemcpy( d_hamming, hamming_coef, idim.m_size * sizeof( float ), cudaMemcpyHostToDevice );
  cudaMemcpyToSymbol( d_ma, fft_ma, idim.width * sizeof( cuFloatComplex ), 0, cudaMemcpyHostToDevice );
}

void prepare_arys(Dimension4 idim, Dimension4 odim) {
  cout << "Preparing arrays:" << endl;
  prepare_host_arys( idim, odim );
  prepare_device_arys( idim, odim );
}

void initialize_streams(Dimension4 idim, Dimension4 odim) {
  cout << "Initializing streams..." << endl;
  fft_range_handle = new cufftHandle[idim.depth];
  fft_doppler_handle = new cufftHandle[idim.depth];
  fft_pdop_handle = new cufftHandle[idim.depth];

  int rank = 1;                   // --- 1D FFTs
  int nn[] = { idim.height };        // --- Size of the Fourier transform
  int istride = idim.width, ostride = idim.width;   // --- Distance between two successive input/output elements
  int idist = 1, odist = 1;       // --- Distance between batches
  int inembed[] = { 0 };          // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };          // --- Output size with pitch (ignored for 1D transforms)
  int batch = idim.width;          // --- Number of batched executions

  streams = new cudaStream_t[idim.depth];
  for (int i = 0; i < idim.depth; i++) {
    cudaStreamCreate( &streams[i] );

    cufftPlanMany( &fft_range_handle[i], rank, nn,
                   inembed, istride, idist,
                   onembed, ostride, odist, CUFFT_C2C, batch );
    cufftPlan1d( &fft_doppler_handle[i], idim.width, CUFFT_C2C, idim.height );
    cufftPlan1d( &fft_pdop_handle[i], idim.width, CUFFT_C2C, odim.height );

    cufftSetStream( fft_range_handle[i], streams[i] );
    cufftSetStream( fft_doppler_handle[i], streams[i] );
    cufftSetStream( fft_pdop_handle[i], streams[i] );
  }
}

void read_matrix(Dimension4 idim, int sector, int elevation) {
  cout << "Reading matrices from network..." << endl;
  char *buff = new char[NUM_BYTES_PER_SAMPLE * idim.m_size];
  for (int j = 0; j < idim.height; j++) {
    server->recv( buff + j * (NUM_BYTES_PER_SAMPLE * idim.width), NUM_BYTES_PER_SAMPLE * idim.width );
  }
  //cout << "done!" << endl;
  // cout << sector_id << " received." << endl;
  Sector s( idim.height, idim.width );
  s.fromByteArray( buff );
  delete[] buff;

  int a, b;

  // cout << "bikin matriks" << endl;
  int idx = 0;
  #pragma unroll
  for (int i = 0; i < idim.height; i++) {
    #pragma unroll
    for (int j = 0; j < idim.width; j++) {
      // cin >> a >> b;
      a = idx++;
      b = idx++;
      p_iq[idim.copy_at_depth(j,i,0,0)] = make_cuFloatComplex( s.hh[a], s.hh[b] );
      p_iq[idim.copy_at_depth(j,i,1,0)] = make_cuFloatComplex( s.vv[a], s.vv[b] );
      p_iq[idim.copy_at_depth(j,i,2,0)] = make_cuFloatComplex( s.vh[a], s.vh[b] );
    }
  }

  // for (int i=0; i<n_sweeps; i++) {
  //   for (int j=0; j<n_samples; j++) {
  //     cout << "(" << p_iq[hh_index_start+i*n_samples+j].x << "," << p_iq[hh_index_start+i*n_samples+j].y << ") ";
  //   }
  //   cout << endl;
  // }
  // for (int i=0; i<n_sweeps; i++) {
  //   for (int j=0; j<n_samples; j++) {
  //     cout << "(" << p_iq[vv_index_start+i*n_samples+j].x << "," << p_iq[vv_index_start+i*n_samples+j].y << ") ";
  //   }
  //   cout << endl;
  // }
  // for (int i=0; i<n_sweeps; i++) {
  //   for (int j=0; j<n_samples; j++) {
  //     cout << "(" << p_iq[vh_index_start+i*n_samples+j].x << "," << p_iq[vh_index_start+i*n_samples+j].y << ") ";
  //   }
  //   cout << endl;
  // }
  // exit(0);
}

void copy_matrix_to_device(Dimension4 idim, int sector, int elevation, int stream) {
  cout << "Copying matrices to device..." << endl;
  cudaMemcpyAsync(
      &d_iq[idim.copy_at_depth(0,0,0,stream)],
      &p_iq[idim.copy_at_depth(0,0,0,stream)],
      idim.m_size * idim.copies * sizeof( cuFloatComplex ),
      cudaMemcpyHostToDevice,
      streams[stream] );
}

void perform_stage_1(Dimension4 idim, int stream) {
  cout << "Performing Stage I..." << endl;
  cout << "n_sweeps: " << idim.height << endl;
  cout << "n_samples: " << idim.width << endl;
  cout << "stream: " << stream << endl;
  cout << "input_ary_size: " << idim.m_size << endl;
  cout << "offset: " << idim.copy_at_depth(0,0,0,stream) << endl;
  //__apply_hamming<<<idim.height,idim.width*idim.copies,0,streams[stream]>>>( d_iq, d_hamming, idim.m_size, idim.copy_at_depth(0,0,0,stream) );
  __apply_hamming<<<idim.height,idim.width,0,streams[stream]>>>( d_iq, d_hamming, idim.copy_at_depth(0,0,0,stream) );
}

void perform_stage_2(Dimension4 idim, int stream) {
  cout << "Performing Stage II..." << endl;

}

void perform_stage_3(Dimension4 idim, Dimension4 odim, int stream) {
  cout << "Performing Stage III..." << endl;

}

void advance() {
  cout << "Advancing to next sector..." << endl;
  current_sector = (current_sector + 1) % n_sectors;
  if (current_sector == 0) {
    current_elevation = (current_elevation + 1) % n_elevations;
  }
}

void copy_result_to_host(Dimension4 idim, Dimension4 odim, int sector, int elevation, int stream) {

  cout << 1 << endl;
  cuFloatComplex *dump = new cuFloatComplex[idim.m_size];
  cout << 2 << endl;
  cudaMemcpyAsync(
      dump,
      &d_iq[idim.copy_at_depth(0,0,0,stream)],
      idim.m_size * sizeof( cuFloatComplex ),
      cudaMemcpyDeviceToHost,
      streams[stream] );
  cout << 3 << endl;

  for (int i = 0; i < idim.height; i++) {
    for (int j = 0; j < idim.width; j++) {
      int idx = idim.copy_at_depth(j,i,0,0);
      cout << "(" << dump[idx].x << "," << dump[idx].y << ") ";
    }
    cout << endl;
  }
  cout << 4 << endl;
  exit( 0 );

  cout << "Copying result to host..." << endl;

  cudaMemcpyAsync(
      &result[odim.copy_at_depth(0,0,0,elevation)],
      &d_result[idim.copy_at_depth(0,0,0,stream)],
      odim.m_size * odim.copies * sizeof( float ),
      cudaMemcpyDeviceToHost,
      streams[stream] );
}

void send_results(Dimension4 odim) {
  cout << "Sending results to network..." << endl;

}

void do_process(Dimension4 idim, Dimension4 odim) {
  cout << "Starting main loop..." << endl;
  read_matrix( idim, current_sector, current_elevation );
  copy_matrix_to_device( idim, current_sector, current_elevation, current_stream );
  do {
    perform_stage_1( idim, current_stream );
    perform_stage_2( idim, current_stream );
    perform_stage_3( idim, odim, current_stream );
    int
        prev_sector = current_sector,
        prev_elevation = current_elevation,
        prev_stream = current_stream;
    advance();
    read_matrix( idim, current_sector, current_elevation );
    copy_matrix_to_device( idim, current_sector, current_elevation, current_stream );
    copy_result_to_host( idim, odim, prev_sector, prev_elevation, prev_stream );
    send_results( odim );
  } while (true);
}

void destroy_streams() {
  cout << "Destroying streams..." << endl;

}

void destroy_device_arys() {
  cout << "Destroying device arrays..." << endl;

}

void destroy_host_arys() {
  cout << "Destroying host arrays..." << endl;

}

void destroy_arrays() {
  cout << "Destroying arrays:" << endl;
  destroy_device_arys();
  destroy_host_arys();
}

int main(int argc, char **argv) {

  ios_base::sync_with_stdio( false );

  int num_streams = 1;
  if (argc > 1) {
    num_streams = atoi( argv[1] );
    num_streams = num_streams < 1 ? 1 : num_streams;
  }

  Dimension4 idim(n_samples,n_sweeps,3,num_streams);
  Dimension4 odim(1,n_sweeps/2,2,n_elevations);

  setup_ports();

  generate_constants( idim, 7 );
  prepare_arys( idim, odim );
  initialize_streams( idim, odim );
  do_process( idim, odim );
  destroy_streams();
  destroy_arrays();

  return 0;
}