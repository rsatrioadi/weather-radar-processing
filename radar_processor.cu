#include "radar_processor.h"

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

__constant__ cuFloatComplex d_ma[512];

__global__ void __apply_hamming(cuFloatComplex *a, float *b, int b_length, int offset) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float factor = b[idx % b_length];
  a[offset + idx] = make_cuFloatComplex( factor * cuCrealf( a[offset + idx] ),
                                         factor * cuCimagf( a[offset + idx] ));
}

RadarProcessor::RadarProcessor(int num_sectors, int num_sweeps, int num_samples, int num_elevations,
                               int num_cuda_streams) :
    n_sectors( num_sectors ),
    n_sweeps( num_sweeps ),
    n_samples( num_samples ),
    n_elevations( num_elevations ),
    n_cuda_streams( num_cuda_streams ),
    input_columns( num_samples ),
    input_rows( num_sweeps ),
    input_ary_size( num_samples * num_sweeps ),
    output_columns( this->o_types ),
    output_rows( num_sweeps / 2 ),
    output_ary_size((this->o_types) * (num_sweeps / 2)) {

  cout << "Creating RadarProcessor object. " << endl;
  hh_index_start = 0 + (input_columns * input_rows) * 0;
  vv_index_start = 0 + (input_columns * input_rows) * 1;
  vh_index_start = 0 + (input_columns * input_rows) * 2;
  input_stream_index_offset = (input_columns * input_rows) * 3;
}

int RadarProcessor::start() {
  cout << "Starting..." << endl;
  generate_constants();
  prepare_arys();
  initialize_streams();
  do_process();
  destroy_streams();
  destroy_arrays();
  return 0;
}

void RadarProcessor::set_comms(int in_port, int *out_ports, int out_length) {
  cout << "Creating UDP server for port " << in_port << "..." << endl;
  server = new udpserver( in_port );
  cout << "Creating " << out_length << " UDP clients:" << endl;
  for (int i = 0; i < out_length; i++) {
    cout << "Creating UDP client for port " << out_ports[i] << "..." << endl;
    udpclient c( out_ports[i] );
    clients.push_back( c );
  }
}

void RadarProcessor::prepare_arys() {
  cout << "Preparing arrays:" << endl;
  prepare_host_arys();
  prepare_device_arys();
}

void RadarProcessor::prepare_host_arys() {
  cout << "Preparing host arrays..." << endl;
  // 3 is for hh, vv, and vh
  cudaMallocHost((void **) &p_iq, n_cuda_streams * 3 * input_ary_size * sizeof( cuFloatComplex ));
  result = new float[output_ary_size * n_elevations];
}

void RadarProcessor::prepare_device_arys() {
  cout << "Preparing device arrays..." << endl;
  cudaMalloc( &d_hamming, input_ary_size * sizeof( float ));
  cudaMalloc( &d_iq, n_cuda_streams * 3 * input_ary_size * sizeof( cuFloatComplex ));
  cudaMalloc( &d_sum, n_cuda_streams * input_ary_size * sizeof( cuFloatComplex ));
  cudaMalloc( &d_result, output_ary_size * n_elevations * sizeof( float ));

  cudaMemcpy( d_hamming, hamming_coef, input_ary_size * sizeof( float ), cudaMemcpyHostToDevice );
//
//  float dd[input_ary_size];
//
//  cudaMemcpy(
//      dd,
//      d_hamming,
//      input_ary_size * sizeof( float ),
//      cudaMemcpyDeviceToHost );
//
//  for (int i = 0; i < n_sweeps; i++) {
//    for (int j = 0; j < n_samples; j++) {
//      cout << dd[i * n_samples + j] << " ";
//    }
//    cout << endl;
//  }
//  exit(0);
  cudaMemcpyToSymbol( d_ma, fft_ma, n_samples * sizeof( cuFloatComplex ), 0, cudaMemcpyHostToDevice );
}

void RadarProcessor::generate_constants() {
  cout << "Generating constants..." << endl;
  generate_hamming_coefficients( n_samples, n_sweeps );
  generate_ma_coefficients( ma_count );
}

void RadarProcessor::initialize_streams() {
  cout << "Initializing streams..." << endl;
  fft_range_handle = new cufftHandle[n_cuda_streams];
  fft_doppler_handle = new cufftHandle[n_cuda_streams];
  fft_pdop_handle = new cufftHandle[n_cuda_streams];

  int rank = 1;                   // --- 1D FFTs
  int nn[] = { n_sweeps };        // --- Size of the Fourier transform
  int istride = n_samples, ostride = n_samples;   // --- Distance between two successive input/output elements
  int idist = 1, odist = 1;       // --- Distance between batches
  int inembed[] = { 0 };          // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };          // --- Output size with pitch (ignored for 1D transforms)
  int batch = n_samples;          // --- Number of batched executions

  streams = new cudaStream_t[n_cuda_streams];
  for (int i = 0; i < n_cuda_streams; i++) {
    cudaStreamCreate( &streams[i] );

    cufftPlanMany( &fft_range_handle[i], rank, nn,
                   inembed, istride, idist,
                   onembed, ostride, odist, CUFFT_C2C, batch );
    cufftPlan1d( &fft_doppler_handle[i], n_samples, CUFFT_C2C, n_sweeps );
    cufftPlan1d( &fft_pdop_handle[i], n_samples, CUFFT_C2C, output_rows );

    cufftSetStream( fft_range_handle[i], streams[i] );
    cufftSetStream( fft_doppler_handle[i], streams[i] );
    cufftSetStream( fft_pdop_handle[i], streams[i] );
  }
}

void RadarProcessor::do_process() {
  cout << "Starting main loop..." << endl;
  read_matrix( current_sector, current_elevation );
  copy_matrix_to_device( current_sector, current_elevation, current_stream );
  do {
    perform_stage_1( current_stream );
    perform_stage_2( current_stream );
    perform_stage_3( current_stream );
    int
        prev_sector = current_sector,
        prev_elevation = current_elevation,
        prev_stream = current_stream;
    advance();
    read_matrix( current_sector, current_elevation );
    copy_matrix_to_device( current_sector, current_elevation, current_stream );
    copy_result_to_host( prev_sector, prev_elevation, prev_stream );
    send_results();
  } while (true);
}

void RadarProcessor::read_matrix(int sector, int elevation) {
  cout << "Reading matrices from network..." << endl;
  char *buff = new char[NUM_BYTES_PER_SAMPLE * n_samples * n_sweeps];
  for (int j = 0; j < n_sweeps; j++) {
    server->recv( buff + j * (NUM_BYTES_PER_SAMPLE * n_samples), NUM_BYTES_PER_SAMPLE * n_samples );
  }
  //cout << "done!" << endl;
  // cout << sector_id << " received." << endl;
  Sector s( n_sweeps, n_samples );
  s.fromByteArray( buff );
  delete[] buff;

  int a, b;
  int offset = sector * input_stream_index_offset;

  // cout << "bikin matriks" << endl;
  int idx = 0;
  #pragma unroll
  for (int i = 0; i < n_sweeps; i++) {
    #pragma unroll
    for (int j = 0; j < n_samples; j++) {
      // cin >> a >> b;
      a = idx++;
      b = idx++;
      p_iq[offset + hh_index_start + i * n_samples + j] = make_cuFloatComplex( s.hh[a], s.hh[b] );
      p_iq[offset + vv_index_start + i * n_samples + j] = make_cuFloatComplex( s.vv[a], s.vv[b] );
      p_iq[offset + vh_index_start + i * n_samples + j] = make_cuFloatComplex( s.vh[a], s.vh[b] );
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

void RadarProcessor::copy_matrix_to_device(int sector, int elevation, int stream) {
  cout << "Copying matrices to device..." << endl;
  int offset = stream * input_stream_index_offset;
  cudaMemcpyAsync(
      &d_iq[offset],
      &p_iq[offset],
      input_stream_index_offset * sizeof( cuFloatComplex ),
      cudaMemcpyHostToDevice,
      streams[stream] );
}

void RadarProcessor::perform_stage_1(int stream) {
  cout << "Performing Stage I..." << endl;
  int offset = stream * input_stream_index_offset;
  cout << "n_sweeps: " << n_sweeps << endl;
  cout << "n_samples: " << n_samples << endl;
  cout << "stream: " << stream << endl;
  cout << "input_ary_size: " << input_ary_size << endl;
  cout << "offset: " << offset << endl;
  __apply_hamming<<<n_sweeps,n_samples*3,0,streams[stream]>>>( d_iq, d_hamming, input_ary_size, offset );
}

void RadarProcessor::perform_stage_2(int stream) {
  cout << "Performing Stage II..." << endl;

}

void RadarProcessor::perform_stage_3(int stream) {
  cout << "Performing Stage III..." << endl;

}

void RadarProcessor::advance() {
  cout << "Advancing to next sector..." << endl;
  current_sector = (current_sector + 1) % n_sectors;
  if (current_sector == 0) {
    current_elevation = (current_elevation + 1) % n_elevations;
  }
}

void RadarProcessor::copy_result_to_host(int sector, int elevation, int stream) {

  cout << 1 << endl;
  cuFloatComplex *dump = new cuFloatComplex[input_stream_index_offset];
  cout << 2 << endl;
  cudaMemcpyAsync(
      dump,
      &d_iq[stream * input_stream_index_offset],
      input_stream_index_offset * sizeof( cuFloatComplex ),
      cudaMemcpyDeviceToHost,
      streams[stream] );
  cout << 3 << endl;

  for (int i = 0; i < n_sweeps; i++) {
    for (int j = 0; j < n_samples; j++) {
      cout << "(" << dump[i * n_samples + j].x << "," << dump[i * n_samples + j].y << ") ";
    }
    cout << endl;
  }
  cout << 4 << endl;
  exit( 0 );

  cout << "Copying result to host..." << endl;
  int stream_offset = stream * output_ary_size;
  int elevation_offset = elevation * output_ary_size;

  cudaMemcpyAsync(
      &result[elevation_offset],
      &d_result[stream_offset],
      output_ary_size * sizeof( float ),
      cudaMemcpyDeviceToHost,
      streams[stream] );
}

void RadarProcessor::send_results() {
  cout << "Sending results to network..." << endl;

}

void RadarProcessor::destroy_streams() {
  cout << "Destroying streams..." << endl;

}

void RadarProcessor::destroy_arrays() {
  cout << "Destroying arrays:" << endl;
  destroy_device_arys();
  destroy_host_arys();
}

void RadarProcessor::destroy_device_arys() {
  cout << "Destroying device arrays..." << endl;

}

void RadarProcessor::destroy_host_arys() {
  cout << "Destroying host arrays..." << endl;

}

void RadarProcessor::generate_hamming_coefficients(int m, int n) {
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

void RadarProcessor::generate_ma_coefficients(int n) {
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
  cout << "Running with " << num_streams << " streams." << endl;
  RadarProcessor proc( 143, 1024, 512, 9, num_streams );
  int *out_ports = new int[2];
  out_ports[0] = 19002;
  out_ports[1] = 19003;
  proc.set_comms( 19001, out_ports, 2 );
  proc.start();
}
