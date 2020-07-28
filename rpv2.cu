#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <cufft.h>
#include <sys/time.h>
#include <assert.h>
#include "floats.h"
#include "dimension.h"

#include <msgpack.hpp>
#include "zhelpers.hpp"

#define NUM_BYTES_PER_SAMPLE (3*2*2)

using namespace std;

struct sectormsg {
  int id;
  vector<int> 
      i_hh, q_hh,
      i_vv, q_vv,
      i_vh, q_vh;
  MSGPACK_DEFINE_MAP( id, i_hh, q_hh, i_vv, q_vv, i_vh, q_vh );
};

const int
    n_sectors = 143,
    n_sweeps = 1024,
    n_samples = 512,
    n_elevations = 9;
static const int k_range_resolution = 30;
static constexpr float k_calibration = 1941.05;
static const int ma_count = 7;
int
    current_sector = 0,
    current_sweep = 0,
    current_sample = 0,
    current_elevation = 0,
    current_stream = 0;

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
cuFloatComplex *d_tmp;
float *d_result;

// cufft
cufftHandle
    *fft_range_handle,
    *fft_doppler_handle,
    *fft_pdop_handle;

cudaStream_t *streams;

zmq::context_t context( 1 );
zmq::socket_t subscriber( context, ZMQ_SUB );
zmq::socket_t publisher( context, ZMQ_PUB );


__constant__ cuFloatComplex d_ma[512];

__global__ void __apply_hamming(cuFloatComplex *a, float *b, int offset) {
  const unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
  a[offset + idx] = make_cuFloatComplex(
      b[idx]*cuCrealf( a[offset + idx] ),
      b[idx]*cuCimagf( a[offset + idx] ));
}

__global__ void __sum_v4(cuFloatComplex *in, cuFloatComplex *out, int offset) {
  const unsigned int i = 2*blockIdx.x, j = threadIdx.x, n = blockDim.x;
  extern __shared__ cuFloatComplex sdata[];

#pragma unroll
  for (unsigned int d = 0; d < 2; d++) {
    sdata[j + n*d] = make_cuFloatComplex(
        in[offset + j + i*n + n*d].x,
        in[offset + j + i*n + n*d].y );
  }
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (j < s) {
#pragma unroll
      for (unsigned int d = 0; d < 2; d++) {
        sdata[j + n*d] = cuCaddf( sdata[j + n*d], sdata[j + n*d + s] );
      }
    }
    __syncthreads();
  }

  if (j == 0) {
#pragma unroll
    for (unsigned int d = 0; d < 2; d++) {
      out[i*n + n*d] = sdata[j + n*d];
    }
  }
}

__global__ void __avgconj(cuFloatComplex *inout, cuFloatComplex *sum, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  float avgx = sum[offset + i*n].x/n;
  float avgy = sum[offset + i*n].y/n;
  inout[offset + j + i*n] = make_cuFloatComplex( inout[offset + j + i*n].x - avgx,
                                                 (inout[offset + j + i*n].y - avgy)*-1 );
}

__global__ void __conjugate(cuFloatComplex *a, int offset) {
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  a[offset + idx].y *= -1;
}

__global__ void __shift(cuFloatComplex *inout, int n, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x;

  cuFloatComplex temp = inout[offset + j + i*n];
  inout[offset + j + i*n] = inout[offset + (j + n/2) + i*n];
  inout[offset + (j + n/2) + i*n] = temp;
}

__global__ void __clip_v2(cuFloatComplex *inout, int n, int offset) {
  const unsigned int i = threadIdx.x, j = n - blockIdx.x - 1;
  inout[offset + j + i*n] = make_cuFloatComplex( 0, 0 );
}

__global__ void __abssqr(cuFloatComplex *inout, int offset) {
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  float real, imag;
  real = cuCrealf( inout[offset + idx] );
  imag = cuCimagf( inout[offset + idx] );
  inout[offset + idx] = make_cuFloatComplex( real*real + imag*imag, 0 );
}

__global__ void __apply_ma(cuFloatComplex *inout, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  inout[offset + j + i*n] = cuCmulf( inout[offset + j + i*n], d_ma[j] );
}

__global__ void __scale_real(cuFloatComplex *inout, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  inout[offset + j + i*n] = make_cuFloatComplex( inout[offset + j + i*n].x/n, 0 );
}

__global__ void __sum_inplace_v4(cuFloatComplex *in, int offset) {
  const unsigned int i = 2*blockIdx.x, j = threadIdx.x, n = blockDim.x;
  extern __shared__ cuFloatComplex sdata[];

#pragma unroll
  for (unsigned int d = 0; d < 2; d++) {
    sdata[j + n*d] = make_cuFloatComplex( in[offset + j + i*n + n*d].x, in[offset + j + i*n + n*d].y );
  }
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (j < s) {
#pragma unroll
      for (unsigned int d = 0; d < 2; d++) {
        sdata[j + n*d] = cuCaddf( sdata[j + n*d], sdata[j + n*d + s] );
      }
    }
    __syncthreads();
  }

  if (j == 0) {
#pragma unroll
    for (unsigned int d = 0; d < 2; d++) {
      in[offset + (i*n + n*d)] = sdata[j + n*d];
    }
  }
}

__global__ void __calcresult_v2(
    cuFloatComplex *iq,
    float *out,
    int n,
    int offset_hh, int offset_vv, int offset_vh,
    int result_offset) {

  const unsigned int i = threadIdx.x;

  float z = pow( i*k_range_resolution, 2.0 )*k_calibration*iq[offset_hh + i*n].x;
  float zdb = 10*log10( z );
  float zdr = 10*(log10( iq[offset_hh + i*n].x ) - log10( iq[offset_vv + i*n].x ));
  out[result_offset + i*2 + 0] = zdb;
  out[result_offset + i*2 + 1] = zdr;
}


void setup_ports() {
  subscriber.connect( "tcp://localhost:5563" );
  subscriber.setsockopt( ZMQ_SUBSCRIBE, "A", 1 );
  publisher.bind( "tcp://*:5563" );
}

void generate_hamming_coefficients(int m, int n) {
  cout << "Generating Hamming coefficients..." << endl;
  // Calculate normalization power on range cell
  float p_range = 0;
  for (int i = 0; i < m; i++) {
    p_range = p_range + pow( 0.53836 - 0.46164*cos( 2*M_PI*(i)/(m - 1)), 2.0 );
  }
  p_range = p_range/m;

  // Calculate normalization power on Doppler cell
  float p_doppler = 0;
  for (int j = 0; j < n; j++) {
    p_doppler = p_doppler + pow( 0.53836 - 0.46164*cos( 2*M_PI*(j)/(n - 1)), 2.0 );
  }
  p_doppler = p_doppler/n;

  // Constant since FFT is not normalized and the power is computed w.r.t. 50ohm
  const float k_wind = -1/(16383.5*m*n*sqrt( 50 ));
  const float c = k_wind/sqrt( p_range*p_doppler );

  // Generate elements
  hamming_coef = new float[m*n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      hamming_coef[j + i*n] =
          (0.53836 - 0.46164*cos( 2*M_PI*(i)/(m - 1)))*(0.53836 - 0.46164*cos( 2*M_PI*(j)/(n - 1)))*c;
    }
  }
}

void generate_ma_coefficients(Dimension4 dim, int n) {
  cout << "Generating MA coefficients..." << endl;
  float *ma_coef = new float[n];
  float _sum = 0.0;
  for (int i = 0; i < n; i++) {
    ma_coef[i] = exp( -(pow( i - ((n - 1)/2), 2.0 ))/2 );
    _sum += ma_coef[i];
  }
  for (int i = 0; i < n; i++) {
    ma_coef[i] = ma_coef[i]/_sum;
  }

  fftwf_complex *_fft_ma = (fftwf_complex *) fftwf_malloc( sizeof( fftwf_complex )*dim.width );
  fftwf_plan fft_ma_plan = fftwf_plan_dft_1d( dim.width, _fft_ma, _fft_ma, FFTW_FORWARD, FFTW_ESTIMATE );
  for (int j = 0; j < n; j++) {
    _fft_ma[j][0] = ma_coef[j];
    _fft_ma[j][1] = 0;
  }
  for (int j = n; j < dim.width; j++) {
    _fft_ma[j][0] = 0;
    _fft_ma[j][1] = 0;
  }
  fftwf_execute( fft_ma_plan );
  fftwf_destroy_plan( fft_ma_plan );
  fft_ma = new cuFloatComplex[dim.width];
  for (int j = 0; j < dim.width; j++) {
    fft_ma[j] = make_cuFloatComplex( _fft_ma[j][0], _fft_ma[j][1] );
  }
  fftwf_free( _fft_ma );
}

void generate_constants(Dimension4 dim, int ma_count) {
  cout << "Generating constants..." << endl;
  generate_hamming_coefficients( dim.height, dim.width );
  generate_ma_coefficients( dim, ma_count );
}

void prepare_host_arys(Dimension4 idim, Dimension4 sitdim) {
  cout << "Preparing host arrays..." << endl;
  cudaMallocHost((void **) &p_iq, idim.total_size*sizeof( cuFloatComplex ));
  result = new float[sitdim.total_size];
}

void prepare_device_arys(Dimension4 idim, Dimension4 odim) {
  cout << "Preparing device arrays..." << endl;
  cudaMalloc( &d_hamming, idim.m_size*sizeof( float ));
  cudaMalloc( &d_iq, idim.total_size*sizeof( cuFloatComplex ));
  cudaMalloc( &d_tmp, idim.m_size*sizeof( cuFloatComplex ));
  cudaMalloc( &d_result, odim.total_size*sizeof( float ));

  cudaMemcpy( d_hamming, hamming_coef, idim.m_size*sizeof( float ), cudaMemcpyHostToDevice );
  cudaMemcpyToSymbol( d_ma, fft_ma, idim.width*sizeof( cuFloatComplex ), 0, cudaMemcpyHostToDevice );
}

void prepare_arys(Dimension4 idim, Dimension4 odim, Dimension4 sitdim) {
  cout << "Preparing arrays:" << endl;
  prepare_host_arys( idim, sitdim );
  prepare_device_arys( idim, odim );
}

void initialize_streams(Dimension4 idim, Dimension4 odim) {
  cout << "Initializing streams..." << endl;
  fft_range_handle = new cufftHandle[idim.depth];
  fft_doppler_handle = new cufftHandle[idim.depth];
  fft_pdop_handle = new cufftHandle[idim.depth];

  int rank = 1;                 // --- 1D FFTs
  int nn[] = { idim.height };   // --- Size of the Fourier transform
  int istride = idim.width,     // --- Distance between two successive input/output elements
  ostride = idim.width;
  int idist = 1, odist = 1;     // --- Distance between batches
  int inembed[] = { 0 };        // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };        // --- Output size with pitch (ignored for 1D transforms)
  int batch = idim.width;       // --- Number of batched executions

  streams = new cudaStream_t[idim.depth];
  for (int i = 0; i < idim.depth; i++) {
    cudaStreamCreate( &streams[i] );

    cufftPlanMany( &fft_range_handle[i], rank, nn,
                   inembed, istride, idist,
                   onembed, ostride, odist, CUFFT_C2C, batch );
    cufftPlan1d( &fft_doppler_handle[i], idim.width, CUFFT_C2C, idim.height );
    cufftPlan1d( &fft_pdop_handle[i], idim.width, CUFFT_C2C, idim.height/2 );

    cufftSetStream( fft_range_handle[i], streams[i] );
    cufftSetStream( fft_doppler_handle[i], streams[i] );
    cufftSetStream( fft_pdop_handle[i], streams[i] );
  }
}

void read_matrix(Dimension4 idim, int sector, int elevation, int stream) {
  cout << "Reading matrices from network..." << endl;
  
  //  Read envelope with address
  string address = s_recv( subscriber );
  //  Read message contents
  string str = s_recv( subscriber );

  msgpack::object_handle oh = msgpack::unpack( str.data(), str.size() );
  msgpack::object deserialized = oh.get();

  sectormsg s;
  deserialized.convert( s );

  int idx = 0;
#pragma unroll
  for (int j = 0; j < idim.height; j++) {
#pragma unroll
    for (int i = 0; i < idim.width; i++) {
      // cin >> a >> b;
      p_iq[idim.copy_at_depth( i, j, 0, stream )] = make_cuFloatComplex( s.i_hh[idx], s.q_hh[idx] );
      p_iq[idim.copy_at_depth( i, j, 1, stream )] = make_cuFloatComplex( s.i_vv[idx], s.q_vv[idx] );
      p_iq[idim.copy_at_depth( i, j, 2, stream )] = make_cuFloatComplex( s.i_vh[idx], s.q_vh[idx] );
      idx++;
    }
  }

//  for (int j = 0; j < idim.height; j++) {
//    for (int i = 0; i < idim.width; i++) {
//      int idx = idim.copy_at_depth( i, j, 0, stream );
//      cout << "(" << p_iq[idx].x << "," << p_iq[idx].y << ") ";
//    }
//    cout << endl;
//  }
//  exit( 0 );
}

void copy_matrix_to_device(Dimension4 idim, int sector, int elevation, int stream) {
  cout << "Copying matrices to device..." << endl;
  cudaMemcpyAsync(
      &d_iq[idim.copy_at_depth( 0, 0, 0, stream )],
      &p_iq[idim.copy_at_depth( 0, 0, 0, stream )],
      idim.m_size*idim.copies*sizeof( cuFloatComplex ),
      cudaMemcpyHostToDevice,
      streams[stream] );
}

void perform_stage_1(Dimension4 idim, int stream) {
  cout << "Performing Stage I..." << endl;

  int
      offset_hh = idim.copy_at_depth( 0, 0, 0, stream ),
      offset_vv = idim.copy_at_depth( 0, 0, 1, stream ),
      offset_vh = idim.copy_at_depth( 0, 0, 2, stream );

  // apply Hamming coefficients
  __apply_hamming<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, d_hamming, offset_hh );
  __apply_hamming<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, d_hamming, offset_vv );
  __apply_hamming<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, d_hamming, offset_vh );

  // FFT range profile
  cufftExecC2C( fft_range_handle[stream], &d_iq[offset_hh], &d_iq[offset_hh], CUFFT_FORWARD );
  cufftExecC2C( fft_range_handle[stream], &d_iq[offset_vv], &d_iq[offset_vv], CUFFT_FORWARD );
  cufftExecC2C( fft_range_handle[stream], &d_iq[offset_vh], &d_iq[offset_vh], CUFFT_FORWARD );

  // FFT+shift Doppler profile
  __sum_v4<<<idim.height/2, idim.width, 2*idim.width*sizeof(cuFloatComplex), streams[stream]>>>( d_iq, d_tmp,
                                                                                                 offset_hh );
  __avgconj<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, d_tmp, offset_hh );
  __sum_v4<<<idim.height/2, idim.width, 2*idim.width*sizeof(cuFloatComplex), streams[stream]>>>( d_iq, d_tmp,
                                                                                                 offset_vv );
  __avgconj<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, d_tmp, offset_vv );
  __sum_v4<<<idim.height/2, idim.width, 2*idim.width*sizeof(cuFloatComplex), streams[stream]>>>( d_iq, d_tmp,
                                                                                                 offset_vh );
  __avgconj<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, d_tmp, offset_vh );

  cufftExecC2C( fft_doppler_handle[stream], &d_iq[offset_hh], &d_iq[offset_hh], CUFFT_FORWARD );
  cufftExecC2C( fft_doppler_handle[stream], &d_iq[offset_vv], &d_iq[offset_vv], CUFFT_FORWARD );
  cufftExecC2C( fft_doppler_handle[stream], &d_iq[offset_vh], &d_iq[offset_vh], CUFFT_FORWARD );

  __conjugate<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, offset_hh );
  __conjugate<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, offset_vv );
  __conjugate<<<idim.height, idim.width, 0, streams[stream]>>>( d_iq, offset_vh );

  __shift<<<idim.height, idim.width/2, 0, streams[stream]>>>( d_iq, idim.width, offset_hh );
  __shift<<<idim.height, idim.width/2, 0, streams[stream]>>>( d_iq, idim.width, offset_vv );
  __shift<<<idim.height, idim.width/2, 0, streams[stream]>>>( d_iq, idim.width, offset_vh );

  __clip_v2<<<2, idim.height, 0, streams[stream]>>>( d_iq, idim.width, offset_hh );
  __clip_v2<<<2, idim.height, 0, streams[stream]>>>( d_iq, idim.width, offset_vv );
  __clip_v2<<<2, idim.height, 0, streams[stream]>>>( d_iq, idim.width, offset_vh );
}

void perform_stage_2(Dimension4 idim, int stream) {
  cout << "Performing Stage II..." << endl;

  int
      offset_hh = idim.copy_at_depth( 0, 0, 0, stream ),
      offset_vv = idim.copy_at_depth( 0, 0, 1, stream ),
      offset_vh = idim.copy_at_depth( 0, 0, 2, stream );

  // Get absolute value squared
  __abssqr<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_hh );
  __abssqr<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_vv );
  __abssqr<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_vh );

  // FFT PDOP
  cufftExecC2C( fft_pdop_handle[stream], &d_iq[offset_hh], &d_iq[offset_hh], CUFFT_FORWARD );
  cufftExecC2C( fft_pdop_handle[stream], &d_iq[offset_vv], &d_iq[offset_vv], CUFFT_FORWARD );
  cufftExecC2C( fft_pdop_handle[stream], &d_iq[offset_vh], &d_iq[offset_vh], CUFFT_FORWARD );

  // Apply MA coefficients
  __apply_ma<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_hh );
  __apply_ma<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_vv );
  __apply_ma<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_vh );

  // Inverse FFT
  cufftExecC2C( fft_pdop_handle[stream], &d_iq[offset_hh], &d_iq[offset_hh], CUFFT_INVERSE );
  cufftExecC2C( fft_pdop_handle[stream], &d_iq[offset_vv], &d_iq[offset_vv], CUFFT_INVERSE );
  cufftExecC2C( fft_pdop_handle[stream], &d_iq[offset_vh], &d_iq[offset_vh], CUFFT_INVERSE );

  __scale_real<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_hh );
  __scale_real<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_vv );
  __scale_real<<<idim.height/2, idim.width, 0, streams[stream]>>>( d_iq, offset_vh );

  // Sum
  __sum_inplace_v4<<<idim.height/4, idim.width, 2*idim.width*sizeof(cuFloatComplex), streams[stream]>>>( d_iq,
                                                                                                         offset_hh );
  __sum_inplace_v4<<<idim.height/4, idim.width, 2*idim.width*sizeof(cuFloatComplex), streams[stream]>>>( d_iq,
                                                                                                         offset_vv );
  __sum_inplace_v4<<<idim.height/4, idim.width, 2*idim.width*sizeof(cuFloatComplex), streams[stream]>>>( d_iq,
                                                                                                         offset_vh );
}

void perform_stage_3(Dimension4 idim, Dimension4 odim, int sector, int elevation, int stream) {
  cout << "Performing Stage III..." << endl;

  int
      offset_hh = idim.copy_at_depth( 0, 0, 0, stream ),
      offset_vv = idim.copy_at_depth( 0, 0, 1, stream ),
      offset_vh = idim.copy_at_depth( 0, 0, 2, stream );

  // Calculate ZdB, Zdr
  __calcresult_v2<<<1, idim.height/2, 0, streams[stream]>>>(
      d_iq,
      d_result,
      idim.width,
      offset_hh, offset_vv, offset_vh,
      odim.copy_at_depth( 0, 0, 0, stream ));
}

void advance(Dimension4 idim) {
  cout << "Advancing to next sector..." << endl;
  current_sector = (current_sector + 1)%n_sectors;
  if (current_sector == 0) {
    current_elevation = (current_elevation + 1)%n_elevations;
  }
  current_stream = (current_stream + 1)%idim.depth;
}

void copy_result_to_host(Dimension4 idim, Dimension4 odim, Dimension4 sitdim, int sector, int elevation, int stream) {
/*
  cout << 1 << endl;
  cuFloatComplex *dump = new cuFloatComplex[idim.m_size];
  cout << 2 << endl;
  cudaMemcpyAsync(
      dump,
      &d_iq[idim.copy_at_depth( 0, 0, 0, stream )],
      idim.m_size * sizeof( cuFloatComplex ),
      cudaMemcpyDeviceToHost,
      streams[stream] );
  cout << 3 << endl;

  for (int j = 0; j < idim.height/2; j++) {
    for (int i = 0; i < idim.width; i++) {
      int idx = idim.copy_at_depth( i, j, 0, 0 );
      cout << "(" << dump[idx].x << "," << dump[idx].y << ") ";
    }
    cout << endl;
  }
  cout << 4 << endl;
  exit( 0 );
*/
  cout << "Copying result to host..." << endl;

  cudaMemcpyAsync(
      &result[sitdim.copy_at_depth( 0, 0, sector, elevation )],
      &d_result[odim.copy_at_depth( 0, 0, 0, stream )],
      odim.m_size*sizeof( float ),
      cudaMemcpyDeviceToHost,
      streams[stream] );

//  cout << "zdb:" << endl;
//  for (int i=0; i<sitdim.height; i++) {
//    cout << result[sitdim.copy_at_depth(0,i,sector,elevation)] << endl;
//  }
//  exit(0);
}

void send_results(Dimension4 sitdim, int sector, int elevation) {
  cout << "Sending results to network..." << endl;

  float *zdb = new float[sitdim.height];
  float *zdr = new float[sitdim.height];

  for (int i = 0; i < sitdim.height; i++) {
    zdb[i] = result[sitdim.copy_at_depth( 0, i, sector, elevation )];
    zdr[i] = result[sitdim.copy_at_depth( 1, i, sector, elevation )];
  }

  unsigned char *zdb_buff = new unsigned char[sizeof( float )*sitdim.height + 2]; // + 2 for sector id
  unsigned char *zdr_buff = new unsigned char[sizeof( float )*sitdim.height + 2]; // + 2 for sector id
  zdb_buff[0] = (sector >> 8) & 0xff;
  zdb_buff[1] = (sector) & 0xff;
  zdr_buff[0] = (sector >> 8) & 0xff;
  zdr_buff[1] = (sector) & 0xff;
  aftoab( zdb, sitdim.height, &zdb_buff[2] );
  aftoab( zdr, sitdim.height, &zdr_buff[2] );

  // zdb_client->send((const char *) zdb_buff, sitdim.height*sizeof( float ) + 2 );
  // zdr_client->send((const char *) zdr_buff, sitdim.height*sizeof( float ) + 2 );
}

void do_process(Dimension4 idim, Dimension4 odim, Dimension4 sitdim) {
  cout << "Starting main loop..." << endl;
  read_matrix( idim, current_sector, current_elevation, current_stream );
  copy_matrix_to_device( idim, current_sector, current_elevation, current_stream );
  do {
    perform_stage_1( idim, current_stream );
    perform_stage_2( idim, current_stream );
    perform_stage_3( idim, odim, current_sector, current_elevation, current_stream );
    int
        prev_sector = current_sector,
        prev_elevation = current_elevation,
        prev_stream = current_stream;
    advance( idim );
    read_matrix( idim, current_sector, current_elevation, current_stream );
    copy_matrix_to_device( idim, current_sector, current_elevation, current_stream );
    copy_result_to_host( idim, odim, sitdim, prev_sector, prev_elevation, prev_stream );
    send_results( sitdim, prev_sector, prev_elevation );
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

  int num_streams = 2;
  if (argc > 1) {
    num_streams = atoi( argv[1] );
    num_streams = num_streams < 2 ? 2 : num_streams;
  }

  Dimension4 idim( n_samples, n_sweeps, 3, num_streams );
  Dimension4 odim( 2, n_sweeps/2, 1, num_streams );
  Dimension4 sitdim( 2, n_sweeps/2, n_sectors, n_elevations );

  setup_ports();

  generate_constants( idim, ma_count );
  prepare_arys( idim, odim, sitdim );
  initialize_streams( idim, odim );
  do_process( idim, odim, sitdim );
  destroy_streams();
  destroy_arrays();

  return 0;
}
