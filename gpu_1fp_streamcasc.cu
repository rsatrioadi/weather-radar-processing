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
// #include <fstream>

using namespace std;
using namespace udpbroadcast;

#define k_rangeres 30
#define k_calib 1941.05

#define RESULT_SIZE 2

#define NUM_SECTORS 143
#define NUM_ELEVATIONS 9

#define NUM_BYTES_PER_SAMPLE (3*2*2)

//#define NSTREAMS 16

#define DEBUG

inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf( stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString( result ));
    assert( result == cudaSuccess );
  }
#endif
  return result;
}

float *generate_hamming_coef(int m, int n) {

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
  const float K_wind = -1 / (16383.5 * m * n * sqrt( 50 ));
  const float c = K_wind / sqrt( p_range * p_doppler );

  // Generate elements
  float *_hamming_coef = new float[m * n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      _hamming_coef[i * n + j] =
          (0.53836 - 0.46164 * cos( 2 * M_PI * (i) / (m - 1))) * (0.53836 - 0.46164 * cos( 2 * M_PI * (j) / (n - 1))) *
          c;
    }
  }

  return _hamming_coef;
}

float *generate_ma_coef(int n) {
  float *_ma_coef = new float[n];
  float _sum = 0.0;
  for (int i = 0; i < n; i++) {
    _ma_coef[i] = exp( -(pow( i - ((n - 1) / 2), 2.0 )) / 2 );
    _sum += _ma_coef[i];
  }
  for (int i = 0; i < n; i++) {
    _ma_coef[i] = _ma_coef[i] / _sum;
  }
  return _ma_coef;
}

__constant__ cuFloatComplex d_ma[512];

__global__ void __apply_hamming(cuFloatComplex *a, float *b, int offset) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[offset + idx] = make_cuFloatComplex( b[idx] * cuCrealf( a[offset + idx] ), b[idx] * cuCimagf( a[offset + idx] ));
}

__global__ void __apply_hamming_v2(cuFloatComplex *a, float *b, int offset) {
  const unsigned int i = 4 * blockIdx.x, j = threadIdx.x, n = blockDim.x;
#pragma unroll
  for (unsigned int d = 0; d < 4; d++) {
    a[offset + i * n + j + n * d] = make_cuFloatComplex(
        b[i * n + j + n * d] * cuCrealf( a[offset + i * n + j + n * d] ),
        b[i * n + j + n * d] * cuCimagf( a[offset + i * n + j + n * d] ));
  }
}

__global__ void __apply_hamming_v3(cuFloatComplex *a, float *b, int offset) {
  const unsigned int i = blockIdx.x, j = 4 * threadIdx.x, n = 4 * blockDim.x;
#pragma unroll
  for (unsigned int d = 0; d < 4; d++) {
    a[offset + i * n + j + d] = make_cuFloatComplex( b[i * n + j + d] * cuCrealf( a[offset + i * n + j + d] ),
                                                     b[i * n + j + d] * cuCimagf( a[offset + i * n + j + d] ));
  }
}

__global__ void __sum(cuFloatComplex *in, cuFloatComplex *out, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  out[offset + i * n + j] = make_cuFloatComplex( in[offset + i * n + j].x, in[offset + i * n + j].y );
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
      out[offset + i * n + j] = cuCaddf( out[offset + i * n + j], out[offset + i * n + j + s] );
    }
    __syncthreads();
  }
}

__global__ void __sum_v2(cuFloatComplex *in, cuFloatComplex *out, int offset) {
  const unsigned int i = 2 * blockIdx.x, j = threadIdx.x, n = blockDim.x;

#pragma unroll
  for (unsigned int d = 0; d < 2; d++) {
    out[offset + i * n + j + n * d] = make_cuFloatComplex( in[offset + i * n + j + n * d].x,
                                                           in[offset + i * n + j + n * d].y );
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
#pragma unroll
      for (unsigned int d = 0; d < 2; d++) {
        out[offset + i * n + j + n * d] = cuCaddf( out[offset + i * n + j + n * d],
                                                   out[offset + i * n + j + n * d + s] );
      }
    }
    __syncthreads();
  }
}

__global__ void __sum_v3(cuFloatComplex *in, cuFloatComplex *out, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;
  extern __shared__ cuFloatComplex sdata[];

  sdata[j] = make_cuFloatComplex( in[offset + i * n + j].x, in[offset + i * n + j].y );
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
      // out[offset+i*n+j] = cuCaddf(out[offset+i*n+j], out[offset+i*n+j+s]);
      sdata[j] = cuCaddf( sdata[j], sdata[j + s] );
    }
    __syncthreads();
  }

  if (j == 0) {
    out[i * n] = sdata[j];
  }
}

__global__ void __sum_v4(cuFloatComplex *in, cuFloatComplex *out, int offset) {
  const unsigned int i = 2 * blockIdx.x, j = threadIdx.x, n = blockDim.x;
  extern __shared__ cuFloatComplex sdata[];

#pragma unroll
  for (unsigned int d = 0; d < 2; d++) {
    sdata[j + n * d] = make_cuFloatComplex( in[offset + i * n + j + n * d].x, in[offset + i * n + j + n * d].y );
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
#pragma unroll
      for (unsigned int d = 0; d < 2; d++) {
        sdata[j + n * d] = cuCaddf( sdata[j + n * d], sdata[j + n * d + s] );
      }
    }
    __syncthreads();
  }

  if (j == 0) {
#pragma unroll
    for (unsigned int d = 0; d < 2; d++) {
      out[i * n + n * d] = sdata[j + n * d];
    }
  }
}

__global__ void __avgconj(cuFloatComplex *inout, cuFloatComplex *sum, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  float avgx = sum[offset + i * n].x / n;
  float avgy = sum[offset + i * n].y / n;
  inout[offset + i * n + j] = make_cuFloatComplex( inout[offset + i * n + j].x - avgx,
                                                   (inout[offset + i * n + j].y - avgy) * -1 );
}

__global__ void __conjugate(cuFloatComplex *a, int offset) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[offset + idx].y *= -1;
}

__global__ void __shift(cuFloatComplex *inout, int n, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x;

  cuFloatComplex temp = inout[offset + i * n + j];
  inout[offset + i * n + j] = inout[offset + i * n + (j + n / 2)];
  inout[offset + i * n + (j + n / 2)] = temp;
}

__global__ void __clip(cuFloatComplex *inout, int n, int offset) {
  const unsigned int i = blockIdx.x, j = n - threadIdx.x - 1;
  inout[offset + i * n + j] = make_cuFloatComplex( 0, 0 );
}

__global__ void __clip_v2(cuFloatComplex *inout, int n, int offset) {
  const unsigned int i = threadIdx.x, j = n - blockIdx.x - 1;
  inout[offset + i * n + j] = make_cuFloatComplex( 0, 0 );
}

__global__ void __abssqr(cuFloatComplex *inout, int n, int offset) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float real, imag;
  real = cuCrealf( inout[offset + idx] );
  imag = cuCimagf( inout[offset + idx] );
  inout[offset + idx] = make_cuFloatComplex( real * real + imag * imag, 0 );
}

__global__ void __apply_ma(cuFloatComplex *inout, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  inout[offset + i * n + j] = cuCmulf( inout[offset + i * n + j], d_ma[j] );
}

__global__ void __scale_real(cuFloatComplex *inout, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  inout[offset + i * n + j] = make_cuFloatComplex( inout[offset + i * n + j].x / n, 0 );
}

__global__ void __sum_inplace(cuFloatComplex *g_idata, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;

  // __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
      // g_idata[i] = make_cuFloatComplex(g_idata[i].x+g_idata[i + s].x, 0);
      g_idata[offset + i * n + j] = cuCaddf( g_idata[offset + i * n + j], g_idata[offset + i * n + j + s] );
    }
    __syncthreads();
  }
}

__global__ void __sum_inplace_v2(cuFloatComplex *g_idata, int offset) {
  const unsigned int i = 2 * blockIdx.x, j = threadIdx.x, n = blockDim.x;

  // __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
      // g_idata[i] = make_cuFloatComplex(g_idata[i].x+g_idata[i + s].x, 0);
#pragma unroll
      for (unsigned int d = 0; d < 2; d++) {
        g_idata[offset + i * n + j + n * d] = cuCaddf( g_idata[offset + i * n + j + n * d],
                                                       g_idata[offset + i * n + j + n * d + s] );
      }
    }
    __syncthreads();
  }
}

__global__ void __sum_inplace_v3(cuFloatComplex *in, int offset) {
  const unsigned int i = blockIdx.x, j = threadIdx.x, n = blockDim.x;
  extern __shared__ cuFloatComplex sdata[];

  sdata[j] = make_cuFloatComplex( in[offset + i * n + j].x, in[offset + i * n + j].y );
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
      sdata[j] = cuCaddf( sdata[j], sdata[j + s] );
    }
    __syncthreads();
  }

  if (j == 0) {
    in[offset + (i * n)] = sdata[j];
  }
}

__global__ void __sum_inplace_v4(cuFloatComplex *in, int offset) {
  const unsigned int i = 2 * blockIdx.x, j = threadIdx.x, n = blockDim.x;
  extern __shared__ cuFloatComplex sdata[];

#pragma unroll
  for (unsigned int d = 0; d < 2; d++) {
    sdata[j + n * d] = make_cuFloatComplex( in[offset + i * n + j + n * d].x, in[offset + i * n + j + n * d].y );
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (j < s) {
#pragma unroll
      for (unsigned int d = 0; d < 2; d++) {
        sdata[j + n * d] = cuCaddf( sdata[j + n * d], sdata[j + n * d + s] );
      }
    }
    __syncthreads();
  }

  if (j == 0) {
#pragma unroll
    for (unsigned int d = 0; d < 2; d++) {
      in[offset + (i * n + n * d)] = sdata[j + n * d];
    }
  }
}

__global__ void
__calcresult_v2(cuFloatComplex *hh, cuFloatComplex *vv, cuFloatComplex *vh, float *out, int n, int offset,
                int result_offset) {
  const unsigned int i = threadIdx.x;

  float z = pow( i * k_rangeres, 2.0 ) * k_calib * hh[offset + i * n].x;
  float zdb = 10 * log10( z );
  float zdr = 10 * (log10( hh[offset + i * n].x ) - log10( vv[offset + i * n].x ));
  out[result_offset + i * RESULT_SIZE + 0] = zdb;
  out[result_offset + i * RESULT_SIZE + 1] = zdr;
}

void tick(timeval *begin) {
  gettimeofday( begin, NULL );
}

void tock(timeval *begin, timeval *end, string caption) {
  unsigned long long bb, e;

  gettimeofday( end, NULL );
  bb = (unsigned long long) (begin->tv_sec) * 1000000 + (unsigned long long) (begin->tv_usec) / 1;
  e = (unsigned long long) (end->tv_sec) * 1000000 + (unsigned long long) (end->tv_usec) / 1;

  cout << caption << ": " << e - bb << endl;
}

int main(int argc, char **argv) {
  ios_base::sync_with_stdio( false );

  // // tes serialize floats to unsigned chars:
  // float floats[3] = {1.2, 3.4, -5.6e7};
  // unsigned char* bytes = new unsigned char[3*4];
  // aftoab(floats,3,bytes);
  // float floats2[3];
  // abtoaf(bytes,3,floats2);
  // for (int i=0; i<3; i++) {
  //     cout << floats2[i] << " ";
  // }
  // cout << endl;
  // exit(0);


  int NSTREAMS = atoi( argv[1] );
  if (NSTREAMS < 1) {
    NSTREAMS = 1;
  }

  struct timeval tb, te;

  tick( &tb );

  cuFloatComplex *iqhh, *iqvv, *iqvh;
  cuFloatComplex *p_iqhh, *p_iqvv, *p_iqvh;
  float *result;
  int sector_id;
  int elev = 0;

  const int m = 1024; // NUM_SWEEPS
  const int n = 512;  // NUM_SAMPLES

  const int ma_count = 7;

  iqhh = new cuFloatComplex[m * n];
  iqvv = new cuFloatComplex[m * n];
  iqvh = new cuFloatComplex[m * n];
  cudaMallocHost((void **) &p_iqhh, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  cudaMallocHost((void **) &p_iqvv, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  cudaMallocHost((void **) &p_iqvh, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  result = new float[NSTREAMS * (m / 2) * RESULT_SIZE * NUM_ELEVATIONS];

  // Generate Hamming coefficients
  const float *hamming_coef = generate_hamming_coef( m, n );

  // Generate MA coefficients
  float *ma_coef = generate_ma_coef( ma_count );
  fftwf_complex *_fft_ma = (fftwf_complex *) fftwf_malloc( sizeof( fftwf_complex ) * n );
  fftwf_plan fft_ma_plan = fftwf_plan_dft_1d( n, _fft_ma, _fft_ma, FFTW_FORWARD, FFTW_ESTIMATE );
  for (int j = 0; j < ma_count; j++) {
    _fft_ma[j][0] = ma_coef[j];
    _fft_ma[j][1] = 0;
  }
  for (int j = ma_count; j < n; j++) {
    _fft_ma[j][0] = 0;
    _fft_ma[j][1] = 0;
  }
  fftwf_execute( fft_ma_plan );
  fftwf_destroy_plan( fft_ma_plan );
  cuFloatComplex *fft_ma;
  fft_ma = new cuFloatComplex[n];
  for (int j = 0; j < n; j++) {
    fft_ma[j] = make_cuFloatComplex( _fft_ma[j][0], _fft_ma[j][1] );
  }
  fftwf_free( _fft_ma );

  // Device buffers
  /*__constant__*/ float *d_hamming;
  cuFloatComplex *d_iqhh, *d_iqvv, *d_iqvh;
  cuFloatComplex *d_sum;
  float *d_result;
  //float *d_powhh, *d_powvv;

  cudaMalloc( &d_hamming, m * n * sizeof( float ));
  // cudaMalloc(&d_ma, n*sizeof(cuFloatComplex));
  cudaMalloc( &d_iqhh, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  cudaMalloc( &d_iqvv, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  cudaMalloc( &d_iqvh, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  cudaMalloc( &d_sum, NSTREAMS * m * n * sizeof( cuFloatComplex ));
  cudaMalloc( &d_result, NSTREAMS * (m / 2) * RESULT_SIZE * NUM_ELEVATIONS * sizeof( float ));

  cudaMemcpy( d_hamming, hamming_coef, m * n * sizeof( float ), cudaMemcpyHostToDevice );
  cudaMemcpyToSymbol( d_ma, fft_ma, n * sizeof( cuFloatComplex ), 0, cudaMemcpyHostToDevice );

  // CUFFT initialization
  cufftHandle *fft_range_handle = new cufftHandle[NSTREAMS];
  cufftHandle *fft_doppler_handle = new cufftHandle[NSTREAMS];
  cufftHandle *fft_pdop_handle = new cufftHandle[NSTREAMS];

  int rank = 1;                   // --- 1D FFTs
  int nn[] = { m };               // --- Size of the Fourier transform
  int istride = n, ostride = n;   // --- Distance between two successive input/output elements
  int idist = 1, odist = 1;       // --- Distance between batches
  int inembed[] = { 0 };          // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };          // --- Output size with pitch (ignored for 1D transforms)
  int batch = n;                  // --- Number of batched executions

  cudaStream_t stream[NSTREAMS];
  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamCreate( &stream[i] );

    cufftPlanMany( &fft_range_handle[i], rank, nn,
                   inembed, istride, idist,
                   onembed, ostride, odist, CUFFT_C2C, batch );
    cufftPlan1d( &fft_doppler_handle[i], n, CUFFT_C2C, m );
    cufftPlan1d( &fft_pdop_handle[i], n, CUFFT_C2C, m / 2 );

    cufftSetStream( fft_range_handle[i], stream[i] );
    cufftSetStream( fft_doppler_handle[i], stream[i] );
    cufftSetStream( fft_pdop_handle[i], stream[i] );
  }

  tock( &tb, &te, "initialization" );
  float ms; // elapsed time in milliseconds

  // create events
  cudaEvent_t startEvent, stopEvent;

  cudaEventCreate( &startEvent );
  cudaEventCreate( &stopEvent );
  // cudaEventCreate(&dummyEvent);

  cudaEventRecord( startEvent, 0 );
  tick( &tb );

  // ofstream myFile;
  // myFile.open("out/gpu.bin", ios::out | ios::binary);
  sector_id = 0;

  udpserver server( 19001 ); // receive raw data
  udpclient zdbClient( 19002 ); // send zdb result
  udpclient zdrClient( 19003 ); // send zdr result


  // cout << "siap terima " << sector_id << endl;
  char *buff = new char[NUM_BYTES_PER_SAMPLE * n * m];
  for (int j = 0; j < m; j++) {
    server.recv( buff + j * (NUM_BYTES_PER_SAMPLE * n), NUM_BYTES_PER_SAMPLE * n );
  }
  //cout << "done!" << endl;
  cout << sector_id << " received." << endl;
  Sector s( m, n );
  s.fromByteArray( buff );
  delete[] buff;

  int a, b;

  // cout << "bikin matriks" << endl;
  int idx = 0;
#pragma unroll
  for (int i = 0; i < m; i++) {
#pragma unroll
    for (int j = 0; j < n; j++) {
      // cin >> a >> b;
      a = idx++;
      b = idx++;
      p_iqhh[i * n + j] = make_cuFloatComplex( s.hh[a], s.hh[b] );
      p_iqvv[i * n + j] = make_cuFloatComplex( s.vv[a], s.vv[b] );
      p_iqvh[i * n + j] = make_cuFloatComplex( s.vh[a], s.vh[b] );
    }
  }

  // cout << "copy ke device" << endl;
  // memcpy(&p_iqhh[0], iqhh, m*n*sizeof(cuFloatComplex));
  // memcpy(&p_iqvv[0], iqvv, m*n*sizeof(cuFloatComplex));
  // memcpy(&p_iqvh[0], iqvh, m*n*sizeof(cuFloatComplex));

  cudaMemcpyAsync( &d_iqhh[0], &p_iqhh[0], m * n * sizeof( cuFloatComplex ), cudaMemcpyHostToDevice, stream[0] );
  cudaMemcpyAsync( &d_iqvv[0], &p_iqvv[0], m * n * sizeof( cuFloatComplex ), cudaMemcpyHostToDevice, stream[0] );
  cudaMemcpyAsync( &d_iqvh[0], &p_iqvh[0], m * n * sizeof( cuFloatComplex ), cudaMemcpyHostToDevice, stream[0] );

  do {
    // for(int stream_id=0; stream_id < NSTREAMS; stream_id++) {

    // tick(&tb);

    // Read 1 sector data
    // cin >> sector_id;

    int stream_id = sector_id % NSTREAMS;
    int offset = stream_id * (m * n);
    int result_offset = stream_id * (m / 2) * RESULT_SIZE;


    // cout << "stage I" << endl;

    // apply Hamming coefficients
    __apply_hamming<<<m, n, 0, stream[stream_id]>>>( d_iqhh, d_hamming, offset );
    __apply_hamming<<<m, n, 0, stream[stream_id]>>>( d_iqvv, d_hamming, offset );
    __apply_hamming<<<m, n, 0, stream[stream_id]>>>( d_iqvh, d_hamming, offset );
    // exit(0);

    // FFT range profile
    cufftExecC2C( fft_range_handle[stream_id], &d_iqhh[offset], &d_iqhh[offset], CUFFT_FORWARD );
    cufftExecC2C( fft_range_handle[stream_id], &d_iqvv[offset], &d_iqvv[offset], CUFFT_FORWARD );
    cufftExecC2C( fft_range_handle[stream_id], &d_iqvh[offset], &d_iqvh[offset], CUFFT_FORWARD );

    // FFT+shift Doppler profile
    __sum_v4<<<m/2, n, 2*n*sizeof(cuFloatComplex), stream[stream_id]>>>( d_iqhh, d_sum, offset );
    __avgconj<<<m, n, 0, stream[stream_id]>>>( d_iqhh, d_sum, offset );
    __sum_v4<<<m/2, n, 2*n*sizeof(cuFloatComplex), stream[stream_id]>>>( d_iqvv, d_sum, offset );
    __avgconj<<<m, n, 0, stream[stream_id]>>>( d_iqvv, d_sum, offset );
    __sum_v4<<<m/2, n, 2*n*sizeof(cuFloatComplex), stream[stream_id]>>>( d_iqvh, d_sum, offset );
    __avgconj<<<m, n, 0, stream[stream_id]>>>( d_iqvh, d_sum, offset );

    cufftExecC2C( fft_doppler_handle[stream_id], &d_iqhh[offset], &d_iqhh[offset], CUFFT_FORWARD );
    cufftExecC2C( fft_doppler_handle[stream_id], &d_iqvv[offset], &d_iqvv[offset], CUFFT_FORWARD );
    cufftExecC2C( fft_doppler_handle[stream_id], &d_iqvh[offset], &d_iqvh[offset], CUFFT_FORWARD );

    __conjugate<<<m, n, 0, stream[stream_id]>>>( d_iqhh, offset );
    __conjugate<<<m, n, 0, stream[stream_id]>>>( d_iqvv, offset );
    __conjugate<<<m, n, 0, stream[stream_id]>>>( d_iqvh, offset );

    __shift<<<m, n/2, 0, stream[stream_id]>>>( d_iqhh, n, offset );
    __shift<<<m, n/2, 0, stream[stream_id]>>>( d_iqvv, n, offset );
    __shift<<<m, n/2, 0, stream[stream_id]>>>( d_iqvh, n, offset );

    __clip_v2<<<2, m, 0, stream[stream_id]>>>( d_iqhh, n, offset );
    __clip_v2<<<2, m, 0, stream[stream_id]>>>( d_iqvv, n, offset );
    __clip_v2<<<2, m, 0, stream[stream_id]>>>( d_iqvh, n, offset );

    // cout << "stage II" << endl;
    // Get absolute value squared
    __abssqr<<<m/2, n, 0, stream[stream_id]>>>( d_iqhh, n, offset );
    __abssqr<<<m/2, n, 0, stream[stream_id]>>>( d_iqvv, n, offset );
    __abssqr<<<m/2, n, 0, stream[stream_id]>>>( d_iqvh, n, offset );

    // FFT PDOP
    cufftExecC2C( fft_pdop_handle[stream_id], &d_iqhh[offset], &d_iqhh[offset], CUFFT_FORWARD );
    cufftExecC2C( fft_pdop_handle[stream_id], &d_iqvv[offset], &d_iqvv[offset], CUFFT_FORWARD );
    cufftExecC2C( fft_pdop_handle[stream_id], &d_iqvh[offset], &d_iqvh[offset], CUFFT_FORWARD );

    // Apply MA coefficients
    __apply_ma<<<m/2, n, 0, stream[stream_id]>>>( d_iqhh, offset );
    __apply_ma<<<m/2, n, 0, stream[stream_id]>>>( d_iqvv, offset );
    __apply_ma<<<m/2, n, 0, stream[stream_id]>>>( d_iqvh, offset );

    // Inverse FFT
    cufftExecC2C( fft_pdop_handle[stream_id], &d_iqhh[offset], &d_iqhh[offset], CUFFT_INVERSE );
    cufftExecC2C( fft_pdop_handle[stream_id], &d_iqvv[offset], &d_iqvv[offset], CUFFT_INVERSE );
    cufftExecC2C( fft_pdop_handle[stream_id], &d_iqvh[offset], &d_iqvh[offset], CUFFT_INVERSE );

    __scale_real<<<m/2, n, 0, stream[stream_id]>>>( d_iqhh, offset );
    __scale_real<<<m/2, n, 0, stream[stream_id]>>>( d_iqvv, offset );
    __scale_real<<<m/2, n, 0, stream[stream_id]>>>( d_iqvh, offset );

    // Sum
    // __sum_inplace_v2<<<m/4,n,0,stream[stream_id]>>>(d_iqhh, offset);
    // __sum_inplace_v2<<<m/4,n,0,stream[stream_id]>>>(d_iqvv, offset);
    __sum_inplace_v4<<<m/4, n, 2*n*sizeof(cuFloatComplex), stream[stream_id]>>>( d_iqhh, offset );
    __sum_inplace_v4<<<m/4, n, 2*n*sizeof(cuFloatComplex), stream[stream_id]>>>( d_iqvv, offset );
    __sum_inplace_v4<<<m/4, n, 2*n*sizeof(cuFloatComplex), stream[stream_id]>>>( d_iqvh, offset );

//    cudaMemcpy( iqhh, &d_iqhh[offset], m * n * sizeof( cuFloatComplex ), cudaMemcpyDeviceToHost );
//    cudaMemcpy( iqvv, &d_iqvv[offset], m * n * sizeof( cuFloatComplex ), cudaMemcpyDeviceToHost );
//    cudaMemcpy( iqvh, &d_iqvh[offset], m * n * sizeof( cuFloatComplex ), cudaMemcpyDeviceToHost );
//
//    for (int i = 0; i < m/2; i++) {
//      for (int j = 0; j < n; j++) {
//        cout << "(" << iqhh[i * n + j].x << "," << iqhh[i * n + j].y << ") ";
//      }
//      cout << endl;
//    }
//    // for (int i=0; i<m; i++) {
//    //     for (int j=0; j<n; j++) {
//    //         cout << iqvv[i*n+j].x << " ";
//    //     }
//    //     cout << endl;
//    // }
//    exit( 0 );

    // cudaMemcpy(iqhh, d_iqhh, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    // cudaMemcpy(iqvv, d_iqvv, m*n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // for (int i=0; i<m/2; i++) {
    //     float z = pow(i*k_rangeres, 2.0) * k_calib * iqhh[i*n].x;
    //     float zdb = 10 * log10(z);
    //     float zdr = 10 * (log10(iqhh[i*n].x)-log10(iqvv[i*n].x));
    //     cout << zdb << " " << zdr << endl;
    // }
    // exit(0);


    // cout << "stage III" << endl;
    // Calculate ZdB, Zdr
    __calcresult_v2<<<1, m/2, 0, stream[stream_id]>>>( d_iqhh, d_iqvv, d_iqvh, d_result, n, offset, result_offset );

    int next_stream_id = (sector_id + 1) % NSTREAMS;
    int next_offset = next_stream_id * (m * n);

    int oldsector = sector_id;
    int oldelev = elev;

    sector_id = (sector_id + 1) % NUM_SECTORS;
    if (sector_id == 0) {
      elev = (elev + 1) % NUM_ELEVATIONS;
    }
    // cout << "siap terima " << sector_id << endl;
    char *buff = new char[NUM_BYTES_PER_SAMPLE * n * m];
    for (int j = 0; j < m; j++) {
      server.recv( buff + j * (NUM_BYTES_PER_SAMPLE * n), NUM_BYTES_PER_SAMPLE * n );
    }
    cout << sector_id << " received." << endl;
    s.fromByteArray( buff );
    delete[] buff;

    idx = 0;
#pragma unroll
    for (int i = 0; i < m; i++) {
#pragma unroll
      for (int j = 0; j < n; j++) {
        // cin >> a >> b;
        a = idx++;
        b = idx++;
        p_iqhh[next_offset + i * n + j] = make_cuFloatComplex( s.hh[a], s.hh[b] );
        p_iqvv[next_offset + i * n + j] = make_cuFloatComplex( s.vv[a], s.vv[b] );
        p_iqvh[next_offset + i * n + j] = make_cuFloatComplex( s.vh[a], s.vh[b] );
      }
    }

    // memcpy(&p_iqhh[next_offset], iqhh, m*n*sizeof(cuFloatComplex));
    // memcpy(&p_iqvv[next_offset], iqvv, m*n*sizeof(cuFloatComplex));
    // memcpy(&p_iqvh[next_offset], iqvh, m*n*sizeof(cuFloatComplex));

    cudaMemcpyAsync( &d_iqhh[next_offset], &p_iqhh[next_offset], m * n * sizeof( cuFloatComplex ),
                     cudaMemcpyHostToDevice, stream[next_stream_id] );
    cudaMemcpyAsync( &d_iqvv[next_offset], &p_iqvv[next_offset], m * n * sizeof( cuFloatComplex ),
                     cudaMemcpyHostToDevice, stream[next_stream_id] );
    cudaMemcpyAsync( &d_iqvh[next_offset], &p_iqvh[next_offset], m * n * sizeof( cuFloatComplex ),
                     cudaMemcpyHostToDevice, stream[next_stream_id] );

    cudaMemcpyAsync( &result[result_offset + oldelev * (RESULT_SIZE * (m / 2))], &d_result[result_offset],
                     (m / 2) * RESULT_SIZE * sizeof( float ), cudaMemcpyDeviceToHost, stream[stream_id] );

    float *zdb = new float[m / 2];
    float *zdr = new float[m / 2];

//    cout << "zdb: " << endl;
    idx = 0;
    for (int i = 0; i < m / 2; i++) {
      zdb[i] = result[result_offset + (idx++)];
      zdr[i] = result[result_offset + (idx++)];
//      cout << zdb[i] << endl;
    }
//    exit(0);
    // cout << endl;

    // cout << "zdb: ";
    // for (int i=0; i<m/2; i++) {
    //     cout << zdb[i] << " ";
    // }
    // cout << endl;

    unsigned char *zdbBuff = new unsigned char[sizeof( float ) * (m / 2) + 2];
    unsigned char *zdrBuff = new unsigned char[sizeof( float ) * (m / 2) + 2];
    zdbBuff[0] = (oldsector >> 8) & 0xff;
    zdbBuff[1] = (oldsector) & 0xff;
    zdrBuff[0] = (oldsector >> 8) & 0xff;
    zdrBuff[1] = (oldsector) & 0xff;
    aftoab( zdb, (m / 2), &zdbBuff[2] );
    aftoab( zdr, (m / 2), &zdrBuff[2] );

    // cout << "in bytes: ";
    // for (int i=0; i<2+m/2; i++) {
    //     cout << (int)zdbBuff[i] << " ";
    // }
    // cout << endl;

    zdbClient.send((const char *) zdbBuff, (m / 2) * sizeof( float ) + 2 );
    zdrClient.send((const char *) zdrBuff, (m / 2) * sizeof( float ) + 2 );

    // tock(&tb, &te, "time");

    // for (int i=0; i<m/2; i++) {
    //     myFile.write((char*)&result[result_offset+i*RESULT_SIZE+0], sizeof(float));
    // }

    // exit(0);


    // }
  } while (sector_id < (NUM_SECTORS * 3) - 1);

  // myFile.close();

  tock( &tb, &te, "All (us)" );

  cudaEventRecord( stopEvent, 0 );
  cudaEventSynchronize( stopEvent );
  cudaEventElapsedTime( &ms, startEvent, stopEvent );
  printf( "Time for async transfer and execute (ms): %f\n", ms );

  cudaEventDestroy( startEvent );
  cudaEventDestroy( stopEvent );


  delete[] fft_range_handle;
  delete[] fft_doppler_handle;
  delete[] fft_pdop_handle;

  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamDestroy( stream[i] );
  }

  cudaFree( d_hamming );
  // cudaFree(d_ma);
  cudaFree( d_iqhh );
  cudaFree( d_iqvv );
  cudaFree( d_iqvh );

  cudaFreeHost( p_iqhh );
  cudaFreeHost( p_iqvv );
  cudaFreeHost( p_iqvh );

  // delete[] iqhh;
  // delete[] iqvv;
  // delete[] iqvh;

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
