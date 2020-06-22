#ifndef radarprocessor_h
#define radarprocessor_h

#include <vector>
#include <cuComplex.h>
#include <cufft.h>
#include "udpbroadcast.h"

using namespace std;
using namespace udpbroadcast;

#define NUM_BYTES_PER_SAMPLE (3*2*2)

class RadarProcessor {
  private:

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

    udpserver *server;
    vector<udpclient> clients;

    void prepare_arys();
    void generate_constants();
    void initialize_streams();
    void do_process();
    void destroy_streams();
    void destroy_arrays();

    void read_matrix(int sector, int elevation);
    void copy_matrix_to_device(int sector, int elevation, int stream);
    void perform_stage_1(int stream);
    void perform_stage_2(int stream);
    void perform_stage_3(int stream);
    void advance();
    void copy_result_to_host(int sector, int elevation, int stream);
    void send_results();

    void prepare_host_arys();
    void prepare_device_arys();
    void destroy_device_arys();
    void destroy_host_arys();
    void generate_hamming_coefficients(int m, int n);
    void generate_ma_coefficients(int n);
  public:
    RadarProcessor(int num_sectors, int num_sweeps, int num_samples, int num_elevations, int num_cuda_streams);
    int start();
    void set_comms(int in_port, int *out_ports, int);
    const int
      input_ary_size,
      input_columns,
      input_rows,
      output_ary_size,
      output_columns,
      output_rows;
};

#endif