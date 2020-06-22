#include "radar_processor.h"

int main(int argc, char** argv) {
  ios_base::sync_with_stdio( false );
  int num_streams = 1;
  if ( argc > 1 ){
    num_streams = atoi( argv[1] );
    num_streams = num_streams<1 ? 1 : num_streams;
  }
  RadarProcessor proc( 143, 1024, 512, 9, num_streams );
  int* out_ports = new int[2];
  out_ports[0] = 19002;
  out_ports[1]= 19003;
  proc.set_ports( 19001, out_ports );
  proc.start();
}