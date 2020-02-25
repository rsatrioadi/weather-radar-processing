#include <iostream>
#include <string>
#include <fstream>
#include "udp_client_server.h"
#include "sector.h"

using namespace std;
using namespace udp_client_server;

#define NUM_SWEEPS (512)
#define NUM_SAMPLES (1024)
#define NUM_BYTES_PER_SAMPLE (3*2*2)

int main() {
    udp_server server("0.0.0.0",19001);
    char *buff = new char[NUM_BYTES_PER_SAMPLE*NUM_SAMPLES*NUM_SWEEPS];
    for (int j=0; j<NUM_SWEEPS; j++) {
        server.recv(buff+j*(NUM_BYTES_PER_SAMPLE*NUM_SAMPLES),NUM_BYTES_PER_SAMPLE*NUM_SAMPLES);

    }
    //ofstream("recv000.bin", ios::binary).write(buff, NUM_BYTES_PER_SAMPLE*NUM_SAMPLES*NUM_SWEEPS);
    Sector s(NUM_SWEEPS,NUM_SAMPLES);
    s.fromByteArray(buff);
    s.print();
}
