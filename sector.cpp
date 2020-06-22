#include <iostream>
#include <string>
#include <fstream>

#include "sector.h"

using namespace std;

Sector::Sector(int num_sweeps, int num_samples):
        sweeps(num_sweeps), samples(num_samples) {
    hh = new short[2*sweeps*samples];
    vv = new short[2*sweeps*samples];
    vh = new short[2*sweeps*samples];
}

Sector::~Sector() {
    delete [] hh;
    delete [] vv;
    delete [] vh;
}

void Sector::read(istream &in) {
    int len = 3*2;
    unsigned char hi, lo;
    int i=0;
    while (in.good()) {
        in >> hi;
        in >> lo;
        hh[i*2]   = ((hi<<8)&0xff00) + (lo&0xff);
        in >> hi;
        in >> lo;
        hh[i*2+1] = ((hi<<8)&0xff00) + (lo&0xff);
        in >> hi;
        in >> lo;
        vv[i*2]   = ((hi<<8)&0xff00) + (lo&0xff);
        in >> hi;
        in >> lo;
        vv[i*2+1] = ((hi<<8)&0xff00) + (lo&0xff);
        in >> hi;
        in >> lo;
        vh[i*2]   = ((hi<<8)&0xff00) + (lo&0xff);
        in >> hi;
        in >> lo;
        vh[i*2+1] = ((hi<<8)&0xff00) + (lo&0xff);
        i++;
    }
    // for (int i=0; i<2*len*sweeps*samples; i++) {
    //     cout << (int)tmp[i] << " ";
    // }
}

void Sector::fromByteArray(char *buff) {
    int idx = 0;
    for (int i=0; i<sweeps*samples; i++) {
        hh[i*2]   = ((((unsigned char)buff[idx++])<<8)&0xff00) + (((unsigned char)buff[idx++])&0xff);
        hh[i*2+1] = ((((unsigned char)buff[idx++])<<8)&0xff00) + (((unsigned char)buff[idx++])&0xff);
        vv[i*2]   = ((((unsigned char)buff[idx++])<<8)&0xff00) + (((unsigned char)buff[idx++])&0xff);
        vv[i*2+1] = ((((unsigned char)buff[idx++])<<8)&0xff00) + (((unsigned char)buff[idx++])&0xff);
        vh[i*2]   = ((((unsigned char)buff[idx++])<<8)&0xff00) + (((unsigned char)buff[idx++])&0xff);
        vh[i*2+1] = ((((unsigned char)buff[idx++])<<8)&0xff00) + (((unsigned char)buff[idx++])&0xff);
    }
}

void Sector::print() const {
    cout << "hh:" << endl;
    for (int i=0; i<sweeps*samples; i++) {
        cout << hh[i*2] << " " << hh[i*2+1] << " ";
    }
    cout << endl << "vv:" << endl;
    for (int i=0; i<sweeps*samples; i++) {
        cout << vv[i*2] << " " << vv[i*2+1] << " ";
    }
    cout << endl << "vh:" << endl;
    for (int i=0; i<sweeps*samples; i++) {
        cout << vh[i*2] << " " << vh[i*2+1] << " ";
    }
    cout << endl;
}

// int main() {
//     Sector s000(512,1024);
//     ifstream ifs("sector000.bin", ifstream::in);
//     s000.read(ifs);
//     s000.print();
// }
