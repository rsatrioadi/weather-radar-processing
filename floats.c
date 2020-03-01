#include "floats.h"

void ftob(float f, unsigned char* buffer) {
    buffer[0] = (*(long*)&f & 0xff000000) >> 24;
    buffer[1] = (*(long*)&f & 0x00ff0000) >> 16;
    buffer[2] = (*(long*)&f & 0x0000ff00) >> 8;
    buffer[3] = (*(long*)&f & 0x000000ff);
    // cout << f << " --> ";
    // for (int i=0; i<4; i++) {
    //     cout << (short)buffer[i] << " ";
    // }
    // cout << endl;
}

float btof(unsigned char* buffer) {
    long val = 0;
    val |= ((long)buffer[0]) << 24;
    val |= ((long)buffer[1]) << 16;
    val |= ((long)buffer[2]) << 8;
    val |= ((long)buffer[3]);

    float f = *(float*)&val;

    // for (int i=0; i<4; i++) {
    //     cout << (short)buffer[i] << " ";
    // }
    // cout << "--> " << f << endl;

    return f;
}

void aftoab(float* af, size_t numfloats, unsigned char* ab) {
    for(int i=0; i<numfloats; i++) {
        ftob(af[i], &ab[i*4]);
    }
}

void abtoaf(unsigned char* ab, size_t numfloats, float* af) {
    for(int i=0; i<numfloats; i++) {
        af[i] = btof(&ab[i*4]);
    }
}
