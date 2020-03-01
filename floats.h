#ifndef sar_floats_h
#define sar_floats_h

#include <stddef.h>

void ftob(float f, unsigned char* buffer);
float btof(unsigned char* buffer);
void aftoab(float* af, size_t numfloats, unsigned char* ab);
void abtoaf(unsigned char* ab, size_t numfloats, float* af);

#endif