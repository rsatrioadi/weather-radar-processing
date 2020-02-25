#include <fstream>

using namespace std;

#ifndef __SECTOR_H__
#define __SECTOR_H__
class Sector {
    public:
        int sweeps, samples;
        short *hh, *vv, *vh;

        Sector(int num_sweeps, int num_samples);
        ~Sector();
        
        void read(istream &in);
        void fromByteArray(char *buff);
        void print() const;
};
#endif
