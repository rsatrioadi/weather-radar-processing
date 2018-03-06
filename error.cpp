#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <cmath>

using namespace std;

int main(int argc, char **argv) {

    ifstream cpu,gpu;
    cpu.open("out/cpu.bin", ios::in | ios::binary);
    gpu.open("out/gpu.bin", ios::in | ios::binary);

    float sigdelt = 0.f, sig = 0.f;

    int n = 512;

    for (int i=0; i<n; i++) {
    	float ue, uc;
    	cpu.read((char*) &ue, sizeof(float));
    	gpu.read((char*) &uc, sizeof(float));
    	if(isfinite(ue) && isfinite(uc)) {
	    	sigdelt += (ue-uc)*(ue-uc);
	    	sig += ue*ue;
	    	cout << sigdelt << " " << sig << endl;
	    }
    }

    float l2 = sqrt(sigdelt/sig);

    cout << l2 << endl;

    cpu.close();
    gpu.close();
}
