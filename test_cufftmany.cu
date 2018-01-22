#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#define DATASIZE 4
#define BATCH 8

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/********/
/* MAIN */
/********/
int main () {
    // --- Host side input data allocation and initialization
    cufftReal *hostInputData = (cufftReal*)malloc(DATASIZE*BATCH*sizeof(cufftReal));
    for (int i=0; i<BATCH; i++)
        for (int j=0; j<DATASIZE; j++) 
            std::cin >> hostInputData[i*DATASIZE + j];

    // --- Device side input data allocation and initialization
    cufftReal *deviceInputData; gpuErrchk(cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftReal)));
    cudaMemcpy(deviceInputData, hostInputData, DATASIZE * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);

    // --- Host side output data allocation
    cufftComplex *hostOutputData = (cufftComplex*)malloc((DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex));

    // --- Device side output data allocation
    cufftComplex *deviceOutputData; gpuErrchk(cudaMalloc((void**)&deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex)));

    // --- Batched 1D FFTs
    cufftHandle handle;
    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
    int batch = BATCH;                      // --- Number of batched executions
    cufftPlanMany(&handle, rank, n, 
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, batch);

    //cufftPlan1d(&handle, DATASIZE, CUFFT_R2C, BATCH);
    cufftExecR2C(handle,  deviceInputData, deviceOutputData);

    // --- Device->Host copy of the results
    gpuErrchk(cudaMemcpy(hostOutputData, deviceOutputData, (DATASIZE / 2 + 1) * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    for (int i=0; i<BATCH; i++)
        for (int j=0; j<(DATASIZE / 2 + 1); j++)
            printf("%i %i %f %f\n", i, j, hostOutputData[i*(DATASIZE / 2 + 1) + j].x, hostOutputData[i*(DATASIZE / 2 + 1) + j].y);

    cufftDestroy(handle);
    gpuErrchk(cudaFree(deviceOutputData));
    gpuErrchk(cudaFree(deviceInputData));
}
