#include <stdio.h>
#include <complex.h>

#include <cuComplex.h>

#include "cumatrixtools.h"
#include "cuinverse_kernel.cu"

void cuPrintMatrix( cuFloatComplex *C ,int N, int M ) {
    int i,j;
    for(i=0;i<N;i++) {
        for(j=0;j<M;j++) 
            printf(" (%f,%f)\t ", cuCrealf(C[i*N + j]) , cuCimagf(C[i*N + j]) );
        printf(" \n ");
    }
}

///////////////////////////////////////////////////////////////////////////////
const int N=3; //Matrix dimension

const int NBranches = 64;
///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

    int i;
    float complex A[9] = // Base matrix
    {

        0.f+I/sqrtf(2.f), 0.0f+ I/sqrt(2.0f), 0.+ 0.*I,

        0. -I/2. , 0.+I/2 , 0.+ I/sqrt(2.),
        -0.5+0.*I , 0.5+0.*I , -1./sqrt(2.)+.0*I

    };

    cuFloatComplex *h_A, *h_invA;
    cuFloatComplex *d_A, *d_invA, *d_WorkSpace;


    printf("...allocating CPU memory.\n");
    h_A = (cuFloatComplex *) malloc( N*N*sizeof(cuFloatComplex ));
    h_invA = (cuFloatComplex *) malloc( NBranches*N*N*sizeof(cuFloatComplex ));

    printf("...allocating GPU memory.\n");

    cudaMalloc((void **)&d_A, NBranches*N*N*sizeof(cuFloatComplex ));

    cudaMalloc((void **)&d_invA, NBranches*N*N*sizeof(cuFloatComplex ));

    cudaMalloc((void **)&d_WorkSpace, NBranches*cgeMatrixInverse_WorkSpace()*N*N*sizeof(cuFloatComplex ));


    printf("...Copying memory.\n ");
    for(i=0;i<N*N;i++ ) 
        h_A[i] = make_cuFloatComplex( crealf(A[i]) , cimagf(A[i]) );

    cudaMemcpy(d_A, h_A, N*N*sizeof(cuFloatComplex) , cudaMemcpyHostToDevice);


    printf("...The base matrix is:\n");
    cuPrintMatrix( h_A , N, N );


    printf("\n...Calling the kernel.\n");

    cudaThreadSynchronize();
    cgeMatrixInverse_kernel<<<2,32>>>(d_invA, d_A , N ,d_WorkSpace); // Divinding the 64 branches in 2 blocks of 32 threads

    cudaThreadSynchronize();

    cudaMemcpy(h_invA, d_invA, NBranches*N*N*sizeof(float), cudaMemcpyDeviceToHost);


    printf("\n The inverse of the first branch is \n");
    cuPrintMatrix( h_invA , N, N );


    printf("\n The inverse of the second branch is \n");
    cuPrintMatrix( h_invA + N*N , N, N );


    printf("\n and so on ..\n");

    free(h_A);
    free(h_invA);


    cudaFree(d_A);
    cudaFree(d_invA);

    cudaThreadExit();
    printf("\n-------------------------------------------------------\n");
}