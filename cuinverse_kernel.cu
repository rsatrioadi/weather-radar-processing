__global__ void cgeMatrixInverse_kernel(cuFloatComplex *invA , cuFloatComplex *A , int N , cuFloatComplex *Work) {

    int i;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    cuFloatComplex * ThreadWorkSpace;

    ThreadWorkSpace = Work + idx*cgeMatrixInverse_WorkSpace()*N*N;

    for(i=0; i<N*N; i++) 
        A[ i + idx*N*N ] =  A[i];

    A[ idx*N*N ] = make_cuFloatComplex( (float) idx , 1./sqrtf(2));

    cgeMatrixInverse(invA + idx*N*N , A + idx*N*N , N , ThreadWorkSpace);
}