#include <cuComplex.h>
#ifndef _CUMATRIXTOOLS_H_
#define _CUMATRIXTOOLS_H_


////////////////////////////////////////////////////////////////////
__device__ __host__ static __inline__ void
cgeSquareMatrixProduct(cuFloatComplex *out, cuFloatComplex *A, cuFloatComplex *B, int N  )
{
    int i,j,k;
    cuFloatComplex sum;

    for(i=0;i<N;i++)
        for(j=0;j<N;j++){
            //sum = 0. + 0.*I;
            sum  = make_cuFloatComplex(0.,0.);

            for(k=0;k<N;k++){
                sum =  cuCaddf(sum  , cuCmulf(A[i*N+k],B[k*N+j]) );
            }

            out[i*N+j] = sum;
        }
}
//////////////////////////////////////////////////////////////////////

__device__ __host__ static __inline__ void cgeTranspose(cuFloatComplex  *At, cuFloatComplex  *A, int N)
{
    int i,j;
    for( i = 0; i<N; i++)
        for( j = 0; j<N; j++)
            At[i+j*N] = A[i*N+j] ;
}

//////////////////////////////////////////////////////////////////////////////////////
//                LU decomposition of complex matrices
/////////////////////////////////////////////////////////////////////////////////////
__device__ __host__ static __inline__ void cgeDoolittle_LU_Decomposition(cuFloatComplex *LU, cuFloatComplex *A, int n)
{

    int i, j, k, p;
    cuFloatComplex *p_k, *p_row, *p_col;

    for(k=0; k<n*n; k++) 
        LU[k]=A[k];


    for (k = 0, p_k = LU; k < n; p_k += n, k++) {

        for (j = k; j < n; j++) {

            for (p = 0, p_col = LU; p < k; p_col += n,  p++)

                //           *(p_k + j) -= *(p_k + p) * *(p_col + j);
                *(p_k + j) = cuCsubf( *(p_k + j) , cuCmulf( *(p_k + p) , *(p_col + j) ));
        }

        if ( cuCabsf(*(p_k + k)) != 0.0 )  //return -1;


            for (i = k+1, p_row = p_k + n; i < n; p_row += n, i++) {

                for (p = 0, p_col = LU; p < k; p_col += n, p++)

                    // *(p_row + k) -= *(p_row + p) * *(p_col + k);
                    *(p_row + k) = cuCsubf( *(p_row + k) , cuCmulf(  *(p_row + p) , *(p_col + k) ));
                    //*(p_row + k) /= *(p_k + k);


                *(p_row + k) = cuCdivf( *(p_row + k) , *(p_k + k)  );
            }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Back substitution for lower triangular matrices assuming that the diagonal is filled with 1's
// Given  T x = b ,
// where T is a lower triangula matrix NxN with 1's in the diagonal
// b is a known vector
// x is an unknown vector
// the routine otputs x = xout
//////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __host__ static __inline__ void cgeBackSubstitutionLowerDiagOne(cuFloatComplex *xout, cuFloatComplex *T , cuFloatComplex *b ,int N  )
{

    cuFloatComplex bTemp;
    int i,j;

    for(i=0;i<N;i++){

        bTemp = b[i];
        for(j=0;j<i;j++) 
            bTemp = cuCsubf( bTemp , cuCmulf(T[ i*N + j], xout[j] ) );

        xout[i] = bTemp ;    
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// Back substitution for upper triangular matrice
////////////////////////////////////////////////////////////////////////////////////
__device__ __host__ static __inline__ void cgeBackSubstitutionUpper(cuFloatComplex *xout, cuFloatComplex *T , cuFloatComplex *b ,int N  )
{

    cuFloatComplex bTemp;
    int i,j;

    for(i=N-1;i>=0;i--){

        bTemp = b[i];
        for(j=i+1;j<N;j++) 
            bTemp = cuCsubf( bTemp , cuCmulf(T[ i*N + j], xout[j] ) );

        xout[i] = cuCdivf( bTemp , T[ i*N + i] );    
    }
}

///////////////////////
__device__ __host__ static __inline__ void cgeIdentity(cuFloatComplex *Id, int N)
{

    int i,j;
    for(i=0;i<N;i++)

        for(j=0;j<N;j++){
            Id[i*N+j] = make_cuFloatComplex(0.f,0.f);
        }


    for(i=0;i<N;i++){
        Id[i*N+i] = make_cuFloatComplex(1.f,0.f);
    }
}

//////////////////////////////////////////////////////////////////////////////////////
//  Inverse of a matrix using the triangularization of matrices
//  Warning:
//          It does not destroys the original matrix A
//          A work space for 3 complex matrices must be supplied in W
//////////////////////////////////////////////////////////////////////////////////////
__device__ __host__ static __inline__ int cgeMatrixInverse_WorkSpace(){ 
    return 3;
}

__device__ __host__ static __inline__ void cgeMatrixInverse(cuFloatComplex *inv , cuFloatComplex *A , int N , cuFloatComplex *W)
{

    int i;
    cuFloatComplex *Id;
    cuFloatComplex *invL, *invU;


    Id = W;      //Double purpose work space
    invL = W + N*N;

    invU = W + 2*N*N;


    cgeIdentity( Id, N);


    cgeDoolittle_LU_Decomposition(inv,A,N);


    for(i=0;i<N;i++)   
        cgeBackSubstitutionLowerDiagOne( invL + i*N , inv , Id + i*N ,  N  );


    for(i=0;i<N;i++) 
        cgeBackSubstitutionUpper( invU + i*N , inv , Id + i*N ,  N  );


    cgeSquareMatrixProduct(Id , invL , invU , N  );


    cgeTranspose(inv , Id , N);

}


#endif