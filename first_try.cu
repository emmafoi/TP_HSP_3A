// this is the cuda file of TP1
// MULTIPLICATION DE MATRICES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
//pour curandCreateGenerator() pour la génération de nombres random :
#include <curand.h>

//Paramètres : 
// n : nombre de lignes de la matrice
// p : nombre de colonnes de la matrice si n différent de p,
// M : pointeur de la matrice


//CREATION D'UNE MATRICE SUR CPU

//cudaMalloc(void **devPtr, size_t size) : devPtr = Pointer to allocated device memory; size_t = Requested allocation size in bytes

//Cette fonction initialise une matrice de taille n x p
__host__ void MatrixInit(float *M, int n, int p) {
    // Allocate memory
    cudaMalloc((void **) & M, n * p * sizeof(float));
}

//Initialisez les valeurs de la matrice de façon aléatoire entre -1 et 1.
//On commence par créer une fonction qui renvoie un flottant aléatoirement, puis on fait une fonction qui parcourt la matrice initialisée pour lui assigner des valeurs aléatoires.

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int Rand(int argc, char *argv[])
{
    size_t n = 100;
    size_t i;
    curandGenerator_t gen;
    float *devData, *hostData;

    /* Allocate n floats on host */
    hostData = (float *)calloc(n, sizeof(float));

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
        cudaMemcpyDeviceToHost));

    /* Show result */
    for(i = 0; i < n; i++) {
        printf("%1.4f ", hostData[i]);
    }
    printf("\n");

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);    
    return EXIT_SUCCESS;
}

__host__ void MatrixInitRand(float *M, int n, int p) {
    for(int i = 0; i < n*p; i++){
        *M[i] = Rand();
        }
}


int main(){
    float *M;
    int n = 3;
    int p = 5;
    
    MatrixInit(M,n,p);
    
    MatrixInitRand(M,n,p)
    
    // This call waits for all of the submitted GPU work to complete
    cudaDeviceSynchronize();
    
    return 0;
}

    


//AFFICHAGE D'UNE MATRICE SUR CPU

//Cette fonction affiche une matrice de taille n x p
//void MatrixPrint(float *M, int n, int p){}


//ADDITION DE DEUX MATRICES SUR CPU
//void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){}
//__global__ void vector_add(float *out, float *a, float *b, int n) {
//    for(int i = 0; i < n; i++){
//        out[i] = a[i] + b[i];
//    }
//}


//ADDITION DE DEUX MATRICES SUR GPU
//__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){}


// MULTIPLICATION DE DEUX MATRICES NxN SUR CPU
//void MatrixMult(float *M1, float *M2, float *Mout, int n){}


// MULTIPLICATION DE DEUX MATRICES NxN SUR GPU
//__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n){}
