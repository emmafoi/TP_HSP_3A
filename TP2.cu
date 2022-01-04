//Partie 2

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

void MatrixInitRand(float *M, int n);
void MatrixInitZero(float *M, int n);
void MatrixPrint(float *M, int n);
__global__ void cudaConv(float *E, float *F, float *S);
__global__ void cudaMoyen2(float *E, float *F, int n);

// 3.1

void MatrixInitRand(float *M, int n){
    for (int i = 0; i < n; i++){
        M[i] = (float)(rand()%1000)/1000 ; 
        //flottant entre 0 et 1 de précision 10⁻3
    }
}

void MatrixInitZero(float *M, int n){
    for (int i = 0; i < n; i++){
        M[i] = 0 ; 
        //flottant entre 0 et 1 de précision 10⁻3
    }
}

void MatrixPrint(float *M, int n){
    for (int i = 0; i < n ; i++){
        if((i+1)%n ==0){
            printf("%f\n",M[i]);
        }else{
            printf("%f ",M[i]);
        }
    }
}

// 3.2

/* A REPRENDRE*/
__global__ void cudaConv(float *E, float *F, float *S){
    int idx = threadIdx.x;
    S[idx] = E[idx] * F[idx];
}

/* A REPRENDRE*/
__global__ void cudaMoyen2(float *E, float *F, int n){
    // n = taille d'une ligne de la matrice
    int idx = threadIdx.x;
    S[idx] = (float)((E[idx] + E[idx+1] + E[idx + n] + E[idx + n + 1] )/4);
}


int main(){
    
    // 3.1 
    
    //matrice raw_data
    int n1 = 32;
    const int ARRAY_SIZE1 = n1*n1;
    const int ARRAY_BYTES1 = ARRAY_SIZE1 * sizeof(float);
    
    //matrice C1_data
    int n21 = 28;
    int n22 = 6;
    const int ARRAY_SIZE2 = n21*n21*n22;
    const int ARRAY_BYTES2 = ARRAY_SIZE2 * sizeof(float);
    
    //matrice S1_data : issue du sous-échantillonnage de facteur 2 de C1_data
    int n31 = 14;
    int n32 = 6;
    const int ARRAY_SIZE3 = n31*n31*n32;
    const int ARRAY_BYTES3 = ARRAY_SIZE3* sizeof(float);
    
    //matrice C1_kernel : 6 noyaux de conv de taille 5x5
    int n41 = 5;
    int n42 = 6;
    const int ARRAY_SIZE4 = n41*n41*n42;
    const int ARRAY_BYTES4 = ARRAY_SIZE4 * sizeof(float);
    
    //allocation de mémoire pour les matrices sur CPU
    float *raw_data, *C1_data, *S1_data, *C1_kernel;
    raw_data = (float*)malloc(ARRAY_BYTES1);
    C1_data = (float*)malloc(ARRAY_BYTES2);
    S1_data = (float*)malloc(ARRAY_BYTES3);
    C1_kernel = (float*)malloc(ARRAY_BYTES4);
    
     
    //initialisation :
    MatrixInitRand(raw_data, ARRAY_SIZE1);
    MatrixInitZero(C1_data, ARRAY_SIZE2);
    MatrixInitZero(S1_data, ARRAY_SIZE3);
    MatrixInitRand(C1_kernel, ARRAY_SIZE4);
    
    // pour tester :
    //MatrixPrint(C1_data, n21* n21* n22);
    
    
    // 3.2
       
    //allocation de mémoire sur GPU
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;
    cudaMalloc((void **) &d_raw_data, ARRAY_BYTES1);
    cudaMalloc((void **) &d_C1_data, ARRAY_BYTES2);
    cudaMalloc((void **) &d_S1_data, ARRAY_BYTES3);
    cudaMalloc((void **) &d_C1_kernel, ARRAY_BYTES4);
    
    //transfert de données pour le calcul sur gpu
    //entrée:
    cudaMemcpy(d_raw_data, raw_data, ARRAY_BYTES1, cudaMemcpyHostToDevice);
    //sortie:
    cudaMemcpy(d_C1_data, C1_data, ARRAY_BYTES2, cudaMemcpyHostToDevice);
    //filtre:
    cudaMemcpy(d_C1_kernel, C1_kernel, ARRAY_BYTES4, cudaMemcpyHostToDevice);
    
    // Layer 2 : convolution
    cudaConv<<<n21, n21, n22>>>(d_raw_data,d_C1_data, d_C1_kernel);
    
    //récupération des données sur le cpu
    cudaMemcpy(C1_data, d_C1_data, ARRAY_BYTES2, cudaMemcpyDeviceToHost);
    
    
    // Layer 3 : moyenneur
    cudaMoyen2<<<n31, n31, n32>>>(d_C1_data,d_S1_data, n31);
    
    
    
    //libération des ressources 
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C1_kernel);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    
    // This call waits for all of the submitted GPU work to complete
    cudaDeviceSynchronize();
    
    return 0;
}